"""JSON形式の会話データから音声合成を行うモジュール

感情情報を含むJSONデータから、適切な話者マッピングと感情パラメータを
使用して高品質な音声合成を実行します。
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests
import json
import base64
from pathlib import Path

from ..models.constants import AIVIS_BASE_URL


class JsonSynthesisAdapter:
    """JSONデータを使用して音声合成を行うアダプタークラス
    
    このクラスは以下の責任を持ちます：
    - JSONデータに基づく音声合成リクエストの作成
    - 話者とキャラクターのマッピング処理
    - 感情に基づくパラメータの調整
    - 合成音声の保存と再生
    """
    
    def __init__(self, base_url: str = AIVIS_BASE_URL):
        """初期化
        
        Args:
            base_url: AIVISサーバーのベースURL
        """
        self.base_url = base_url
    
    def synthesize_dialogue(
        self,
        dialogue_data: List[Dict],
        character_mapping: Dict[str, int],
        emotion_mapping: Optional[Dict[str, Dict[str, int]]] = None,
        emotion_params: Optional[Dict[str, Dict[str, float]]] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        progress_callback=None
    ) -> List[Dict]:
        """会話データから音声を合成
        
        Args:
            dialogue_data: 会話データ
            character_mapping: キャラクターと話者IDのマッピング
            emotion_mapping: キャラクターの感情と話者IDのマッピング
            emotion_params: 感情ごとのパラメータ調整
            start_index: 開始インデックス
            end_index: 終了インデックス
            progress_callback: 進捗を報告するコールバック関数
            
        Returns:
            List[Dict]: 合成された音声データと関連情報のリスト
        """
        if end_index is None:
            end_index = len(dialogue_data) - 1
        
        # 処理範囲のバリデーション
        start_index = max(0, start_index)
        end_index = min(len(dialogue_data) - 1, end_index)
        
        # 感情マッピングとパラメータが未指定の場合は空の辞書を使用
        emotion_mapping = emotion_mapping or {}
        emotion_params = emotion_params or {}
        
        audio_results = []
        total_items = end_index - start_index + 1
        
        for i, idx in enumerate(range(start_index, end_index + 1)):
            dialogue = dialogue_data[idx]
            
            # 進捗報告
            if progress_callback:
                progress = i / total_items
                progress_callback(progress, i, total_items, dialogue)
            
            character = dialogue["speaker"]
            text = dialogue["text"]
            emotion = dialogue.get("dominant_emotion", "")
            
            # 話者IDを取得
            speaker_id = self._get_speaker_id(character, emotion, character_mapping, emotion_mapping)
            
            if speaker_id is None:
                print(f"警告: {character}の話者IDが見つかりません。このセグメントはスキップされます。")
                continue
            
            try:
                # 音声合成の実行
                audio_data, params = self._synthesize_segment(
                    text, speaker_id, emotion, emotion_params
                )
                
                if audio_data:
                    audio_results.append({
                        "index": idx,
                        "character": character,
                        "text": text,
                        "emotion": emotion,
                        "speaker_id": speaker_id,
                        "audio_data": audio_data,
                        "params": params
                    })
                    
                # 連続リクエストを避けるための短い遅延
                time.sleep(0.1)
                
            except Exception as e:
                print(f"エラー: セグメント {idx} の処理中にエラーが発生しました: {str(e)}")
        
        # 最終進捗報告
        if progress_callback:
            progress_callback(1.0, total_items, total_items, None)
        
        return audio_results
    
    def _get_speaker_id(
        self,
        character: str,
        emotion: str,
        character_mapping: Dict[str, int],
        emotion_mapping: Dict[str, Dict[str, int]]
    ) -> Optional[int]:
        """キャラクターと感情に基づいて適切な話者IDを取得
        
        Args:
            character: キャラクター名
            emotion: 感情名
            character_mapping: キャラクターと話者IDのマッピング
            emotion_mapping: キャラクターの感情と話者IDのマッピング
            
        Returns:
            Optional[int]: 話者ID、見つからない場合はNone
        """
        # 感情マッピングをチェック
        if (emotion and character in emotion_mapping and 
            emotion in emotion_mapping[character]):
            return emotion_mapping[character][emotion]
        
        # キャラクターマッピングをチェック
        if character in character_mapping:
            return character_mapping[character]
        
        # マッピングが見つからない場合
        return None
    
    def _synthesize_segment(
        self,
        text: str,
        speaker_id: int,
        emotion: str,
        emotion_params: Dict[str, Dict[str, float]]
    ) -> Tuple[Optional[bytes], Dict]:
        """単一テキストの音声合成を実行
        
        Args:
            text: 合成するテキスト
            speaker_id: 話者ID
            emotion: 感情名
            emotion_params: 感情ごとのパラメータ調整
            
        Returns:
            Tuple[Optional[bytes], Dict]: 音声データとパラメータ情報
        """
        try:
            # 音声クエリの作成
            response = requests.post(
                f"{self.base_url}/audio_query",
                params={"text": text, "speaker": speaker_id}
            )
            
            if response.status_code != 200:
                print(f"警告: 音声クエリの作成に失敗しました: {response.status_code}")
                return None, {}
            
            query = response.json()
            
            # 感情に基づくパラメータ調整
            if emotion and emotion in emotion_params:
                params = emotion_params[emotion]
                
                # 各パラメータの調整
                if "speedScale" in params:
                    query["speedScale"] = max(0.5, min(2.0, query["speedScale"] * params["speedScale"]))
                if "pitchScale" in params:
                    query["pitchScale"] = max(-0.15, min(0.15, query["pitchScale"] + params["pitchScale"]))
                if "intonationScale" in params:
                    query["intonationScale"] = max(0.0, min(2.0, query["intonationScale"] * params["intonationScale"]))
                if "volumeScale" in params:
                    query["volumeScale"] = max(0.0, min(2.0, query["volumeScale"] * params["volumeScale"]))
            
            # 音声合成の実行
            synth_response = requests.post(
                f"{self.base_url}/synthesis",
                headers={"Content-Type": "application/json"},
                params={"speaker": speaker_id},
                json=query
            )
            
            if synth_response.status_code != 200:
                print(f"警告: 音声合成に失敗しました: {synth_response.status_code}")
                return None, query
            
            return synth_response.content, query
            
        except Exception as e:
            print(f"エラー: 音声合成中に例外が発生しました: {str(e)}")
            return None, {}
    
    def save_audio_files(
        self,
        audio_results: List[Dict],
        output_dir: str
    ) -> List[str]:
        """合成した音声ファイルを保存
        
        Args:
            audio_results: 合成された音声データと関連情報のリスト
            output_dir: 出力ディレクトリ
            
        Returns:
            List[str]: 保存されたファイルパスのリスト
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for item in audio_results:
            if not item["audio_data"]:
                continue
                
            filename = f"{item['index']:04d}_{item['character']}_{item['emotion']}.wav"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'wb') as f:
                    f.write(item["audio_data"])
                saved_files.append(filepath)
            except Exception as e:
                print(f"エラー: ファイル保存中に例外が発生しました: {str(e)}")
        
        return saved_files
    
    def connect_audio_files(
        self,
        audio_results: List[Dict]
    ) -> Optional[bytes]:
        """合成した音声ファイルを連結
        
        Args:
            audio_results: 合成された音声データと関連情報のリスト
            
        Returns:
            Optional[bytes]: 連結された音声データ
        """
        try:
            # 音声データが存在するアイテムだけを抽出
            valid_items = [item for item in audio_results if item["audio_data"]]
            
            if not valid_items:
                return None
            
            # Base64でエンコードした音声データのリストを作成
            encoded_waves = [
                base64.b64encode(item["audio_data"]).decode('utf-8')
                for item in valid_items
            ]
            
            # APIを使って音声を連結
            response = requests.post(
                f"{self.base_url}/connect_waves",
                json=encoded_waves
            )
            
            if response.status_code != 200:
                print(f"警告: 音声の連結に失敗しました: {response.status_code}")
                return None
            
            return response.content
            
        except Exception as e:
            print(f"エラー: 音声連結中に例外が発生しました: {str(e)}")
            return None
    
    def get_speakers(self) -> List[Dict]:
        """利用可能な話者情報を取得
        
        Returns:
            List[Dict]: 話者情報のリスト
        """
        try:
            response = requests.get(f"{self.base_url}/speakers")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"警告: スピーカー情報の取得に失敗しました: {response.status_code}")
                return []
        except Exception as e:
            print(f"エラー: API接続エラー: {str(e)}")
            return []