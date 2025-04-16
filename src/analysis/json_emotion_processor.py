"""JSONフォーマットの会話データに感情分析結果を追加するモジュール

このモジュールは、speakerとtextだけを含むJSONファイルに対して感情分析を実行し、
結果を追加するための機能を提供します。emotion_analyzer.pyの機能を活用して、
会話テキストから8つの基本感情を検出します。
"""

import json
from typing import List, Dict, Optional
import numpy as np

from .emotion import EmotionAnalyzer
from .json_dialogue import JsonDialogueProcessor
from ..models.constants import EMOTION_LABELS, EMOTION_SCORE_THRESHOLD


class JsonEmotionProcessor:
    """JSONフォーマットの会話データに感情分析結果を追加するクラス
    
    このクラスは以下の責任を持ちます：
    - JSONファイルの読み込みと検証
    - テキストの感情分析の実行
    - 感情分析結果のJSONへの追加
    - 結果のJSONファイルへの保存
    """
    
    def __init__(self):
        """初期化処理"""
        self.dialogue_processor = JsonDialogueProcessor()
        self.emotion_analyzer = EmotionAnalyzer()
    
    def process_json_data(self, json_data: List[Dict]) -> List[Dict]:
        """会話データの感情分析を実行し、結果を追加
        
        Args:
            json_data: 処理するJSONデータ
            
        Returns:
            List[Dict]: 感情分析結果が追加されたJSONデータ
        """
        # JSONデータの検証
        if not self.dialogue_processor.validate_json_format(json_data):
            raise ValueError("無効なJSONデータ形式です")
        
        # テキストの抽出
        texts = [item["text"] for item in json_data]
        
        # 感情分析の実行
        print(f"\n{len(texts)}個のテキストに対して感情分析を実行します...")
        emotion_scores = self.emotion_analyzer.analyze_emotions(texts)
        
        # 分析結果をJSONデータに追加
        for i, scores in enumerate(emotion_scores):
            # 感情スコアを辞書形式に変換
            emotion_results = self._format_emotion_results(scores)
            json_data[i]["emotions"] = emotion_results
            
            # 最も強い感情を dominant_emotion として追加
            if np.any(scores):
                dominant_idx = scores.argmax()
                json_data[i]["dominant_emotion"] = EMOTION_LABELS[dominant_idx]
            else:
                json_data[i]["dominant_emotion"] = "中立"
        
        print(f"感情分析が完了しました。{len(json_data)}個のアイテムが処理されました。")
        return json_data
    
    def _format_emotion_results(self, scores: List[float]) -> Dict[str, float]:
        """感情スコアを辞書形式にフォーマット
        
        Args:
            scores: 感情スコアのリスト
            
        Returns:
            Dict[str, float]: 閾値以上のスコアを持つ感情とスコアの辞書
        """
        emotion_dict = {}
        for emotion, score in zip(EMOTION_LABELS, scores):
            if score >= EMOTION_SCORE_THRESHOLD:
                emotion_dict[emotion] = float(score)
        
        # 感情が検出されなかった場合は「中立」としてマーク
        if not emotion_dict:
            emotion_dict["中立"] = 1.0
            
        return emotion_dict
    
    def process_json_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """JSONファイルを処理して感情分析結果を追加
        
        Args:
            input_file: 入力JSONファイルのパス
            output_file: 出力JSONファイルのパス（Noneの場合は自動生成）
            
        Returns:
            str: 出力JSONファイルのパス
        """
        # デフォルトの出力ファイル名を設定
        if output_file is None:
            output_file = input_file.replace('.json', '_with_emotions.json')
            
        # JSONファイルの読み込み
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                print(f"{len(json_data)}件の会話データを読み込みました")
        except Exception as e:
            raise RuntimeError(f"JSONファイルの読み込みに失敗しました: {str(e)}")
        
        # データ処理
        processed_data = self.process_json_data(json_data)
        
        # 結果の保存
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
                print(f"感情分析結果を追加したデータを {output_file} に保存しました")
        except Exception as e:
            raise RuntimeError(f"JSONファイルの保存に失敗しました: {str(e)}")
        
        return output_file
    
    def analyze_sample(self, json_data: List[Dict], sample_size: int = 5) -> None:
        """分析結果のサンプルを表示
        
        Args:
            json_data: 分析結果を含むJSONデータ
            sample_size: 表示するサンプル数
        """
        if not json_data:
            return
            
        print("\n感情分析結果のサンプル:")
        for i, item in enumerate(json_data[:sample_size]):
            print("-" * 50)
            print(f"話者: {item['speaker']}")
            print(f"テキスト: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
            print(f"主要な感情: {item['dominant_emotion']}")
            print("検出された感情:")
            
            # 検出された感情を降順で表示
            for emotion, score in sorted(item["emotions"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score:.3f}")