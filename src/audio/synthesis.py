import os
import sys
import io
import json
import time
from datetime import datetime
import numpy as np
import requests
import sounddevice
import soundfile
import ffmpeg
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ..models.voice import VoiceParams, VoiceStyle
from ..models.constants import (
    AIVIS_BASE_URL,
    DEFAULT_OUTPUT_SAMPLING_RATE,
    EMOTION_SCORE_THRESHOLD,
    AIVIS_PATH
)

def ensure_aivis_server(url: str) -> Tuple[bool, str]:
    """AivisSpeech-Engineの状態を確認する
    
    Args:
        url: サーバーのURL
        
    Returns:
        (bool, str): 成功したかどうかとメッセージのタプル
    """
    success = False
    try:
        response = requests.get(f"{url}/version")
        if response.status_code == 200:
            return True, "AivisSpeech-Engineに接続しました。"
        return False, "AivisSpeech-Engineが応答しません。"
    except requests.exceptions.RequestException:    
    
    # サーバーが応答しない場合、Aivis Engineを起動
        try:
            exe_path = AIVIS_PATH
            if os.path.exists(exe_path):
                subprocess.Popen(exe_path)
                print("Aivis Engineを起動しています...")
                time.sleep(10)  # エンジンの起動を待つ
                
                # 再度サーバーの状態を確認
                response = requests.get(f"{url}/version")
                if response.status_code == 200:
                    return True, "Aivis Engineが正常に起動しました。"
                else:
                    return False, "Aivis Engineの起動に失敗しました。"
            else:
                return False, "Aivis Engineの実行ファイルが見つかりません。"
        except Exception as e:
            return False, f"Aivis Engineの起動中にエラーが発生しました: {str(e)}"

class AivisAdapter:
    """SentioVoxの音声合成アダプター
    
    感情分析の結果に基づいて、AIVISエンジンを用いた感情豊かな音声合成を実現します。
    各感情に対応する音声パラメータを適切に調整し、自然で表現力豊かな音声を生成します。
    
    生成した音声は、再生だけでなくファイルとしての保存にも対応し、
    WAVやM4A形式でエクスポートすることができます。
    """
    def __init__(self):
        self.URL = AIVIS_BASE_URL
        
        # AivisSpeech-Engineの状態を確認
        success, message = ensure_aivis_server(self.URL)
        if not success:
            print(f"\nエラー: {message}")
            print("音声合成を利用するには、AivisSpeech-Engineが必要です。")
            sys.exit(1)
            
        print(f"\n{message}")  # 成功メッセージを表示
        
        # 各感情スタイルに対応する音声パラメータの設定
        self.voice_parameters = {
            VoiceStyle.NORMAL: VoiceParams(
                888753761, 1.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.1),
            VoiceStyle.JOY: VoiceParams(
                888753764, 1.2, 1.15, 1.1, 0.03, 1.1, 0.1, 0.1),
            VoiceStyle.SADNESS: VoiceParams(
                888753765, 0.7, 0.85, 0.9, -0.02, 0.9, 0.2, 0.1),
            VoiceStyle.ANTICIPATION: VoiceParams(
                888753762, 1.05, 1.1, 1.05, 0.02, 1.05, 0.1, 0.1),
            VoiceStyle.SURPRISE: VoiceParams(
                888753762, 1.3, 1.2, 1.15, 0.05, 1.2, 0.1, 0.1),
            VoiceStyle.ANGER: VoiceParams(
                888753765, 1.3, 1.2, 1.05, 0.04, 1.3, 0.1, 0.1),
            VoiceStyle.FEAR: VoiceParams(
                888753763, 1.1, 1.1, 1.1, 0.03, 0.9, 0.2, 0.1),
            VoiceStyle.DISGUST: VoiceParams(
                888753765, 1.15, 1.05, 0.95, 0.02, 1.1, 0.2, 0.1),
            VoiceStyle.TRUST: VoiceParams(
                888753763, 1.02, 1.0, 0.95, 0.01, 1.0, 0.1, 0.1)
        }

    def calculate_mixed_parameters(
        self,
        emotion_scores: Dict[VoiceStyle, float]
    ) -> Tuple[int, Dict[str, float]]:
        """複数の感情スコアを考慮してパラメータを混合"""
        # float32をfloatに変換
        emotion_scores = {k: float(v) for k, v in emotion_scores.items()}
        
        total_score = sum(emotion_scores.values())
        if total_score == 0:
            return self.voice_parameters[VoiceStyle.NORMAL].style_id, {}

        # 最も強い感情を特定
        dominant_emotion = max(
            emotion_scores.items(),
            key=lambda x: x[1]
        )[0]
        style_id = self.voice_parameters[dominant_emotion].style_id

        # パラメータの混合処理
        mixed_params = {}
        param_names = [
            'intonationScale',
            'tempoDynamicsScale',
            'speedScale',
            'pitchScale',
            'volumeScale',
            'prePhonemeLength',
            'postPhonemeLength'
        ]
        for param_name in param_names:
            mixed_params[param_name] = 0.0

        # 各感情のウェイトに基づいてパラメータを混合
        for emotion, score in emotion_scores.items():
            weight = score / total_score
            params = self.voice_parameters[emotion].scale_params(weight)
            for key in mixed_params:
                mixed_params[key] += params[key]

        # すべての値をfloatに変換
        mixed_params = {k: float(v) for k, v in mixed_params.items()}
        return style_id, mixed_params

    def speak_continuous(
        self,
        segments: List[str],
        emotion_scores_list: List[List[float]],
        save_path: Optional[str] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """連続的な音声合成を実行し、必要に応じてファイルに保存
        
        Args:
            segments: テキストセグメントのリスト
            emotion_scores_list: 感情スコアのリスト
            save_path: 保存先のパス（指定がない場合は一時ファイルを使用）
            play_audio: 音声を再生するかどうか
            
        Returns:
            保存したファイルのパス（save_pathが指定されている場合）
        """
        # 全セグメントの音声合成
        combined_audio = None
        rate = None
        
        for text, scores in zip(segments, emotion_scores_list):
            if not text.endswith('。'):
                text += '。'
            
            style_id, params = self.calculate_mixed_parameters(
                self._convert_scores_to_dict(scores)
            )
            
            # 音声合成の実行
            segment_audio = self._synthesize_segment(text, style_id, params)
            if segment_audio is not None:
                audio_data, current_rate = segment_audio
                if combined_audio is None:
                    combined_audio = audio_data
                    rate = current_rate
                else:
                    combined_audio = np.concatenate([combined_audio, audio_data])

        if combined_audio is None:
            print("警告: 音声合成に失敗しました")
            return None

        # 保存パスの決定
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"output_{timestamp}.m4a"

        # WAVファイルとして一時保存
        temp_wav = Path(save_path).with_suffix('.wav')
        soundfile.write(str(temp_wav), combined_audio, rate)

        # M4Aに変換
        try:
            stream = ffmpeg.input(str(temp_wav))
            stream = ffmpeg.output(
                stream,
                str(save_path),
                acodec='aac',
                audio_bitrate='192k'
            )
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            print(f"音声ファイルを保存しました: {save_path}")
        except ffmpeg.Error as e:
            print(f"M4Aへの変換に失敗しました: {str(e)}")
            save_path = str(temp_wav)  # WAVファイルを代替として使用
        finally:
            if temp_wav.exists() and save_path != str(temp_wav):
                temp_wav.unlink()  # 一時WAVファイルを削除

        # 音声の再生（オプション）
        if play_audio:
            sounddevice.play(combined_audio, rate)
            sounddevice.wait()

        return save_path

    def _synthesize_segment(
        self,
        text: str,
        style_id: int,
        params: Dict[str, float]
    ) -> Optional[Tuple[np.ndarray, int]]:
        """単一のテキストセグメントを合成"""
        try:
            # クエリパラメータの設定
            query_params = {
                "text": text,
                "speaker": style_id,
                "outputSamplingRate": DEFAULT_OUTPUT_SAMPLING_RATE,
                "outputStereo": False,
            }

            # 音声クエリの生成
            query_response = requests.post(
                f"{self.URL}/audio_query",
                params=query_params
            ).json()

            # パラメータの適用
            query_response.update(params)
            query_response.update({
                "volumeScale": 1.2,
                "prePhonemeLength": 0.1,
                "postPhonemeLength": 0.1,
            })

            # 音声合成の実行
            audio_response = requests.post(
                f"{self.URL}/synthesis",
                params={"speaker": style_id},
                headers={"accept": "audio/wav", "Content-Type": "application/json"},
                data=json.dumps(query_response)
            )

            # 音声データの処理
            with io.BytesIO(audio_response.content) as stream:
                audio_data, rate = soundfile.read(stream)
                return audio_data, rate

        except Exception as e:
            print(f"セグメントの合成中にエラーが発生しました: {str(e)}")
            return None
        
    def _convert_scores_to_dict(
        self,
        scores: List[float]
    ) -> Dict[VoiceStyle, float]:
        """感情スコアの配列を辞書形式に変換"""
        emotion_dict = {}
        for emotion, score in zip([
            "喜び", "悲しみ", "期待", "驚き",
            "怒り", "恐れ", "嫌悪", "信頼"
        ], scores):
            if score >= EMOTION_SCORE_THRESHOLD:
                style = self.map_emotion_to_voice_style(emotion)
                emotion_dict[style] = float(score)
        
        if not emotion_dict:
            emotion_dict[VoiceStyle.NORMAL] = 1.0
            
        return emotion_dict

    def map_emotion_to_voice_style(self, emotion: str) -> VoiceStyle:
        """感情をボイススタイルにマッピング"""
        emotion_to_style = {
            "喜び": VoiceStyle.JOY,
            "悲しみ": VoiceStyle.SADNESS,
            "期待": VoiceStyle.ANTICIPATION,
            "驚き": VoiceStyle.SURPRISE,
            "怒り": VoiceStyle.ANGER,
            "恐れ": VoiceStyle.FEAR,
            "嫌悪": VoiceStyle.DISGUST,
            "信頼": VoiceStyle.TRUST
        }
        return emotion_to_style.get(emotion, VoiceStyle.NORMAL)