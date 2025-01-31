import sys
import io
import json
import numpy as np
import requests
import sounddevice
import soundfile
from typing import Dict, List, Tuple
from ..models.voice import VoiceParams, VoiceStyle
from ..models.constants import (
    AIVIS_BASE_URL,
    DEFAULT_OUTPUT_SAMPLING_RATE,
    EMOTION_SCORE_THRESHOLD
)
from ..utils.aivis_utils import ensure_aivis_server


class AivisAdapter:
    """SentioVoxの音声合成アダプター
    
    感情分析の結果に基づいて、AIVISエンジンを用いた感情豊かな音声合成を実現します。
    各感情に対応する音声パラメータ（イントネーション、テンポ、ピッチなど）を
    適切に調整し、自然で表現力豊かな音声を生成します。
    
    複数の感情が検出された場合は、それらを適切に混合し、
    より微妙な感情表現を可能にします。
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
        emotion_scores_list: List[List[float]]
    ) -> None:
        """連続的な音声合成を実行"""
        # 全セグメントとそれに対応する感情スコアの処理
        combined_params = []
        for text, scores in zip(segments, emotion_scores_list):
            if not text.endswith('。'):
                text += '。'
            
            style_id, params = self.calculate_mixed_parameters(
                self._convert_scores_to_dict(scores)
            )
            combined_params.append({
                'text': text,
                'style_id': style_id,
                'params': params
            })

        # 音声合成の実行
        combined_audio = None
        rate = None
        
        for params in combined_params:
            # クエリパラメータの設定
            query_params = {
                "text": params['text'],
                "speaker": params['style_id'],
                "outputSamplingRate": DEFAULT_OUTPUT_SAMPLING_RATE,
                "outputStereo": False,
            }

            # 音声クエリの生成
            query_response = requests.post(
                f"{self.URL}/audio_query",
                params=query_params
            ).json()

            # パラメータの適用
            query_response.update(params['params'])
            query_response.update({
                "volumeScale": 1.2,
                "prePhonemeLength": 0.1,
                "postPhonemeLength": 0.1,
            })

            # 音声合成の実行
            audio_response = requests.post(
                f"{self.URL}/synthesis",
                params={"speaker": params['style_id']},
                headers={"accept": "audio/wav", "Content-Type": "application/json"},
                data=json.dumps(query_response)
            )

            # 音声データの処理
            with io.BytesIO(audio_response.content) as stream:
                segment_data, current_rate = soundfile.read(stream)
                if combined_audio is None:
                    combined_audio = segment_data
                    rate = current_rate
                else:
                    combined_audio = np.concatenate([combined_audio, segment_data])

        # 結合した音声の再生
        if combined_audio is not None:
            sounddevice.play(combined_audio, rate)
            sounddevice.wait()

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