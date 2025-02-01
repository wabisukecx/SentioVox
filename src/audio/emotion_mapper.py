"""感情と音声パラメータのマッピングを担当するモジュール

このモジュールは感情分析の結果を音声合成パラメータに変換する役割を
担います。各感情に対応する適切な音声特性を定義し、それらを組み合わせて
自然な感情表現を実現します。
"""

from typing import Dict, List, Tuple
from ..models.voice import VoiceParams, VoiceStyle
from ..models.constants import EMOTION_SCORE_THRESHOLD

class EmotionVoiceMapper:
    """感情から音声パラメータへのマッピングを行うクラス
    
    このクラスは、感情スコアを解析して適切な音声パラメータに
    変換します。複数の感情が存在する場合は、それらを適切に
    組み合わせて自然な音声表現を実現します。
    """
    
    def __init__(self):
        """音声パラメータの初期化
        
        各感情スタイルに対応する基本的な音声パラメータを設定します。
        これらのパラメータは経験的に調整された値であり、自然な
        感情表現を実現するように設計されています。
        """
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

    def convert_scores_to_dict(
        self,
        scores: List[float]
    ) -> Dict[VoiceStyle, float]:
        """感情スコアの配列を辞書形式に変換
        
        8つの基本感情に対応するスコアを受け取り、閾値以上の
        スコアを持つ感情のみを抽出して音声スタイルとマッピングします。
        
        Args:
            scores: 感情スコアの配列（8つの基本感情に対応）
            
        Returns:
            Dict[VoiceStyle, float]: 感情スタイルと強度のマッピング
        
        Note:
            スコアが閾値未満の感情は除外され、有効な感情が
            存在しない場合は通常スタイルが設定されます。
        """
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
        """感情をボイススタイルにマッピング
        
        テキストで表現された感情を、対応する音声スタイルに
        変換します。未知の感情の場合は通常スタイルを返します。
        
        Args:
            emotion: 感情を表す文字列
            
        Returns:
            VoiceStyle: 対応する音声スタイル
        """
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

    def calculate_mixed_parameters(
        self,
        emotion_scores: Dict[VoiceStyle, float]
    ) -> Tuple[int, Dict[str, float]]:
        """複数の感情スコアを考慮してパラメータを混合
        
        各感情の強度に基づいて音声パラメータを適切に混合し、
        より自然な感情表現を実現します。支配的な感情を基準に
        スタイルIDを決定し、他の感情の影響を重み付けして
        パラメータを調整します。
        
        Args:
            emotion_scores: 感情スタイルと強度のマッピング
            
        Returns:
            Tuple[int, Dict[str, float]]: スタイルIDとパラメータの辞書
            
        Note:
            パラメータは各感情の強度に応じて重み付けされ、
            最終的なパラメータは全ての有効な感情の影響を
            反映します。
        """
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
        mixed_params = {
            'intonationScale': 0.0,
            'tempoDynamicsScale': 0.0,
            'speedScale': 0.0,
            'pitchScale': 0.0,
            'volumeScale': 0.0,
            'prePhonemeLength': 0.0,
            'postPhonemeLength': 0.0
        }

        # 各感情のウェイトに基づいてパラメータを混合
        for emotion, score in emotion_scores.items():
            weight = score / total_score
            params = self.voice_parameters[emotion].scale_params(weight)
            for key in mixed_params:
                mixed_params[key] += params[key]

        # すべての値をfloatに確実に変換
        mixed_params = {k: float(v) for k, v in mixed_params.items()}
        return style_id, mixed_params
