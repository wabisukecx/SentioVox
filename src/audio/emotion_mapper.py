"""感情と音声パラメータのマッピングを担当するモジュール

このモジュールは感情分析の結果を音声合成パラメータに変換する役割を
担います。各感情に対応する適切な音声特性を定義し、それらを組み合わせて
自然な感情表現を実現します。
"""

from typing import Dict, List, Tuple
from ..models.voice import VoiceParams, VoiceStyle
from ..models.constants import (
    EMOTION_SCORE_THRESHOLD,
    VOICE_STYLE_IDS,
    VOICE_PARAMS
)


class EmotionVoiceMapper:
    """感情から音声パラメータへのマッピングを行うクラス
    
    このクラスは、感情スコアを解析して適切な音声パラメータに
    変換します。複数の感情が存在する場合は、それらを適切に
    組み合わせて自然な音声表現を実現します。
    """
    
    def __init__(self):
        """音声パラメータの初期化
        
        各感情スタイルに対応する基本的な音声パラメータを設定します。
        パラメータ値はconstants.pyで一元管理されています。
        """
        self.voice_parameters = {
            VoiceStyle.NORMAL: self._create_voice_params('NORMAL'),
            VoiceStyle.JOY: self._create_voice_params('JOY'),
            VoiceStyle.SADNESS: self._create_voice_params('SADNESS'),
            VoiceStyle.ANTICIPATION: self._create_voice_params('ANTICIPATION'),
            VoiceStyle.SURPRISE: self._create_voice_params('SURPRISE'),
            VoiceStyle.ANGER: self._create_voice_params('ANGER'),
            VoiceStyle.FEAR: self._create_voice_params('FEAR'),
            VoiceStyle.DISGUST: self._create_voice_params('DISGUST'),
            VoiceStyle.TRUST: self._create_voice_params('TRUST')
        }

    def _create_voice_params(self, style_name: str) -> VoiceParams:
        """VoiceParamsオブジェクトを生成
        
        定数から音声パラメータを取得し、VoiceParamsオブジェクトを
        生成します。
        
        Args:
            style_name: 感情スタイル名
            
        Returns:
            VoiceParams: 生成されたパラメータオブジェクト
        """
        params = VOICE_PARAMS[style_name]
        return VoiceParams(
            style_id=VOICE_STYLE_IDS[style_name],
            intonation_scale=params['intonation_scale'],
            tempo_dynamics_scale=params['tempo_dynamics_scale'],
            speed_scale=params['speed_scale'],
            pitch_scale=params['pitch_scale'],
            volume_scale=params['volume_scale'],
            pre_phoneme_length=params['pre_phoneme_length'],
            post_phoneme_length=params['post_phoneme_length']
        )

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
        より自然な感情表現を実現します。
        
        Args:
            emotion_scores: 感情スタイルと強度のマッピング
            
        Returns:
            Tuple[int, Dict[str, float]]: スタイルIDとパラメータの辞書
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