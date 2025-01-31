from dataclasses import dataclass
from enum import Enum


@dataclass
class VoiceParams:
    """音声パラメータを管理するデータクラス"""
    style_id: int                  # スタイルID
    intonation_scale: float        # イントネーションのスケール
    tempo_dynamics_scale: float    # テンポのダイナミクススケール
    speed_scale: float            # 速度スケール
    pitch_scale: float            # ピッチスケール
    volume_scale: float           # 音量スケール
    pre_phoneme_length: float     # 音素前の長さ
    post_phoneme_length: float    # 音素後の長さ

    def scale_params(self, weight: float) -> dict:
        """パラメータをウェイトに基づいてスケーリング"""
        return {
            'intonationScale': float(self.intonation_scale * weight),
            'tempoDynamicsScale': float(self.tempo_dynamics_scale * weight),
            'speedScale': float(self.speed_scale * weight),
            'pitchScale': float(self.pitch_scale * weight),
            'volumeScale': float(self.volume_scale * weight),
            'prePhonemeLength': float(self.pre_phoneme_length * weight),
            'postPhonemeLength': float(self.post_phoneme_length * weight)
        }


class VoiceStyle(Enum):
    """音声スタイルの列挙型"""
    NORMAL = "通常"
    JOY = "喜び"
    SADNESS = "悲しみ"
    ANTICIPATION = "期待"
    SURPRISE = "驚き"
    ANGER = "怒り"
    FEAR = "恐れ"
    DISGUST = "嫌悪"
    TRUST = "信頼"