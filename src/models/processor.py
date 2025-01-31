"""音声データの処理を担当するモジュール

このモジュールは音声データの低レベル処理を担当し、以下の機能を提供します：
- 音声データからの不要な無音部分の除去
- 複数の音声セグメントの結合
- 音声品質の最適化

すべての処理は NumPy を使用して効率的に実装されています。
"""

import numpy as np
from typing import List
from ..models.constants import (
    SILENCE_THRESHOLD,
    MARGIN_SAMPLES,
    SILENCE_DURATION,
    TARGET_DB,
    FADE_SAMPLES
)

class AudioProcessor:
    """音声データの処理を行うクラス
    
    このクラスは音声データに対する様々な処理を提供します。
    すべてのメソッドは静的メソッドとして実装され、
    インスタンス状態に依存しない純粋な関数として機能します。
    """
    
    @staticmethod
    def trim_silence(
        audio_data: np.ndarray,
        threshold: float = SILENCE_THRESHOLD,
        margin_samples: int = MARGIN_SAMPLES
    ) -> np.ndarray:
        """音声データから不要な無音部分をトリミング
        
        音声信号の振幅が指定された閾値以下の部分を無音と判定し、
        前後の無音部分を除去します。急激な切り替えを避けるため、
        無音部分の境界付近に少量のサンプルを残します。
        
        Args:
            audio_data: 音声データ配列
            threshold: 無音と判定する振幅の閾値
            margin_samples: 無音部分の前後に残すサンプル数
            
        Returns:
            np.ndarray: トリミング後の音声データ
        """
        # 音声の振幅を計算
        amplitude = np.abs(audio_data)
        
        # 先頭の無音をトリミング
        start = 0
        for i in range(len(amplitude)):
            if amplitude[i] > threshold:
                start = max(0, i - margin_samples)
                break
        
        # 末尾の無音をトリミング
        end = len(amplitude)
        for i in range(len(amplitude) - 1, -1, -1):
            if amplitude[i] > threshold:
                end = min(len(amplitude), i + margin_samples)
                break
        
        return audio_data[start:end]

    @staticmethod
    def combine_segments_with_silence(
        segments: List[np.ndarray],
        silence_duration: int = SILENCE_DURATION
    ) -> np.ndarray:
        """音声セグメントを適切な無音区間を挿入して結合
        
        複数の音声セグメントを結合する際に、自然な間隔となるように
        一定の無音区間を挿入します。無音の長さは日本語の自然な
        間隔を考慮して設定されています。
        
        Args:
            segments: 音声セグメントのリスト
            silence_duration: セグメント間の無音の長さ（サンプル数）
            
        Returns:
            np.ndarray: 結合された音声データ
            
        Note:
            silence_durationのデフォルト値は約0.033秒の無音に相当し、
            日本語の一般的な文章の間隔として自然な長さです。
        """
        if not segments:
            return np.array([])
            
        # 固定長の無音データを生成
        silence = np.zeros(silence_duration)
        
        # セグメント間に無音を挿入しながら結合
        combined = []
        for i, segment in enumerate(segments):
            if i > 0:
                combined.append(silence)
            combined.append(segment)
        
        return np.concatenate(combined)

    @staticmethod
    def normalize_audio(
        audio_data: np.ndarray,
        target_db: float = TARGET_DB
    ) -> np.ndarray:
        """音声データの音量を正規化
        
        音声データの平均音量が指定されたデシベル値となるように
        正規化を行います。音声の品質を保ちながら、適切な音量レベル
        を維持します。
        
        Args:
            audio_data: 音声データ配列
            target_db: 目標とする平均音量（dB）
            
        Returns:
            np.ndarray: 正規化された音声データ
        """
        # 実効値（RMS）を計算
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # 現在のdBを計算
        current_db = 20 * np.log10(rms) if rms > 0 else -100
        
        # 必要なゲインを計算
        gain = 10 ** ((target_db - current_db) / 20)
        
        # ゲインを適用
        return audio_data * gain

    @staticmethod
    def apply_fade(
        audio_data: np.ndarray,
        fade_samples: int = FADE_SAMPLES,
        fade_type: str = 'both'
    ) -> np.ndarray:
        """音声データにフェード効果を適用
        
        音声の開始部分にフェードイン、終了部分にフェードアウトを
        適用して、急激な音量変化を防ぎます。
        
        Args:
            audio_data: 音声データ配列
            fade_samples: フェードを適用するサンプル数
            fade_type: フェードの種類（'in', 'out', 'both'のいずれか）
            
        Returns:
            np.ndarray: フェード効果が適用された音声データ
            
        Note:
            fade_samplesのデフォルト値は約0.004秒のフェード時間に相当します。
        """
        result = audio_data.copy()
        
        if fade_type in ['in', 'both']:
            # フェードイン（徐々に音量を上げる）
            fade_in = np.linspace(0, 1, fade_samples)
            result[:fade_samples] *= fade_in
            
        if fade_type in ['out', 'both']:
            # フェードアウト（徐々に音量を下げる）
            fade_out = np.linspace(1, 0, fade_samples)
            result[-fade_samples:] *= fade_out
            
        return result

    @staticmethod
    def remove_dc_offset(audio_data: np.ndarray) -> np.ndarray:
        """音声データからDCオフセットを除去
        
        音声信号の平均値を0に調整することで、不要なDCオフセットを
        除去します。これにより、音声品質が向上し、後続の処理が
        より正確になります。
        
        Args:
            audio_data: 音声データ配列
            
        Returns:
            np.ndarray: DCオフセットが除去された音声データ
        """
        # 平均値を計算して信号から減算
        return audio_data - np.mean(audio_data)

    @staticmethod
    def apply_preprocessing(
        audio_data: np.ndarray,
        normalize: bool = True,
        remove_dc: bool = True,
        apply_fade: bool = True
    ) -> np.ndarray:
        """音声データに一連の前処理を適用
        
        複数の音声処理を組み合わせて、音声品質を最適化します。
        このメソッドは推奨される前処理をまとめて適用する
        ユーティリティメソッドとして機能します。
        
        Args:
            audio_data: 音声データ配列
            normalize: 音量の正規化を行うかどうか
            remove_dc: DCオフセットの除去を行うかどうか
            apply_fade: フェード効果を適用するかどうか
            
        Returns:
            np.ndarray: 前処理が適用された音声データ
        """
        result = audio_data.copy()
        
        if remove_dc:
            result = AudioProcessor.remove_dc_offset(result)
            
        if normalize:
            result = AudioProcessor.normalize_audio(result)
            
        if apply_fade:
            result = AudioProcessor.apply_fade(result)
            
        return result