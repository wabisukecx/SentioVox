"""音声データの処理を担当するモジュール

このモジュールは音声データの低レベル処理を担当し、以下の機能を提供します：
- 音声データからの不要な無音部分の除去
- 複数の音声セグメントの結合
- 音声品質の最適化と正規化
- フェード効果の適用
- 自然な区切りでの音声分割

すべての処理は NumPy を使用して効率的に実装されています。各処理は
独立したメソッドとして提供され、必要に応じて組み合わせることが
できます。
"""

import numpy as np
from typing import List
from ..models.constants import (
    SILENCE_THRESHOLD,
    MARGIN_SAMPLES,
    SILENCE_DURATION,
    TARGET_DB,
    FADE_SAMPLES,
    MIN_SEGMENT_LENGTH,
    MAX_SEGMENT_LENGTH,
    MIN_AUDIO_QUALITY,
    MAX_DC_OFFSET,
    MIN_PEAK_THRESHOLD,
    PREPROCESSING_CONFIG,
    DEFAULT_OUTPUT_SAMPLING_RATE,
    SPLIT_WINDOW_SIZE,
    SPLIT_SMOOTHING_WINDOW,
    SPLIT_MARGIN,
    MIN_SPLIT_SEGMENT,
    MAX_AMPLITUDE_THRESHOLD
)


class AudioProcessor:
    """音声データの処理を行うクラス
    
    このクラスは音声データに対する様々な処理を提供します。
    すべてのメソッドは静的メソッドとして実装され、
    インスタンス状態に依存しない純粋な関数として機能します。
    
    主な機能：
    - 無音区間の検出と除去
    - 音量の正規化
    - フェード効果の適用
    - DCオフセットの除去
    - 音声セグメントの結合
    - 自然な区切りでの音声分割
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
            
        Note:
            - threshold は 0.0-1.0 の範囲で指定します
            - margin_samples はサンプル数単位で指定します
        """
        if len(audio_data) == 0:
            return audio_data

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
        
        # トリミング後のデータが短すぎる場合は元のデータを返す
        if (end - start) < MIN_SEGMENT_LENGTH:
            return audio_data
        
        return audio_data[start:end]

    @staticmethod
    def find_natural_split_point(
        audio_data: np.ndarray,
        around_position: int,
        sampling_rate: int = DEFAULT_OUTPUT_SAMPLING_RATE
    ) -> int:
        """最も自然な分割ポイントを見つける
        
        振幅が最も小さい（無音に近い）位置を探して、
        自然な分割ポイントとして返します。位置の前後にマージンを設けて、
        より自然な分割点を探索します。
        
        Args:
            audio_data: 音声データ
            around_position: この位置の周辺を探索
            sampling_rate: サンプリングレート（Hz）
            
        Returns:
            int: 最適な分割ポイント
        """
        # マージンをサンプル数に変換
        margin_samples = int(SPLIT_MARGIN * sampling_rate)
        
        # 探索範囲を設定（マージンを考慮）
        start = max(margin_samples, around_position - SPLIT_WINDOW_SIZE // 2)
        end = min(len(audio_data) - margin_samples, around_position + SPLIT_WINDOW_SIZE // 2)
        
        if end <= start:
            return around_position  # 探索範囲が無効な場合
        
        # 探索範囲の振幅を計算
        amplitudes = np.abs(audio_data[start:end])
        
        # 移動平均を計算して急激な変化を避ける
        window = np.ones(SPLIT_SMOOTHING_WINDOW) / SPLIT_SMOOTHING_WINDOW
        smoothed = np.convolve(amplitudes, window, mode='same')
        
        # 振幅が大きすぎる部分を避ける
        valid_positions = smoothed < MAX_AMPLITUDE_THRESHOLD
        if not np.any(valid_positions):
            return around_position  # 適切な分割点が見つからない場合
        
        # 最も振幅の小さい点を見つける
        valid_smoothed = np.where(valid_positions, smoothed, np.inf)
        min_pos = np.argmin(valid_smoothed)
        
        return start + min_pos

    @staticmethod
    def split_segment(
        audio_data: np.ndarray,
        max_samples: int,
        sampling_rate: int = DEFAULT_OUTPUT_SAMPLING_RATE
    ) -> List[np.ndarray]:
        """長いセグメントを自然な区切りで分割
        
        Args:
            audio_data: 分割する音声データ
            max_samples: 最大サンプル数
            sampling_rate: サンプリングレート（Hz）
            
        Returns:
            List[np.ndarray]: 分割された音声セグメントのリスト
        """
        segments = []
        remaining = audio_data
        min_samples = int(MIN_SPLIT_SEGMENT * sampling_rate)
        
        while len(remaining) > max_samples:
            # 自然な分割ポイントを見つける
            split_point = AudioProcessor.find_natural_split_point(
                remaining, max_samples, sampling_rate
            )
            
            # 分割が不適切な場合（セグメントが短すぎる場合）は
            # 次の候補点を探す
            if split_point < min_samples:
                split_point = max_samples
            
            # セグメントを分割
            segments.append(remaining[:split_point])
            remaining = remaining[split_point:]
        
        if len(remaining) >= min_samples:
            segments.append(remaining)
            
        return segments

    @staticmethod
    def combine_segments_with_silence(
        segments: List[np.ndarray],
        silence_duration: int = SILENCE_DURATION,
        sampling_rate: int = DEFAULT_OUTPUT_SAMPLING_RATE
    ) -> np.ndarray:
        """音声セグメントを適切な無音区間を挿入して結合
        
        複数の音声セグメントを結合する際に、自然な間隔となるように
        一定の無音区間を挿入します。長いセグメントは自然な区切りで
        分割されます。
        
        Args:
            segments: 音声セグメントのリスト
            silence_duration: セグメント間の無音の長さ（サンプル数）
            sampling_rate: サンプリングレート（Hz）
            
        Returns:
            np.ndarray: 結合された音声データ
        """
        if not segments:
            return np.array([])
            
        # 秒をサンプル数に変換
        max_samples = int(MAX_SEGMENT_LENGTH * sampling_rate)
            
        # セグメントの長さをチェックと分割
        validated_segments = []
        for segment in segments:
            if len(segment) > max_samples:
                print(f"注意: セグメントを自然な区切りで分割します ({len(segment)} サンプル)")
                split_segments = AudioProcessor.split_segment(segment, max_samples, sampling_rate)
                validated_segments.extend(split_segments)
            else:
                validated_segments.append(segment)
            
        # 固定長の無音データを生成
        silence = np.zeros(silence_duration)
        
        # セグメント間に無音を挿入しながら結合
        combined = []
        for i, segment in enumerate(validated_segments):
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
        if len(audio_data) == 0:
            return audio_data

        # 実効値（RMS）を計算
        rms = np.sqrt(np.mean(np.square(audio_data)))
        if rms < MIN_AUDIO_QUALITY:
            print(f"音声レベルが基準値を下回っています (RMS: {rms:.3f})")
        
        # 現在のdBを計算
        current_db = 20 * np.log10(rms) if rms > 0 else -100
        
        # 必要なゲインを計算
        gain = 10 ** ((target_db - current_db) / 20)
        
        # ゲインの範囲を制限
        gain = np.clip(gain, 0.1, 10.0)
        
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
        if len(audio_data) == 0:
            return audio_data

        result = audio_data.copy()
        fade_samples = min(fade_samples, len(audio_data) // 2)
        
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
        if len(audio_data) == 0:
            return audio_data

        # 平均値を計算して信号から減算
        mean_value = np.mean(audio_data)
        if abs(mean_value) > MAX_DC_OFFSET:
            print(f"警告: 大きなDCオフセットを検出 ({mean_value:.3f})")
        return audio_data - mean_value

    @staticmethod
    def apply_preprocessing(
        audio_data: np.ndarray,
        normalize: bool = PREPROCESSING_CONFIG['normalize'],
        remove_dc: bool = PREPROCESSING_CONFIG['remove_dc'],
        apply_fade: bool = PREPROCESSING_CONFIG['apply_fade']
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
            
        Note:
            処理の順序は以下の通りです：
            1. DCオフセットの除去
            2. 音量の正規化
            3. フェード効果の適用
        """
        result = audio_data.copy()
        
        if remove_dc:
            result = AudioProcessor.remove_dc_offset(result)
            
        if normalize:
            result = AudioProcessor.normalize_audio(result)
            
        if apply_fade:
            result = AudioProcessor.apply_fade(result)
            
        return result

    @staticmethod
    def check_audio_quality(audio_data: np.ndarray) -> bool:
        """音声データの品質をチェック
        
        音声データの品質を評価し、問題がある場合は警告を出力します。
        RMSレベルは一般的な会話音声の範囲（0.01-0.5）を基準とします。
        
        Args:
            audio_data: チェックする音声データ
            
        Returns:
            bool: 音声品質が基準を満たしている場合はTrue
        """
        if len(audio_data) == 0:
            print("警告: 空の音声データです")
            return False

        # RMSレベルのチェック
        rms = np.sqrt(np.mean(np.square(audio_data)))
        if rms < MIN_AUDIO_QUALITY:
            print(f"音声レベルが基準値を下回っています (RMS: {rms:.3f})")
            return True  # 警告のみで処理は継続

        # ピーク値のチェック（クリッピング検出）
        peak = np.max(np.abs(audio_data))
        if peak > MIN_PEAK_THRESHOLD:
            print(f"警告: クリッピングの可能性があります (ピーク値: {peak:.3f})")
            return False

        # DCオフセットのチェック（著しい偏りの検出）
        dc_offset = np.mean(audio_data)
        if abs(dc_offset) > MAX_DC_OFFSET:
            print(f"警告: 大きなDCオフセットが存在します ({dc_offset:.3f})")
            return False

        return True

    @staticmethod
    def validate_segment_length(
        segment: np.ndarray,
        sampling_rate: int = DEFAULT_OUTPUT_SAMPLING_RATE
    ) -> bool:
        """音声セグメントの長さが適切かどうかを検証
        
        Args:
            segment: 検証する音声セグメント
            sampling_rate: サンプリングレート（Hz）
            
        Returns:
            bool: セグメントの長さが適切な場合はTrue
        """
        duration = len(segment) / sampling_rate
        min_length = MIN_SPLIT_SEGMENT
        max_length = MAX_SEGMENT_LENGTH
        
        if duration < min_length:
            print(f"警告: セグメントが短すぎます ({duration:.2f}秒)")
            return False
        
        if duration > max_length:
            print(f"警告: セグメントが長すぎます ({duration:.2f}秒)")
            return False
            
        return True

    @staticmethod
    def analyze_segment_properties(
        segment: np.ndarray,
        sampling_rate: int = DEFAULT_OUTPUT_SAMPLING_RATE
    ) -> dict:
        """音声セグメントの各種プロパティを分析
        
        Args:
            segment: 分析する音声セグメント
            sampling_rate: サンプリングレート（Hz）
            
        Returns:
            dict: 分析結果を含む辞書
        """
        # 基本的な統計量を計算
        rms = np.sqrt(np.mean(np.square(segment)))
        peak = np.max(np.abs(segment))
        dc_offset = np.mean(segment)
        duration = len(segment) / sampling_rate
        
        # 振幅の分布を分析
        percentiles = np.percentile(np.abs(segment), [25, 50, 75])
        
        return {
            'duration': duration,
            'rms_level': rms,
            'peak_value': peak,
            'dc_offset': dc_offset,
            'amplitude_percentiles': percentiles,
            'sample_count': len(segment)
        }

    @staticmethod
    def get_segment_statistics(segments: List[np.ndarray]) -> dict:
        """複数の音声セグメントの統計情報を取得
        
        Args:
            segments: 分析する音声セグメントのリスト
            
        Returns:
            dict: 統計情報を含む辞書
        """
        if not segments:
            return {}
            
        # 各セグメントの長さを取得
        lengths = [len(seg) for seg in segments]
        
        # RMSレベルを計算
        rms_levels = [
            np.sqrt(np.mean(np.square(seg)))
            for seg in segments
        ]
        
        return {
            'segment_count': len(segments),
            'total_samples': sum(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': np.mean(lengths),
            'min_rms': min(rms_levels),
            'max_rms': max(rms_levels),
            'avg_rms': np.mean(rms_levels)
        }