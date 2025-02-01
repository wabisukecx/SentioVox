"""音声合成システムのメインモジュール

感情分析に基づく音声合成システムの中核機能を提供し、各コンポーネントを統合して
テキストから感情豊かな音声を生成するプロセスを管理します。
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import ffmpeg
import sounddevice
import soundfile
import subprocess
import traceback

from ..models.constants import (
    AIVIS_BASE_URL,
    AUDIO_CODEC,
    AUDIO_BITRATE,
    FFMPEG_LOG_LEVEL,
    FFMPEG_TIMEOUT,
    PREPROCESSING_CONFIG
)
from .process_manager import ensure_aivis_server, AivisProcessManager
from .processor import AudioProcessor
from .emotion_mapper import EmotionVoiceMapper
from .aivis_client import AivisClient


class AivisAdapter:
    """音声合成アダプター

    感情分析の結果に基づいて音声を合成し、高品質な音声出力を生成します。
    各コンポーネントを連携させ、エラー発生時も適切に処理を継続します。
    """

    def __init__(self) -> None:
        """AIVISアダプターの初期化とコンポーネントのセットアップ"""
        success, message = ensure_aivis_server(AIVIS_BASE_URL)
        if not success:
            raise RuntimeError(
                f"\nエラー: {message}\n"
                "音声合成を利用するには、AivisSpeech-Engineが必要です。"
            )
        print(f"\n{message}")

        self.audio_processor = AudioProcessor()
        self.emotion_mapper = EmotionVoiceMapper()
        self.aivis_client = AivisClient(AIVIS_BASE_URL)
        self.process_manager = AivisProcessManager()

    def cleanup(self) -> None:
        """AIVISプロセスのクリーンアップを実行"""
        if hasattr(self, 'process_manager'):
            self.process_manager.cleanup()

    def speak_continuous(
        self,
        segments: List[str],
        emotion_scores_list: List[List[float]],
        save_path: Optional[str] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """連続的な音声合成を実行

        Args:
            segments: 合成するテキストセグメントのリスト
            emotion_scores_list: 感情スコアのリスト
            save_path: 保存先のファイルパス
            play_audio: 音声を再生するかどうかのフラグ

        Returns:
            Optional[str]: 保存されたファイルのパス（成功時）
        """
        print("\n音声合成を開始します...")
        audio_segments, rate = self._synthesize_segments(segments, emotion_scores_list)

        if not audio_segments:
            print("警告: すべての音声合成に失敗しました")
            return None

        combined_audio = self._combine_audio_segments(audio_segments)
        if combined_audio is None:
            return None

        output_path = None
        if save_path is not None:
            output_path = self._save_audio_file(combined_audio, rate, save_path)

        if play_audio:
            self._play_audio(combined_audio, rate)

        return output_path

    def _synthesize_segments(
        self,
        segments: List[str],
        emotion_scores_list: List[List[float]]
    ) -> Tuple[List[np.ndarray], Optional[int]]:
        """各セグメントの音声合成を実行

        Args:
            segments: 合成するテキストセグメントのリスト
            emotion_scores_list: 感情スコアのリスト

        Returns:
            Tuple[List[np.ndarray], Optional[int]]: 
                音声セグメントのリストとサンプリングレート
        """
        audio_segments = []
        rate = None

        for i, (text, scores) in enumerate(zip(segments, emotion_scores_list), 1):
            if not text.strip():
                continue

            print(f"\nセグメント {i}/{len(segments)} を処理中...")
            try:
                style_id, params = self.emotion_mapper.calculate_mixed_parameters(
                    self.emotion_mapper.convert_scores_to_dict(scores)
                )
                
                segment_result = self.aivis_client.synthesize_segment(text, style_id, params)
                if segment_result is None:
                    print(f"警告: セグメント {i} の合成に失敗しました")
                    continue

                audio_data, current_rate = segment_result
                audio_data = self.audio_processor.apply_preprocessing(
                    audio_data,
                    **PREPROCESSING_CONFIG
                )
                
                audio_segments.append(audio_data)
                if rate is None:
                    rate = current_rate
                print(f"セグメント {i} の合成が完了しました")

            except Exception as e:
                print(f"エラー: セグメント {i} の処理中に例外が発生しました: {str(e)}")
                continue

        return audio_segments, rate

    def _combine_audio_segments(
        self,
        audio_segments: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """音声セグメントを結合

        Args:
            audio_segments: 結合する音声セグメントのリスト

        Returns:
            Optional[np.ndarray]: 結合された音声データ
        """
        try:
            print("\n音声データを結合しています...")
            combined_audio = self.audio_processor.combine_segments_with_silence(
                audio_segments
            )
            print(f"結合後の音声データの形状: {combined_audio.shape}")
            return combined_audio

        except Exception as e:
            print(f"音声データの結合中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
            return None

    def _play_audio(self, audio_data: np.ndarray, rate: int) -> None:
        """音声データを再生

        Args:
            audio_data: 再生する音声データ
            rate: サンプリングレート
        """
        try:
            sounddevice.play(audio_data, rate)
            sounddevice.wait()
        except Exception as e:
            print(f"音声の再生中にエラーが発生しました: {str(e)}")

    def _save_audio_file(
        self,
        audio_data: np.ndarray,
        rate: int,
        save_path: str
    ) -> Optional[str]:
        """音声データをファイルとして保存

        Args:
            audio_data: 保存する音声データ
            rate: サンプリングレート
            save_path: 保存先のファイルパス

        Returns:
            Optional[str]: 保存されたファイルのパス
        """
        temp_wav = Path(save_path).with_suffix('.wav')
        try:
            soundfile.write(str(temp_wav), audio_data, rate)
            print(f"一時WAVファイルを保存しました: {temp_wav}")

            if self._convert_to_m4a(temp_wav, save_path):
                return save_path
            return str(temp_wav)

        except Exception as e:
            print(f"音声ファイルの保存中にエラーが発生しました: {str(e)}")
            if temp_wav.exists():
                return str(temp_wav)
            return None

        finally:
            self._cleanup_temp_file(temp_wav, save_path)

    def _convert_to_m4a(self, temp_wav: Path, save_path: str) -> bool:
        """WAVファイルをM4Aに変換

        Args:
            temp_wav: 変換元のWAVファイルパス
            save_path: 保存先のM4Aファイルパス

        Returns:
            bool: 変換の成否
        """
        try:
            process = (
                ffmpeg
                .input(str(temp_wav))
                .output(
                    str(save_path),
                    acodec=AUDIO_CODEC,
                    audio_bitrate=AUDIO_BITRATE,
                    loglevel=FFMPEG_LOG_LEVEL
                )
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            process.communicate(timeout=FFMPEG_TIMEOUT)
            print(f"音声ファイルを保存しました: {save_path}")
            return True

        except subprocess.TimeoutExpired:
            process.kill()
            print("FFmpegの処理がタイムアウトしました。WAVファイルを使用します。")
            return False

        except ffmpeg.Error as e:
            print(f"M4Aへの変換に失敗しました: {str(e)}")
            print("WAVファイルを代替として使用します。")
            return False

    def _cleanup_temp_file(self, temp_wav: Path, save_path: str) -> None:
        """一時ファイルの削除

        Args:
            temp_wav: 削除する一時ファイルのパス
            save_path: 最終的な保存先パス
        """
        if temp_wav.exists() and save_path != str(temp_wav):
            try:
                temp_wav.unlink()
                print("一時WAVファイルを削除しました")
            except Exception as e:
                print(f"一時ファイルの削除中にエラーが発生しました: {str(e)}")