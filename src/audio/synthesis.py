"""音声合成システムのメインモジュール

このモジュールは、感情分析に基づく音声合成システムの中核となる機能を提供します。
各コンポーネント（プロセス管理、音声処理、感情マッピング、AIVISクライアント）を
統合し、テキストから感情豊かな音声を生成するプロセス全体を管理します。

このモジュールの主な責任：
1. 各コンポーネントの初期化と管理
2. 音声合成プロセス全体の調整
3. エラーハンドリングとリカバリ
4. 音声ファイルの保存と再生の制御
"""

import os
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import numpy as np
import ffmpeg
import sounddevice
import soundfile
from ..models.constants import AIVIS_BASE_URL
from .process_manager import ensure_aivis_server, AivisProcessManager
from .processor import AudioProcessor
from .emotion_mapper import EmotionVoiceMapper
from .aivis_client import AivisClient

class AivisAdapter:
    """音声合成アダプター
    
    このクラスは、感情分析の結果に基づいて音声を合成し、高品質な
    音声出力を生成します。各コンポーネントを適切に連携させ、
    エラーが発生した場合でも適切に処理を継続します。
    """
    
    def __init__(self):
        """初期化
        
        各コンポーネントを初期化し、AIVISエンジンの状態を確認します。
        エンジンが利用できない場合は、適切なエラーメッセージと共に
        プログラムを終了します。
        """
        # AIVISエンジンの状態確認
        success, message = ensure_aivis_server(AIVIS_BASE_URL)
        if not success:
            raise RuntimeError(
                f"\nエラー: {message}\n"
                "音声合成を利用するには、AivisSpeech-Engineが必要です。"
            )
        print(f"\n{message}")
        
        # 各コンポーネントの初期化
        self.audio_processor = AudioProcessor()
        self.emotion_mapper = EmotionVoiceMapper()
        self.aivis_client = AivisClient(AIVIS_BASE_URL)
        self.process_manager = AivisProcessManager()

    def cleanup(self):
        """AIVISプロセスのクリーンアップ"""
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
        
        複数のテキストセグメントを感情スコアに基づいて合成し、
        必要に応じてファイルに保存します。処理の各段階で適切な
        エラーハンドリングを行い、可能な限り処理を継続します。
        
        Args:
            segments: テキストセグメントのリスト
            emotion_scores_list: 感情スコアのリスト
            save_path: 保存先のパス（オプション）
            play_audio: 音声を再生するかどうか
            
        Returns:
            Optional[str]: 保存したファイルのパス（保存時のみ）
        """
        audio_segments = []
        rate = None
        
        print("\n音声合成を開始します...")
        
        # 各セグメントの処理
        for i, (text, scores) in enumerate(zip(segments, emotion_scores_list), 1):
            if not text.strip():
                continue

            print(f"\nセグメント {i}/{len(segments)} を処理中...")
            
            try:
                # 感情スコアから音声パラメータを計算
                style_id, params = self.emotion_mapper.calculate_mixed_parameters(
                    self.emotion_mapper.convert_scores_to_dict(scores)
                )
                
                # 音声合成の実行
                segment_result = self.aivis_client.synthesize_segment(
                    text, style_id, params
                )
                
                if segment_result is not None:
                    audio_data, current_rate = segment_result
                    
                    # 音声データの前処理
                    audio_data = self.audio_processor.apply_preprocessing(
                        audio_data,
                        normalize=True,
                        remove_dc=True,
                        apply_fade=True
                    )
                    
                    audio_segments.append(audio_data)
                    if rate is None:
                        rate = current_rate
                    print(f"セグメント {i} の合成が完了しました")
                else:
                    print(f"警告: セグメント {i} の合成に失敗しました")
                    continue
            
            except Exception as e:
                print(f"エラー: セグメント {i} の処理中に例外が発生しました: {str(e)}")
                continue

        # セグメントの合成結果の確認
        if not audio_segments:
            print("警告: すべての音声合成に失敗しました")
            return None
            
        try:
            # 音声セグメントの結合
            print("\n音声データを結合しています...")
            combined_audio = self.audio_processor.combine_segments_with_silence(
                audio_segments
            )
            print(f"結合後の音声データの形状: {combined_audio.shape}")
            
        except Exception as e:
            print(f"音声データの結合中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

        # 保存と再生の処理
        output_path = self._save_audio_file(
            combined_audio,
            rate,
            save_path
        )
        
        if output_path is None:
            return None

        # 音声の再生（オプション）
        if play_audio:
            try:
                sounddevice.play(combined_audio, rate)
                sounddevice.wait()
            except Exception as e:
                print(f"音声の再生中にエラーが発生しました: {str(e)}")

        return output_path

    def _save_audio_file(
        self,
        audio_data: np.ndarray, 
        rate: int,
        save_path: Optional[str]
    ) -> Optional[str]:
        """音声データをファイルとして保存
        
        音声データをWAVファイルとして保存し、必要に応じてM4Aに
        変換します。一時ファイルの適切な管理とエラーハンドリングを
        行います。

        Args:
            audio_data: 音声データ配列
            rate: サンプリングレート
            save_path: 保存先のパス（オプション）

        Returns:
            Optional[str]: 保存したファイルのパス、エラー時はNone
        """
        # 保存パスの設定
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"output_{timestamp}.m4a"

        # WAVファイルとして一時保存
        temp_wav = Path(save_path).with_suffix('.wav')
        try:
            soundfile.write(str(temp_wav), audio_data, rate)
            print(f"一時WAVファイルを保存しました: {temp_wav}")

            # M4Aへの変換処理
            try:
                process = (
                    ffmpeg
                    .input(str(temp_wav))
                    .output(
                        str(save_path),
                        acodec='aac',
                        audio_bitrate='192k',
                        loglevel='error'
                    )
                    .overwrite_output()
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

                # タイムアウト付きで変換処理の完了を待機
                try:
                    stdout, stderr = process.communicate(timeout=30)
                    print(f"音声ファイルを保存しました: {save_path}")
                except subprocess.TimeoutExpired:
                    process.kill()
                    print("FFmpegの処理がタイムアウトしました。WAVファイルを使用します。")
                    save_path = str(temp_wav)
                    return save_path

            except ffmpeg.Error as e:
                print(f"M4Aへの変換に失敗しました: {str(e)}")
                print("WAVファイルを代替として使用します。")
                save_path = str(temp_wav)
                return save_path

        except Exception as e:
            print(f"音声ファイルの保存中にエラーが発生しました: {str(e)}")
            if temp_wav.exists():
                save_path = str(temp_wav)
                return save_path
            return None

        finally:
            # 一時ファイルのクリーンアップ
            if temp_wav.exists() and save_path != str(temp_wav):
                try:
                    temp_wav.unlink()
                    print("一時WAVファイルを削除しました")
                except Exception as e:
                    print(f"一時ファイルの削除中にエラーが発生しました: {str(e)}")

        return save_path