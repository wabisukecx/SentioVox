"""音声録音モジュール

このモジュールは、高品質な音声入力の録音と保存を担当します。
PyAudioを使用してマイク入力を処理し、設定可能なパラメータに基づいて
音声を録音します。処理の各段階で適切なエラーハンドリングと
進捗表示を行います。

主な機能：
1. マイクからの音声入力の取得
2. 音声品質の最適化
3. WAVファイルとしての保存
4. リソースの適切な管理
"""

import io
import wave
import time
from typing import Optional, Tuple
import numpy as np
import pyaudio
from ..models.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FORMAT,
    DEFAULT_CHANNELS,
    DEFAULT_RATE
)

class AudioRecorder:
    """音声録音を管理するクラス
    
    このクラスは以下の責任を持ちます：
    - 音声入力デバイスの初期化と管理
    - 録音パラメータの制御
    - 音声データの取得と保存
    - リソースの適切なクリーンアップ
    """
    
    def __init__(self):
        """録音パラメータの初期化
        
        デフォルトのパラメータを設定し、音声入力に使用する
        リソースを初期化します。
        """
        # 録音パラメータの設定
        self.chunk = DEFAULT_CHUNK_SIZE
        self.format = getattr(pyaudio, DEFAULT_FORMAT)
        self.channels = DEFAULT_CHANNELS
        self.rate = DEFAULT_RATE
        
        # リソース管理用の変数
        self._stream = None
        self._pyaudio = None
        self._is_recording = False
        self._recorded_frames = []

    def _initialize_pyaudio(self) -> pyaudio.PyAudio:
        """PyAudioの初期化
        
        音声入力システムを初期化し、必要なリソースを確保します。
        既存のインスタンスがある場合は再利用します。
        
        Returns:
            pyaudio.PyAudio: 初期化されたPyAudioインスタンス
        """
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio

    def _cleanup(self):
        """リソースのクリーンアップ
        
        録音に使用したリソースを適切に解放します。
        エラーが発生した場合でも、可能な限りリソースの
        解放を試みます。
        """
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                print(f"ストリームのクローズ中にエラー: {str(e)}")
            finally:
                self._stream = None
            
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception as e:
                print(f"PyAudioの終了中にエラー: {str(e)}")
            finally:
                self._pyaudio = None

    def record_chunk(
        self,
        filename: str,
        duration_seconds: int = 10
    ) -> Optional[str]:
        """指定された時間だけ音声を録音
        
        マイクから音声を録音し、WAVファイルとして保存します。
        録音中は進捗状況を表示し、ユーザーに残り時間を
        知らせます。
        
        Args:
            filename: 保存するファイルのパス
            duration_seconds: 録音時間（秒）
            
        Returns:
            Optional[str]: 録音ファイルのパス（成功時）
            エラー時はNoneを返します。
        """
        buffer = io.BytesIO()
        total_frames = int(self.rate / self.chunk * duration_seconds)
        recorded_chunks = 0
        last_update_time = time.time()

        try:
            p = self._initialize_pyaudio()
            self._stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._get_callback()
            )

            print(f"* {filename} に録音を開始します")
            self._is_recording = True
            self._stream.start_stream()

            try:
                # 録音の進捗状況を表示しながら待機
                while recorded_chunks < total_frames and self._is_recording:
                    time.sleep(0.1)  # CPU負荷を軽減
                    recorded_chunks = len(self._recorded_frames)
                    
                    # 1秒ごとに進捗を更新
                    current_time = time.time()
                    if current_time - last_update_time >= 1.0:
                        remaining = duration_seconds - (recorded_chunks * self.chunk / self.rate)
                        print(f"残り時間: {remaining:.1f} 秒")
                        last_update_time = current_time

            except KeyboardInterrupt:
                print("\n* 録音が中断されました")
                self._is_recording = False
                raise

            finally:
                print("\n* 録音を終了します...")
                self._is_recording = False
                self._stream.stop_stream()
                self._stream.close()

            # 録音データの保存
            if self._recorded_frames:
                try:
                    self._save_wav_file(filename)
                    print(f"* 録音を {filename} として保存しました")
                    return filename
                except Exception as e:
                    print(f"録音の保存中にエラーが発生しました: {str(e)}")
                    return None
            else:
                print("* 録音データがありません")
                return None

        except Exception as e:
            print(f"録音中にエラーが発生しました: {str(e)}")
            return None

        finally:
            self._cleanup()
            self._recorded_frames = []

    def _get_callback(self):
        """録音コールバック関数を生成
        
        非同期の音声入力処理のためのコールバック関数を返します。
        このコールバックは、音声データが利用可能になるたびに
        呼び出されます。
        
        Returns:
            Callable: 録音コールバック関数
        """
        def callback(in_data, frame_count, time_info, status):
            """非同期録音のコールバック関数
            
            Args:
                in_data: 入力音声データ
                frame_count: フレーム数
                time_info: タイミング情報
                status: ステータスフラグ
                
            Returns:
                Tuple: (データ, pyaudio.paContinue)
            """
            if self._is_recording:
                self._recorded_frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        
        return callback

    def _save_wav_file(self, filename: str):
        """録音データをWAVファイルとして保存
        
        録音したデータを指定されたファイル名でWAV形式で保存します。
        音声フォーマットやチャンネル数などの情報も適切に設定します。
        
        Args:
            filename: 保存するファイルのパス
        """
        if not self._recorded_frames:
            raise ValueError("保存する録音データがありません")

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self._pyaudio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self._recorded_frames))
        except Exception as e:
            raise IOError(f"WAVファイルの保存中にエラーが発生しました: {str(e)}")

    def get_input_devices(self) -> list:
        """利用可能な入力デバイスの一覧を取得
        
        システムで利用可能なすべての音声入力デバイスの
        情報を取得します。
        
        Returns:
            list: 入力デバイス情報のリスト
        """
        devices = []
        p = self._initialize_pyaudio()
        
        try:
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # 入力デバイスのみ
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
        finally:
            if p != self._pyaudio:  # 新しく作成したインスタンスの場合
                p.terminate()
                
        return devices

    def set_device(self, device_index: int) -> bool:
        """使用する入力デバイスを設定
        
        特定の入力デバイスを選択して使用するように設定します。
        
        Args:
            device_index: 使用するデバイスのインデックス
            
        Returns:
            bool: 設定の成功/失敗
        """
        try:
            p = self._initialize_pyaudio()
            device_info = p.get_device_info_by_index(device_index)
            
            if device_info['maxInputChannels'] > 0:
                self.device_index = device_index
                return True
            else:
                print(f"デバイス {device_index} は入力デバイスではありません")
                return False
                
        except Exception as e:
            print(f"デバイスの設定中にエラーが発生しました: {str(e)}")
            return False

    def monitor_audio_level(
        self,
        duration: float = 5.0,
        update_interval: float = 0.1
    ) -> list:
        """音声入力レベルをモニタリング
        
        指定された時間だけ音声入力レベルをモニタリングし、
        結果を返します。これは録音の前に適切な入力レベルを
        確認するのに役立ちます。
        
        Args:
            duration: モニタリング時間（秒）
            update_interval: 更新間隔（秒）
            
        Returns:
            list: 音声レベルの時系列データ
        """
        levels = []
        chunks = []
        
        try:
            p = self._initialize_pyaudio()
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            end_time = time.time() + duration
            
            while time.time() < end_time:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    chunks.append(data)
                    # 音声レベルの計算
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    level = np.abs(audio_data).mean()
                    levels.append(level)
                    
                    # レベルメーターの表示
                    meter_length = int(level / 100)
                    print(f"\rレベル: {'#' * meter_length}{' ' * (50 - meter_length)}", end='')
                    
                    time.sleep(update_interval)
                    
                except IOError as e:
                    print(f"\nオーバーフローを検出しました: {str(e)}")
                    continue

        except Exception as e:
            print(f"\nモニタリング中にエラーが発生しました: {str(e)}")
            
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            self._cleanup()
            print("\nモニタリングを終了しました")
            
        return levels