import io
import wave
import pyaudio
from datetime import datetime
from ..models.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FORMAT,
    DEFAULT_CHANNELS,
    DEFAULT_RATE
)


class AudioRecorder:
    """SentioVoxの音声録音モジュール
    
    高品質な音声録音を管理するクラスです。
    PyAudioを使用して音声入力を処理し、設定可能なパラメータ
    （チャンネル数、サンプリングレート、フォーマットなど）に基づいて
    音声を録音します。
    
    録音データは一時ファイルとして保存され、処理完了後に
    自動的に削除される安全な設計となっています。
    """
    def __init__(self):
        self.chunk = DEFAULT_CHUNK_SIZE
        self.format = getattr(pyaudio, DEFAULT_FORMAT)
        self.channels = DEFAULT_CHANNELS
        self.rate = DEFAULT_RATE

    def record_chunk(self, filename: str, duration_seconds: int = 10) -> str | None:
        """指定された時間だけ音声を録音"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print(f"* {filename} に録音中")
        buffer = io.BytesIO()
        total_frames = int(self.rate / self.chunk * duration_seconds)

        try:
            for i in range(total_frames):
                data = stream.read(self.chunk)
                buffer.write(data)
                if i % (self.rate // self.chunk) == 0:
                    remaining = duration_seconds - (i * self.chunk / self.rate)
                    print(f"残り時間: {remaining:.1f} 秒")
        except KeyboardInterrupt:
            print("\n* 録音が中断されました")
            raise
        finally:
            print("* 録音を保存中...")
            stream.stop_stream()
            stream.close()
            p.terminate()

        if buffer.tell() > 0:
            buffer.seek(0)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(buffer.getvalue())
            print(f"* {filename} として保存されました")
            return filename
        return None