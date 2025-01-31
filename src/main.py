import os
import argparse
from datetime import datetime
from typing import Optional, List
import shutil

from .audio.recorder import AudioRecorder
from .audio.synthesis import AivisAdapter
from .analysis.emotion import EmotionAnalyzer
from .analysis.text import TextProcessor


class EmotionAnalysisSystem:
    """SentioVoxシステムのメインクラス
    
    このクラスは感情分析と音声合成を統合し、テキストや音声入力から
    感情豊かな音声出力を生成するSentioVoxシステムの中核となる機能を提供します。
    音声録音、テキスト処理、感情分析、音声合成の各コンポーネントを
    シームレスに連携させ、より自然な感情表現を実現します。
    """
    def __init__(self):
        self.recorder = AudioRecorder()
        self.aivis = AivisAdapter()
        self.emotion_analyzer = EmotionAnalyzer()
        self.text_processor = TextProcessor()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """システムリソースのクリーンアップ"""
        if hasattr(self, 'recorder'):
            self.recorder._cleanup()
        if hasattr(self, 'aivis'):
            # AivisAdapterのクリーンアップを呼び出し
            self.aivis.cleanup()

    def record_and_analyze(
        self,
        duration: int = 10,
        speak: bool = False,
        save_path: Optional[str] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """音声の録音と分析を実行
    
        Args:
        duration: 録音時間（秒）
        speak: 音声合成を行うかどうか
        save_path: 音声ファイルの保存先パス
        play_audio: 音声を再生するかどうか
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        try:
            recorded_file = self.recorder.record_chunk(filename, duration)
            if recorded_file:
                print("\n録音した音声を分析しています...")
                # 録音ファイルを保存する場合
                if save_path:
                    # 録音ファイルを指定されたパスにコピー
                    shutil.copy2(recorded_file, save_path)
                    print(f"\n録音ファイルを保存しました: {save_path}")
            
                self.process_file(
                    recorded_file,
                    delete_after=True,
                    speak=speak,
                    save_path=save_path,
                    play_audio=play_audio
                )
                return recorded_file
        except KeyboardInterrupt:
            print("\n録音と分析が中断されました")
            if os.path.exists(filename):
                os.remove(filename)
                print(f"\n録音ファイル {filename} が削除されました")
            return None

    def process_file(
        self,
        file_path: str,
        delete_after: bool = False,
        speak: bool = False,
        save_path: Optional[str] = None,
        play_audio: bool = True
    ) -> None:
        """ファイルの処理（音声またはテキスト）
        
        Args:
            file_path: 処理対象のファイルパス
            delete_after: 処理後にファイルを削除するかどうか
            speak: 音声合成を行うかどうか
            save_path: 音声ファイルの保存先パス
            play_audio: 音声を再生するかどうか
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
                segments = self.text_processor.segment_audio(file_path)
            elif file_ext in ['.txt']:
                segments = self.text_processor.segment_text(file_path)
            else:
                raise ValueError(f"未対応のファイル形式です: {file_ext}")

            if not segments:
                raise ValueError("ファイルからセグメントを抽出できませんでした")

            emotion_scores = self.emotion_analyzer.analyze_emotions(segments)
            self.emotion_analyzer.print_results(segments, emotion_scores)

            # speak が True の場合のみ音声合成を実行
            if speak:
                self.analyze_and_speak(
                    segments,
                    emotion_scores,
                    save_path=save_path,
                    play_audio=play_audio
                )

        finally:
            if delete_after and os.path.exists(file_path):
                os.remove(file_path)
                print(f"\n一時ファイル {file_path} を削除しました")

    def analyze_and_speak(
        self,
        segments: List[str],
        emotion_scores: List[List[float]],
        save_path: Optional[str] = None,
        play_audio: bool = True
    ) -> Optional[str]:
        """感情分析結果に基づいて音声合成を実行
        
        Args:
            segments: テキストセグメントのリスト
            emotion_scores: 感情スコアのリスト
            save_path: 音声ファイルの保存先パス
            play_audio: 音声を再生するかどうか
            
        Returns:
            保存したファイルのパス（save_pathが指定されている場合）
        """
        return self.aivis.speak_continuous(
            segments,
            emotion_scores,
            save_path=save_path,
            play_audio=play_audio
        )


def main():
    parser = argparse.ArgumentParser(
        description='SentioVox: 感情分析と音声合成システム'
    )
    parser.add_argument(
        '--file',
        help='分析対象の音声ファイル(.mp3等)またはテキストファイル(.txt)',
        default=None
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='マイクから録音を開始'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='録音時間（秒）'
    )
    parser.add_argument(
        '--speak',
        action='store_true',
        help='分析結果を音声合成で読み上げる'
    )
    parser.add_argument(
        '--output',
        metavar='FILENAME',
        help='音声ファイルの出力先（.wav または .m4a 形式）',
        default=None
    )
    parser.add_argument(
        '--no-play',
        action='store_true',
        help='音声の再生を無効化'
    )

    args = parser.parse_args()

    # 出力ファイル名の検証を追加
    if args.output is not None and not args.output.strip():
        parser.error("--outputオプションにはファイル名を指定する必要があります")

    try:
        with EmotionAnalysisSystem() as system:
            if args.record:
                print("マイク録音を開始します...")
                system.record_and_analyze(
                    args.duration,
                    speak=args.speak,
                    save_path=args.output,
                    play_audio=not args.no_play
                )
            elif args.file:
                system.process_file(
                    args.file,
                    speak=args.speak,
                    save_path=args.output,
                    play_audio=not args.no_play
                )
            else:
                parser.print_help()
    except KeyboardInterrupt:
        print("\n処理が中断されました")
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()