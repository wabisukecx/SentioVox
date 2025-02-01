"""SentioVoxシステムのメインモジュール

テキストや音声入力から感情を分析し、感情豊かな音声を生成する
システムのエントリーポイントとなるモジュールです。
"""

import argparse
import os
from datetime import datetime
from typing import Optional, List

from .analysis.emotion import EmotionAnalyzer
from .analysis.text import TextProcessor
from .audio.recorder import AudioRecorder
from .audio.synthesis import AivisAdapter


class EmotionAnalysisSystem:
    """SentioVoxシステムの統合管理クラス

    音声録音、テキスト処理、感情分析、音声合成の各コンポーネントを
    シームレスに連携させ、感情豊かな音声出力を生成します。
    """

    def __init__(self) -> None:
        """コンポーネントの初期化"""
        self.recorder = AudioRecorder()
        self.emotion_analyzer = EmotionAnalyzer()
        self.text_processor = TextProcessor()
        self._aivis = None

    def __enter__(self) -> 'EmotionAnalysisSystem':
        """コンテキストマネージャーのエントリーポイント"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """終了時のクリーンアップ処理"""
        self.cleanup()

    def cleanup(self) -> None:
        """システムリソースの解放"""
        if hasattr(self, 'recorder'):
            self.recorder._cleanup()
        if self._aivis is not None:
            self._aivis.cleanup()

    def _initialize_aivis(self) -> None:
        """AIVISアダプターの初期化（必要な場合のみ）"""
        if self._aivis is None:
            self._aivis = AivisAdapter()

    def generate_output_path(
        self,
        base_name: str = "output",
        extension: str = "m4a"
    ) -> str:
        """タイムスタンプ付きの出力パスを生成

        Args:
            base_name: 出力ファイルのベース名
            extension: ファイルの拡張子

        Returns:
            str: 生成されたファイルパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"

    def record_and_analyze(
        self,
        duration: int = 10,
        speak: bool = False,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """音声の録音と分析を実行

        Args:
            duration: 録音時間（秒）
            speak: 音声合成による読み上げフラグ
            save_path: 保存先のファイル名

        Returns:
            Optional[str]: 録音ファイルのパス（成功時）
        """
        temp_recording = self.generate_output_path("recording", "wav")

        try:
            recorded_file = self.recorder.record_chunk(temp_recording, duration)
            if not recorded_file:
                return None

            print("\n録音した音声を分析しています...")
            final_save_path = None
            if save_path:
                final_save_path = self.generate_output_path(
                    os.path.splitext(save_path)[0]
                )

            self.process_file(
                recorded_file,
                delete_after=True,
                speak=speak,
                save_path=final_save_path
            )
            return recorded_file

        except KeyboardInterrupt:
            print("\n録音と分析が中断されました")
            if os.path.exists(temp_recording):
                os.remove(temp_recording)
                print(f"\n録音ファイル {temp_recording} が削除されました")
            return None

    def process_file(
        self,
        file_path: str,
        delete_after: bool = False,
        speak: bool = False,
        save_path: Optional[str] = None
    ) -> None:
        """音声/テキストファイルの処理を実行

        Args:
            file_path: 処理対象のファイルパス
            delete_after: 処理後のファイル削除フラグ
            speak: 音声合成による読み上げフラグ
            save_path: 保存先のファイル名
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            # ファイル形式に応じたセグメント抽出
            if file_ext in {'.mp3', '.wav', '.m4a', '.flac'}:
                segments = self.text_processor.segment_audio(file_path)
            elif file_ext == '.txt':
                segments = self.text_processor.segment_text(file_path)
            else:
                raise ValueError(f"未対応のファイル形式です: {file_ext}")

            if not segments:
                raise ValueError("ファイルからセグメントを抽出できませんでした")

            # 感情分析の実行
            emotion_scores = self.emotion_analyzer.analyze_emotions(segments)
            self.emotion_analyzer.print_results(segments, emotion_scores)

            # 音声合成の実行（必要な場合のみAIVISを初期化）
            if speak or save_path:
                self._initialize_aivis()
                final_save_path = None
                if save_path:
                    final_save_path = self.generate_output_path(
                        os.path.splitext(save_path)[0]
                    )

                self.analyze_and_speak(
                    segments,
                    emotion_scores,
                    save_path=final_save_path,
                    play_audio=speak
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
        """感情分析結果に基づく音声合成

        Args:
            segments: テキストセグメントのリスト
            emotion_scores: 感情スコアのリスト
            save_path: 保存先のファイル名
            play_audio: 音声再生フラグ

        Returns:
            Optional[str]: 保存されたファイルのパス（成功時）
        """
        if self._aivis is None:
            raise RuntimeError("AIVISが初期化されていません")
            
        return self._aivis.speak_continuous(
            segments,
            emotion_scores,
            save_path=save_path,
            play_audio=play_audio
        )


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """コマンドライン引数の解析

    Returns:
        tuple[argparse.ArgumentParser, argparse.Namespace]: 
            パーサーオブジェクトとパース済みの引数のタプル
    """
    parser = argparse.ArgumentParser(
        description='SentioVox: 感情分析と音声合成システム'
    )
    parser.add_argument(
        '--file',
        help='分析対象の音声/テキストファイル(.mp3, .wav, .m4a, .flac, .txt)',
        default=None
    )
    parser.add_argument(
        '--record',
        nargs='?',
        const=10,
        type=int,
        metavar='DURATION',
        help='録音時間を指定して録音を開始（デフォルト: 10秒）'
    )
    parser.add_argument(
        '--speak',
        action='store_true',
        help='音声合成による読み上げを実行'
    )
    parser.add_argument(
        '--output',
        nargs='?',
        const='output',
        metavar='FILENAME',
        help='出力音声ファイルのベース名',
        default=None
    )

    args = parser.parse_args()
    return parser, args


def main() -> None:
    """メインエントリーポイント"""
    parser, args = parse_arguments()

    try:
        with EmotionAnalysisSystem() as system:
            if args.record is not None:
                system.record_and_analyze(
                    args.record,
                    speak=args.speak,
                    save_path=args.output
                )
            elif args.file:
                system.process_file(
                    args.file,
                    speak=args.speak,
                    save_path=args.output
                )
            else:
                parser.print_help()

    except KeyboardInterrupt:
        print("\n処理が中断されました")
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()