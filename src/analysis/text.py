import torch
from functools import lru_cache
from typing import List
import spacy
import whisper
from ..utils.warnings import suppress_warnings


class TextProcessor:
    """SentioVoxのテキスト処理エンジン
    
    音声ファイルからのテキスト抽出とテキストの分割を行うクラスです。
    音声認識にはWhisperモデルを使用し、テキストの分割にはSpaCyを使用します。
    
    音声ファイルからの文字起こしと、テキストの文単位での分割の両方に対応し、
    後続の感情分析プロセスのための最適なテキストセグメントを生成します。
    """
    def __init__(self):
        self._whisper_model = None
        self._nlp = None
        self._setup_device()

    def _setup_device(self) -> None:
        """CUDA利用可能性の確認とデバイスのセットアップ"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    @lru_cache(maxsize=1)
    def whisper_model(self):
        """Whisperモデルのロードと初期化"""
        if self._whisper_model is None:
            with suppress_warnings():
                self._whisper_model = whisper.load_model(
                    "turbo",
                    device=self.device
                )
        return self._whisper_model

    @property
    @lru_cache(maxsize=1)
    def nlp(self):
        """SpaCyモデルのロードと初期化"""
        if self._nlp is None:
            self._nlp = spacy.load('ja_ginza', disable=['ner'])
            if not any(
                pipe_name == 'sentencizer'
                for pipe_name, _ in self._nlp.pipeline
            ):
                self._nlp.add_pipe('sentencizer')
        return self._nlp

    def segment_audio(self, audio_path: str) -> List[str]:
        """音声ファイルからテキストセグメントを抽出"""
        with suppress_warnings():
            segments = self.whisper_model.transcribe(
                audio_path,
                language="ja",
                word_timestamps=True
            )["segments"]
        return [seg["text"] for seg in segments]

    def segment_text(self, text_path: str) -> List[str]:
        """テキストファイルから文単位のセグメントを抽出"""
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]