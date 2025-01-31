import psutil
from functools import lru_cache
from typing import List, Dict
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from ..utils.warnings import suppress_warnings
from ..models.constants import (
    EMOTION_SCORE_THRESHOLD,
    SEPARATOR_LINE,
    EMOTION_LABELS,
    DEFAULT_BATCH_SIZE,
    MAX_MEMORY_PERCENT
)


class EmotionAnalyzer:
    """SentioVoxの感情分析エンジン"""
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._emotion_cache = {}
        self._setup_device()

    def _setup_device(self) -> None:
        """CUDA利用可能性の確認とデバイスのセットアップ"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"GPU使用: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA利用不可。CPU処理を使用します。")
        
        self.initial_memory = psutil.Process().memory_info().rss

    def _check_memory_usage(self) -> bool:
        """メモリ使用量のチェック"""
        current_memory = psutil.Process().memory_info().rss
        memory_percent = psutil.virtual_memory().percent
        memory_increase = (current_memory - self.initial_memory) / self.initial_memory
        
        return memory_percent > MAX_MEMORY_PERCENT or memory_increase > 1.0

    def _get_optimal_batch_size(self, texts: List[str]) -> int:
        """最適なバッチサイズの決定"""
        if not texts:
            return DEFAULT_BATCH_SIZE

        avg_length = sum(len(text) for text in texts) / len(texts)
        
        if avg_length > 1000:
            return max(1, DEFAULT_BATCH_SIZE // 4)
        elif avg_length > 500:
            return max(1, DEFAULT_BATCH_SIZE // 2)
        return DEFAULT_BATCH_SIZE

    @property
    @lru_cache(maxsize=1)
    def tokenizer(self):
        """トークナイザーの初期化"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "koshin2001/Japanese-to-emotions",
                local_files_only=True
            )
        return self._tokenizer

    @property
    @lru_cache(maxsize=1)
    def model(self):
        """モデルの初期化"""
        if self._model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                "koshin2001/Japanese-to-emotions",
                local_files_only=True
            )
            self._model = model.to(self.device)
        return self._model

    def _process_single_text(self, text: str) -> np.ndarray:
        """単一テキストの処理"""
        with suppress_warnings():
            inputs = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            score = softmax(outputs.logits, dim=1).cpu().numpy()[0]
            return score.astype(float)

    def analyze_emotions(self, texts: List[str]) -> List[List[float]]:
        """テキストの感情分析を実行"""
        print(f"\n感情分析を開始: 合計 {len(texts)} テキスト")
        results = []
        remaining_texts = texts.copy()  # 処理待ちのテキスト
        batch_size = self._get_optimal_batch_size(texts)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        current_batch = 1

        while remaining_texts:
            if self._check_memory_usage():
                batch_size = max(1, batch_size // 2)
                print(f"メモリ使用量調整: バッチサイズを {batch_size} に変更")
            
            batch_texts = remaining_texts[:batch_size]
            remaining_texts = remaining_texts[batch_size:]
            print(f"\nバッチ {current_batch}/{total_batches} を処理中...")
            current_batch += 1
            
            batch_results = []
            for text in batch_texts:
                if text in self._emotion_cache:
                    batch_results.append(self._emotion_cache[text])
                else:
                    try:
                        score = self._process_single_text(text)
                        self._emotion_cache[text] = score
                        batch_results.append(score)
                    except Exception as e:
                        print(f"警告: テキスト処理中にエラー発生: {str(e)}")
                        # エラーが発生した場合は中立的な感情スコアを設定
                        neutral_score = np.ones(len(EMOTION_LABELS)) / len(EMOTION_LABELS)
                        batch_results.append(neutral_score)
            
            results.extend(batch_results)
            progress = len(results)
            print(f"進捗状況: {progress}/{len(texts)} テキスト処理済み ({progress/len(texts)*100:.1f}%)")
            
            # キャッシュサイズの管理
            if len(self._emotion_cache) > 1000:
                old_keys = list(self._emotion_cache.keys())[:100]
                for key in old_keys:
                    del self._emotion_cache[key]

        # 未処理のテキストがないか最終確認
        if len(results) < len(texts):
            print("\n警告: 一部のテキストが未処理です。再処理を試みます...")
            missing_count = len(texts) - len(results)
            missing_texts = texts[len(results):]
            for text in missing_texts:
                try:
                    score = self._process_single_text(text)
                    results.append(score)
                    print(f"テキスト再処理成功: {text[:30]}...")
                except Exception as e:
                    print(f"警告: テキスト再処理中にエラー発生: {str(e)}")
                    neutral_score = np.ones(len(EMOTION_LABELS)) / len(EMOTION_LABELS)
                    results.append(neutral_score)
        
        print(f"\n感情分析完了: {len(results)}/{len(texts)} テキストを処理")
        return results

    def print_results(self, segments: List[str], emotion_scores: List[List[float]]) -> None:
        """感情分析結果の表示"""
        print("\n感情分析結果:\n")
        for i, (text, scores) in enumerate(zip(segments, emotion_scores)):
            print(f"セグメント {i+1}:")
            print(f"テキスト: {text}")
            print("検出された感情:")
            emotion_pairs = list(zip(EMOTION_LABELS, scores))
            sorted_emotions = sorted(
                emotion_pairs,
                key=lambda x: x[1],
                reverse=True
            )
            for emotion, score in sorted_emotions:
                if score >= EMOTION_SCORE_THRESHOLD:
                    print(f" {emotion}: {score:.3f}")
            dominant_emotion = EMOTION_LABELS[scores.argmax()]
            print(f"主要な感情: {dominant_emotion}")
            print(SEPARATOR_LINE)