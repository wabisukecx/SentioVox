import json
import argparse
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax
import numpy as np

# 感情ラベル
EMOTION_LABELS = [
    "喜び",     # Joy
    "悲しみ",   # Sadness
    "期待",     # Anticipation
    "驚き",     # Surprise
    "怒り",     # Anger
    "恐れ",     # Fear
    "嫌悪",     # Disgust
    "信頼"      # Trust
]

# 感情検出の閾値
EMOTION_SCORE_THRESHOLD = 0.05

class EmotionAnalyzer:
    """テキストの感情分析を実行するクラス"""
    
    def __init__(self, model_name="koshin2001/Japanese-to-emotions", local_files_only=False):
        """感情分析モデルを初期化

        Args:
            model_name: 使用するモデル名
            local_files_only: ローカルファイルのみを使用するかどうか
        """
        self._tokenizer = None
        self._model = None
        self._emotion_cache = {}
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._setup_device()

    def _setup_device(self) -> None:
        """CUDAが利用可能か確認しデバイスを設定"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print(f"GPU使用: {torch.cuda.get_device_name(0)}")
        else:
            print("CPU処理を使用します。")

    @property
    def tokenizer(self):
        """トークナイザーを取得（初回利用時に初期化）"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only
            )
        return self._tokenizer

    @property
    def model(self):
        """モデルを取得（初回利用時に初期化）"""
        if self._model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only
            )
            self._model = model.to(self.device)
        return self._model

    def _process_single_text(self, text: str) -> np.ndarray:
        """単一テキストの感情分析を実行

        Args:
            text: 分析対象のテキスト
            
        Returns:
            np.ndarray: 8つの感情に対するスコア配列
        """
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
        """テキストリストの感情分析を実行

        Args:
            texts: 分析対象のテキストリスト
            
        Returns:
            List[List[float]]: 各テキストの感情スコアリスト
        """
        print(f"\n感情分析を開始: 合計 {len(texts)} テキスト")
        results = []
        
        for i, text in enumerate(texts, 1):
            print(f"テキスト {i}/{len(texts)} を処理中...")
            
            if text in self._emotion_cache:
                results.append(self._emotion_cache[text])
            else:
                try:
                    score = self._process_single_text(text)
                    self._emotion_cache[text] = score
                    results.append(score)
                except Exception as e:
                    print(f"警告: テキスト処理中にエラー発生: {str(e)}")
                    # エラーが発生した場合は中立的な感情スコアを設定
                    neutral_score = np.ones(len(EMOTION_LABELS)) / len(EMOTION_LABELS)
                    results.append(neutral_score)
        
        print(f"\n感情分析完了: {len(results)}/{len(texts)} テキストを処理")
        return results

    def format_emotion_results(self, scores: List[float]) -> Dict[str, float]:
        """感情スコアを辞書形式にフォーマット

        Args:
            scores: 感情スコアのリスト
            
        Returns:
            Dict[str, float]: 閾値以上のスコアを持つ感情とスコアの辞書
        """
        emotion_dict = {}
        for emotion, score in zip(EMOTION_LABELS, scores):
            if score >= EMOTION_SCORE_THRESHOLD:
                emotion_dict[emotion] = float(score)
        
        # 感情が検出されなかった場合は「中立」としてマーク
        if not emotion_dict:
            emotion_dict["中立"] = 1.0
            
        return emotion_dict

class ScytheEmotionAnalyzer:
    """Scythe会話データに感情分析結果を追加するクラス"""
    
    def __init__(self, input_file: str, output_file: str = None):
        """初期化

        Args:
            input_file: 入力JSONファイルのパス
            output_file: 出力JSONファイルのパス
        """
        self.input_file = input_file
        self.output_file = output_file or input_file.replace('.json', '_with_emotions.json')
        self.data = None
        self.emotion_analyzer = EmotionAnalyzer()
    
    def load_data(self) -> None:
        """JSONファイルからデータを読み込む"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"{len(self.data)} 件の会話データを読み込みました")
        except Exception as e:
            raise RuntimeError(f"JSONファイルの読み込みに失敗しました: {str(e)}")
    
    def analyze_conversations(self) -> None:
        """会話データに対して感情分析を実行"""
        if self.data is None:
            self.load_data()
        
        texts = [item["text"] for item in self.data]
        
        # 感情分析の実行
        print("感情分析モデルを使用します")
        emotion_scores = self.emotion_analyzer.analyze_emotions(texts)
        
        for i, scores in enumerate(emotion_scores):
            emotion_results = self.emotion_analyzer.format_emotion_results(scores)
            self.data[i]["emotions"] = emotion_results
            
            # 最も強い感情を dominant_emotion として追加
            if scores.any():
                dominant_idx = scores.argmax()
                self.data[i]["dominant_emotion"] = EMOTION_LABELS[dominant_idx]
            else:
                self.data[i]["dominant_emotion"] = "中立"
    
    def save_data(self) -> None:
        """更新したデータをJSONファイルに保存"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"感情分析結果を追加したデータを {self.output_file} に保存しました")
        except Exception as e:
            print(f"JSONファイルの保存に失敗しました: {str(e)}")
    
    def process(self) -> None:
        """データの読み込み、感情分析、保存を実行"""
        self.load_data()
        self.analyze_conversations()
        self.save_data()
        
        # 分析結果のサンプルを表示
        self._print_sample_results()
    
    def _print_sample_results(self, sample_size: int = 5) -> None:
        """分析結果のサンプルを表示

        Args:
            sample_size: 表示するサンプル数
        """
        if not self.data:
            return
            
        print("\n感情分析結果のサンプル:")
        for i, item in enumerate(self.data[:sample_size]):
            print("-" * 50)
            print(f"話者: {item['speaker']}")
            print(f"テキスト: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
            print(f"主要な感情: {item['dominant_emotion']}")
            print("検出された感情:")
            
            # 検出された感情を降順で表示
            for emotion, score in sorted(item["emotions"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score:.3f}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Scythe会話データに感情分析結果を追加')
    parser.add_argument('--input', default='scythe.json', help='入力JSONファイルのパス')
    parser.add_argument('--output', help='出力JSONファイルのパス（デフォルト: input_with_emotions.json）')
    
    args = parser.parse_args()
    
    try:
        analyzer = ScytheEmotionAnalyzer(
            input_file=args.input,
            output_file=args.output
        )
        analyzer.process()
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()