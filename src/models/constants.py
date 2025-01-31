"""グローバル定数の定義

このモジュールは、SentioVoxシステム全体で使用される定数を定義します。
各定数は論理的なグループに分類され、その目的、単位、推奨値などが
詳細に文書化されています。これにより、システム全体での一貫性のある
パラメータ管理が可能になります。
"""

# 感情分析関連の定数
EMOTION_SCORE_THRESHOLD = 0.05    # 感情を「検出された」とみなす最小スコア
SEPARATOR_LINE = "-" * 50         # 出力結果の区切り線

# 音声処理関連の定数
SILENCE_THRESHOLD = 0.01          # 無音判定の振幅閾値（0.0-1.0の範囲）
MARGIN_SAMPLES = 100              # 無音トリミング時に保持するサンプル数
SILENCE_DURATION = 800            # セグメント間の無音の長さ（サンプル数、24kHzで約0.033秒）
TARGET_DB = -20.0                 # 音量正規化の目標デシベル値（通常-20dB～-16dB）
FADE_SAMPLES = 100                # フェードイン/アウト時のサンプル数（24kHzで約0.004秒）

# AIVISクライアント関連の定数
MAX_RETRIES = 3                   # HTTPリクエストの最大リトライ回数
RETRY_DELAY = 1.0                 # リトライ間の待機時間（秒）
VOLUME_SCALE = 1.2                # 基本音量スケール（1.0が標準）
MODEL_TRUNCATION = 0.8            # モデル切り捨て率（0.0-1.0の範囲、高いほど安定）
NOISE_SCALE = 0.4                 # ノイズスケール（0.0-1.0の範囲、高いほど表現が豊か）
PRE_POST_PHONEME_LENGTH = 0.1     # 音素前後の無音時間（秒）

# 音声録音関連の定数
DEFAULT_CHUNK_SIZE = 1024         # 録音時のチャンクサイズ（バイト）
DEFAULT_FORMAT = "paInt16"        # PyAudioのフォーマット定数（16ビット整数）
DEFAULT_CHANNELS = 1              # 録音チャンネル数（1=モノラル）
DEFAULT_RATE = 16000             # 録音のサンプリングレート（Hz）

# AIVIS関連の定数
AIVIS_BASE_URL = "http://127.0.0.1:10101"  # AIVISサーバーのベースURL
DEFAULT_OUTPUT_SAMPLING_RATE = 24000        # 出力音声のサンプリングレート（Hz）

# バッチ処理関連の定数
DEFAULT_BATCH_SIZE = 8            # テキスト処理のデフォルトバッチサイズ
MAX_MEMORY_PERCENT = 85          # システム全体のメモリ使用率上限（%）

# 感情ラベル（日本語）
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