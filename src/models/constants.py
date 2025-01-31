# 感情分析関連の定数
EMOTION_SCORE_THRESHOLD = 0.05
SEPARATOR_LINE = "-" * 50

# 音声録音関連の定数
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_FORMAT = "paInt16"  # PyAudioのフォーマット定数
DEFAULT_CHANNELS = 1
DEFAULT_RATE = 16000

# AIVIS関連の定数
AIVIS_BASE_URL = "http://127.0.0.1:10101"
AIVIS_PATH = r"C:\Program Files\AivisSpeech\AivisSpeech-Engine\run.exe"
DEFAULT_OUTPUT_SAMPLING_RATE = 24000

# 感情ラベル
EMOTION_LABELS = [
    "喜び", "悲しみ", "期待", "驚き",
    "怒り", "恐れ", "嫌悪", "信頼"
]

# バッチ処理関連
DEFAULT_BATCH_SIZE = 8

# メモリ管理関連
MAX_MEMORY_PERCENT = 85  # システム全体のメモリ使用率の上限（%）