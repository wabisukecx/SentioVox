"""グローバル定数の定義

このモジュールは、SentioVoxシステム全体で使用される定数を定義します。
各定数は論理的なグループに分類され、その目的、単位、推奨値などが
詳細に文書化されています。これにより、システム全体での一貫性のある
パラメータ管理が可能になります。

定数の種類：
1. ファイル出力関連 - ファイル名、拡張子など
2. 感情分析関連 - スコア閾値、モデル設定など
3. 音声処理関連 - 音声品質パラメータ
4. AIVISクライアント関連 - サーバー設定、通信パラメータ
5. 音声録音関連 - 録音品質設定
6. バッチ処理関連 - 処理効率の最適化
7. 音声スタイル関連 - 音声パラメータのプリセット
8. モデル関連 - 機械学習モデルのパラメータ
9. 音声ファイル変換関連 - エンコード設定
"""

# ファイル出力関連の定数
DEFAULT_OUTPUT_FILENAME = "output"       # デフォルトの出力ファイル名
TEMP_RECORDING_PREFIX = "recording"      # 一時録音ファイルの接頭辞
DEFAULT_OUTPUT_EXTENSION = "m4a"         # デフォルトの出力ファイル拡張子
TEMP_WAV_EXTENSION = "wav"              # 一時ファイルの拡張子
SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac'}  # 対応する音声形式

# 感情分析関連の定数
EMOTION_SCORE_THRESHOLD = 0.01     # 感情を「検出された」とみなす最小スコア
SEPARATOR_LINE = "-" * 50          # 出力結果の区切り線
MODEL_MAX_LENGTH = 512            # トークン化時の最大長
CACHE_MAX_SIZE = 1000            # 感情キャッシュの最大サイズ
CACHE_CLEANUP_SIZE = 100         # クリーンアップ時に削除するキャッシュエントリ数

# 音声処理関連の定数
SILENCE_THRESHOLD = 0.01          # 無音判定の振幅閾値（0.0-1.0の範囲）
MARGIN_SAMPLES = 100              # 無音トリミング時に保持するサンプル数
SILENCE_DURATION = 800            # セグメント間の無音の長さ（サンプル数、24kHzで約0.033秒）
TARGET_DB = -20.0                 # 音量正規化の目標デシベル値（通常-20dB～-16dB）
FADE_SAMPLES = 100                # フェードイン/アウト時のサンプル数（24kHzで約0.004秒）
MIN_SEGMENT_LENGTH = 0.1          # 最小セグメント長（秒）
MAX_SEGMENT_LENGTH = 15.0         # 最大セグメント長（秒）

# 音声分割関連の定数
SPLIT_WINDOW_SIZE = 4800          # 分割ポイント探索の窓サイズ（サンプル数、24kHzで0.2秒）
SPLIT_SMOOTHING_WINDOW = 100      # 分割時の移動平均窓サイズ（急激な変化を避けるため）
SPLIT_MARGIN = 0.1                # 分割位置の前後のマージン（秒）
MIN_SPLIT_SEGMENT = 0.5           # 分割後の最小セグメント長（秒）
MAX_AMPLITUDE_THRESHOLD = 0.8     # 分割を避ける最大振幅閾値（0.0-1.0の範囲）

# 音声品質チェック関連の定数
MIN_AUDIO_QUALITY = 0.01          # 最小音声品質（RMS値、一般的な会話音声で0.01-0.5程度）
MAX_DC_OFFSET = 0.1               # 許容される最大DCオフセット値
MIN_PEAK_THRESHOLD = 0.99         # クリッピング検出のピーク値閾値

# AIVISクライアント関連の定数
MAX_RETRIES = 3                   # HTTPリクエストの最大リトライ回数
RETRY_DELAY = 1.0                 # リトライ間の待機時間（秒）
VOLUME_SCALE = 1.2                # 基本音量スケール（1.0が標準）
MODEL_TRUNCATION = 0.8            # モデル切り捨て率（0.0-1.0の範囲、高いほど安定）
NOISE_SCALE = 0.4                 # ノイズスケール（0.0-1.0の範囲、高いほど表現が豊か）
PRE_POST_PHONEME_LENGTH = 0.1     # 音素前後の無音時間（秒）
REQUEST_TIMEOUT = 30              # APIリクエストのタイムアウト時間（秒）
MAX_TEXT_LENGTH = 1000           # 1回のリクエストで処理できる最大テキスト長

# 音声録音関連の定数
DEFAULT_CHUNK_SIZE = 1024         # 録音時のチャンクサイズ（バイト）
DEFAULT_FORMAT = "paInt16"        # PyAudioのフォーマット定数（16ビット整数）
DEFAULT_CHANNELS = 1              # 録音チャンネル数（1=モノラル）
DEFAULT_RATE = 16000             # 録音のサンプリングレート（Hz）
DEFAULT_RECORD_DURATION = 10      # デフォルトの録音時間（秒）
MONITOR_UPDATE_INTERVAL = 0.1     # レベルメーター更新間隔（秒）
LEVEL_METER_WIDTH = 50           # レベルメーター表示幅（文字数）

# バッチ処理関連の定数
DEFAULT_BATCH_SIZE = 8            # テキスト処理のデフォルトバッチサイズ
MAX_MEMORY_PERCENT = 85          # システム全体のメモリ使用率上限（%）
MEMORY_REDUCTION_FACTOR = 2      # メモリ使用量超過時の削減係数
MIN_BATCH_SIZE = 1               # 最小バッチサイズ
LENGTH_THRESHOLD_LARGE = 1000    # 長いテキストの閾値（文字数）
LENGTH_THRESHOLD_MEDIUM = 500    # 中程度のテキストの閾値（文字数）

# AIVIS関連の定数
AIVIS_BASE_URL = "http://127.0.0.1:10101"  # AIVISサーバーのベースURL
DEFAULT_OUTPUT_SAMPLING_RATE = 24000        # 出力音声のサンプリングレート（Hz）
AIVIS_PATH = r"C:\Program Files\AivisSpeech\AivisSpeech-Engine\run.exe"
AIVIS_STARTUP_TIMEOUT = 30       # AIVISサーバー起動待機時間（秒）
AIVIS_HEALTH_CHECK_INTERVAL = 1  # ヘルスチェック間隔（秒）

# モデル関連の定数
MODEL_NAME = "koshin2001/Japanese-to-emotions"  # 感情分析モデル名
WHISPER_MODEL_SIZE = "turbo"      # Whisperモデルサイズ
SPACY_MODEL = "ja_ginza"          # SpaCyモデル名
LOCAL_FILES_ONLY = True           # モデルをローカルから読み込む設定
MODEL_DEVICE_AUTO = True          # デバイス自動選択設定
MODEL_CACHE_DIR = "./models"      # モデルキャッシュディレクトリ

# 音声スタイルID関連の定数
VOICE_STYLE_IDS = {
    'NORMAL': 888753761,         # 通常
    'JOY': 888753764,            # 喜び
    'SADNESS': 888753765,        # 悲しみ
    'ANTICIPATION': 888753762,   # 期待
    'SURPRISE': 888753762,       # 驚き
    'ANGER': 888753765,          # 怒り
    'FEAR': 888753763,           # 恐れ
    'DISGUST': 888753765,        # 嫌悪
    'TRUST': 888753763           # 信頼
}

# 音声パラメータプリセット関連の定数
VOICE_PARAMS = {
    'NORMAL': {
        'intonation_scale': 1.0,
        'tempo_dynamics_scale': 1.0,
        'speed_scale': 1.0,
        'pitch_scale': 0.0,
        'volume_scale': 1.0,
        'pre_phoneme_length': 0.1,
        'post_phoneme_length': 0.1
    },
    'JOY': {
        'intonation_scale': 1.2,
        'tempo_dynamics_scale': 1.15,
        'speed_scale': 1.1,
        'pitch_scale': 0.03,
        'volume_scale': 1.1,
        'pre_phoneme_length': 0.1,
        'post_phoneme_length': 0.1
    },
    'SADNESS': {
        'intonation_scale': 0.7,
        'tempo_dynamics_scale': 0.85,
        'speed_scale': 0.9,
        'pitch_scale': -0.02,
        'volume_scale': 0.9,
        'pre_phoneme_length': 0.2,
        'post_phoneme_length': 0.1
    },
    'ANTICIPATION': {
        'intonation_scale': 1.05,
        'tempo_dynamics_scale': 1.1,
        'speed_scale': 1.05,
        'pitch_scale': 0.02,
        'volume_scale': 1.05,
        'pre_phoneme_length': 0.1,
        'post_phoneme_length': 0.1
    },
    'SURPRISE': {
        'intonation_scale': 1.3,
        'tempo_dynamics_scale': 1.2,
        'speed_scale': 1.15,
        'pitch_scale': 0.05,
        'volume_scale': 1.2,
        'pre_phoneme_length': 0.1,
        'post_phoneme_length': 0.1
    },
    'ANGER': {
        'intonation_scale': 1.3,
        'tempo_dynamics_scale': 1.2,
        'speed_scale': 1.05,
        'pitch_scale': 0.04,
        'volume_scale': 1.3,
        'pre_phoneme_length': 0.1,
        'post_phoneme_length': 0.1
    },
    'FEAR': {
        'intonation_scale': 1.1,
        'tempo_dynamics_scale': 1.1,
        'speed_scale': 1.1,
        'pitch_scale': 0.03,
        'volume_scale': 0.9,
        'pre_phoneme_length': 0.2,
        'post_phoneme_length': 0.1
    },
    'DISGUST': {
        'intonation_scale': 1.15,
        'tempo_dynamics_scale': 1.05,
        'speed_scale': 0.95,
        'pitch_scale': 0.02,
        'volume_scale': 1.1,
        'pre_phoneme_length': 0.2,
        'post_phoneme_length': 0.1
    },
    'TRUST': {
        'intonation_scale': 1.02,
        'tempo_dynamics_scale': 1.0,
        'speed_scale': 0.95,
        'pitch_scale': 0.01,
        'volume_scale': 1.0,
        'pre_phoneme_length': 0.1,
        'post_phoneme_length': 0.1
    }
}

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

# 音声ファイル変換関連の定数
AUDIO_CODEC = 'aac'                # 音声コーデック
AUDIO_BITRATE = '192k'            # 音声ビットレート
FFMPEG_LOG_LEVEL = 'error'        # FFmpegのログレベル
FFMPEG_TIMEOUT = 30               # FFmpeg処理のタイムアウト時間（秒）
MAX_AUDIO_LENGTH = 600           # 最大音声長（秒）

# 音声合成処理関連の定数
PREPROCESSING_CONFIG = {
    'normalize': True,            # 音量の正規化を行うかどうか
    'remove_dc': True,            # DCオフセットの除去を行うかどうか
    'apply_fade': True            # フェード効果を適用するかどうか
}