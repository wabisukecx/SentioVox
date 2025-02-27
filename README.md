# SentioVox: 感情認識音声合成システム

SentioVoxは、テキストや音声入力から感情を分析し、その感情に応じた表現豊かな音声を生成する高度な音声合成システムです。検出された感情に基づいて音声パラメータを自動調整し、より自然で感情豊かな音声合成を実現します。

## 主な機能

システムは以下の包括的な機能を提供します：

- **高品質な音声録音**: 設定可能なパラメータとリアルタイムレベルモニタリングによる音声入力の録音・処理
- **テキスト抽出**: Whisperモデルを使用した高精度な音声認識
- **感情分析**: 8つの基本感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）の検出と分析
- **感情に基づく音声合成**: 検出された感情に合わせてパラメータを調整した自然な音声生成
- **音声エクスポート**: 高品質なAACエンコーディングによるM4A形式での保存

### 感情パラメータマッピング

各感情は以下のパラメータに影響を与えます：

- イントネーション（抑揚）
- テンポのダイナミクス（速度変化）
- 発話速度
- ピッチ（声の高さ）
- 音量
- 音素間の間隔

感情の強度に応じて、これらのパラメータが適応的に調整されます。

## システム要件

SentioVoxを実行するために必要な環境：

- Python 3.8以上
- CUDA対応GPUを推奨（CPUモードも利用可能）
- AIVISサーバー（localhost:10101で動作）
- FFmpeg（音声ファイル変換用）

Windows以外でもパラメータを変更すれば動くと思いますが、
Windows10/11でのご利用を推奨します。

## プロジェクト構造

プロジェクトは以下のような階層構造で整理されています：

```
src/
├── __init__.py
├── main.py                 # メインスクリプト
├── analysis/              # 分析モジュール
│   ├── __init__.py
│   ├── emotion.py         # 感情分析エンジン
│   └── text.py           # テキスト処理エンジン
├── audio/                # 音声処理モジュール
│   ├── __init__.py
│   ├── aivis_client.py   # AIVISクライアント
│   ├── emotion_mapper.py # 感情-音声パラメータマッピング
│   ├── process_manager.py# AIVISプロセス管理
│   ├── processor.py      # 音声データ処理
│   ├── recorder.py       # 音声録音
│   └── synthesis.py      # 音声合成
├── models/               # モデル定義
│   ├── __init__.py
│   ├── constants.py      # システム定数
│   └── voice.py         # 音声パラメータモデル
└── utils/               # ユーティリティ
    ├── __init__.py
    └── warnings.py      # 警告抑制
```

## インストール方法

以下の手順でSentioVoxをセットアップしてください：

1. リポジトリのクローン:
```bash
git clone [リポジトリのURL]
cd SentioVox
```

2. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

3. FFmpegのインストール:

3-1. FFmpegのダウンロード:
   - [FFmpeg公式ダウンロードページ](https://www.ffmpeg.org/download.html)にアクセス
   - 「Windows Builds」のリンクをクリック
   - gyan.devのビルドから「ffmpeg-release-full.7z」をダウンロード

3-2. インストール手順:
   - ダウンロードした7zファイルを展開
   - 展開したフォルダ内の「bin」フォルダを探す
   - このフォルダを任意の場所（例：C:\Program Files\FFmpeg）に移動
   - 展開したファイルに含まれるドキュメントを参照し、必要なDLLファイルが存在することを確認

3-3. 環境変数の設定:
   - Windowsキー + Rを押して「systempropertiesadvanced」と入力
   - 「環境変数」ボタンをクリック
   - システム環境変数の「Path」を選択して「編集」をクリック
   - FFmpegのbinフォルダのパス（例：C:\Program Files\FFmpeg\bin）を追加
   - 「OK」を押して設定を保存

3-4. インストールの確認:
   ```bash
   ffmpeg -version
   ```

## 使用方法

### システムパラメータ設定

システムで使用される主要なパラメータは以下の通りです：

| カテゴリ | パラメータ名 | 値 | 説明 |
|---------|------------|-----|------|
| ファイル出力 | DEFAULT_OUTPUT_FILENAME | "output" | デフォルトの出力ファイル名 |
| | TEMP_RECORDING_PREFIX | "recording" | 一時録音ファイルの接頭辞 |
| | DEFAULT_OUTPUT_EXTENSION | "m4a" | デフォルトの出力ファイル拡張子 |
| 感情分析 | EMOTION_SCORE_THRESHOLD | 0.05 | 感情検出の最小スコア |
| 音声処理 | SILENCE_THRESHOLD | 0.01 | 無音判定の振幅閾値 |
| | MARGIN_SAMPLES | 100 | 無音トリミング時の保持サンプル数 |
| | SILENCE_DURATION | 800 | セグメント間無音の長さ（約0.033秒） |
| | TARGET_DB | -20.0 | 音量正規化の目標値 |
| | FADE_SAMPLES | 100 | フェード処理のサンプル数（約0.004秒） |
| AIVIS設定 | MAX_RETRIES | 3 | HTTPリクエストの最大リトライ回数 |
| | MODEL_TRUNCATION | 0.8 | モデル切り捨て率 |
| | NOISE_SCALE | 0.4 | ノイズスケール（表現の豊かさ） |
| 録音設定 | DEFAULT_CHUNK_SIZE | 1024 | 録音時のチャンクサイズ |
| | DEFAULT_CHANNELS | 1 | 録音チャンネル数（モノラル） |
| | DEFAULT_RATE | 16000 | 録音のサンプリングレート（Hz） |

### コマンドライン引数

SentioVoxは以下のコマンドライン引数をサポートしています：

| 引数 | 説明 | デフォルト値 |
|------|------|------------|
| `--file` | 分析対象の音声/テキストファイルを指定 | なし |
| `--record` | マイクからの録音を開始（秒数を引数として指定可能、例：--record 15） | 10秒 |
| `--speak` | 音声合成出力を有効化 | False |
| `--output` | 合成音声をM4Aファイルとして保存。タイムスタンプ付きで「output_20250201_123456.m4a」の形式で保存されます。 | なし |

### 使用例

以下に一般的な使用シナリオを示します：

1. 音声ファイルを分析し、AIVIS_Engineで音声を再生、音声ファイルとして保存:
```bash
python -m src.main --file input.mp3 --speak --output
```

2. 音声ファイルを分析し、文字起こし結果をコンソールに表示：
```bash
python -m src.main --file input.mp3
```

3. テキストファイルを分析し、音声を再生:
```bash
python -m src.main --file input.txt --speak
```

4. テキストファイルを分析し、音声ファイルとして保存:
```bash
python -m src.main --file input.txt --output
```

4. 音声を15秒間録音し、音声ファイルとして保存(ファイル名を変更):
```bash
python -m src.main --record 15 --speak --output my_recording
```

## エラーハンドリングと例外処理

システムは堅牢なエラーハンドリングを実装しており、以下のような状況に対応します：

- 音声合成エラー: セグメント単位でのリカバリと処理継続
- ファイル変換エラー: WAVフォーマットへの自動フォールバック
- メモリ使用量の監視: 動的なバッチサイズ調整
- プロセス異常終了: 適切なクリーンアップ処理
- 一時ファイルの管理: 確実な削除と後処理

## 音声出力の仕様

システムは以下の音声出力オプションを提供します：

### 自動ファイル名生成
- タイムスタンプ形式: YYYYMMDD_HHMMSS
- 複数ファイルの一意性保証
- 一時ファイルの自動クリーンアップ

### 出力フォーマット
- サンプリングレート: 24kHz（高音質出力用）
- エンコーディング: M4AコンテナでのAAC（192kbps）
- チャンネル: モノラル（音声合成に最適化）

### 音声品質パラメータ
- ダイナミックレンジ: -20dB（ターゲット音量レベル）
- フェード処理: 入出力それぞれ0.004秒
- セグメント間無音: 0.033秒（自然な間隔）
- 無音除去閾値: 0.01（振幅比）

## 注意事項

### 録音時の注意点
- 静かな録音環境を確保してください
- Ctrl+Cで録音を停止できます
- 一時ファイルは自動的に管理されます

### 感情分析について
- 8つの基本感情を処理します
- 感情スコアは0から1の範囲です
- スコアが0.05未満の感情は表示されません

### 音声合成について
- AIVISサーバーが必要です（デフォルト: localhost:10101）
- 感情内容に応じて音声パラメータが適応的に調整されます
- 複数の感情が自然にブレンドされます
- M4A変換にはFFmpegが必要です

## トラブルシューティング

一般的な問題と解決方法：

### 音声ファイル変換の問題
- FFmpegのインストールを確認
- PATH環境変数を確認
- M4A変換失敗時はWAVにフォールバック
- FFmpegのバージョンの互換性を確認

### メモリ管理
- 長い入力に対してバッチサイズが自動調整されます
- メモリ警告が出た場合は入力を分割してください
- 処理中のシステムリソースを監視してください
- 大規模なファイル処理時はGPUメモリの使用状況に注意

### AIVIS接続の問題
- AIVISサーバーの状態を確認
- サーバーのアドレス/ポート設定を確認
- ネットワーク接続を確認
- ファイアウォールの設定を確認

## パフォーマンス最適化

システムには以下の最適化機能が含まれています：

- バッチサイズの自動調整
- GPU利用可能時の自動高速化
- 感情分析のキャッシング
- 効率的なリソース管理
- 設定可能な処理パラメータ

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 開発者

- wabisuke - 初期開発

## 謝辞

本プロジェクトは以下のプロジェクトやチームの成果を活用しています：

- Whisperチーム - 高精度な音声認識モデル
- GiNZAチーム - 高性能な日本語自然言語処理ライブラリ
- SpaCyチーム - 自然言語処理の基盤ライブラリ
- koshin2001氏 - 日本語感情分析モデル「Japanese-to-emotions」
- AIVISチーム - 高品質な音声合成エンジン
- FFmpegプロジェクト - 音声フォーマット変換ツール
