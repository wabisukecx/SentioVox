# SentioVox: 
感情認識音声合成システム

## 概要

SentioVoxは、テキストや音声入力から感情を分析し、その感情に応じた表現豊かな音声を生成する高度な音声合成システムです。検出された感情に基づいて音声パラメータを自動調整し、より自然で感情豊かな音声合成を実現します。

## システムの特徴

SentioVoxは以下の包括的な機能を提供します：

- **高品質な音声録音**  
  設定可能なパラメータとリアルタイムレベルモニタリングによる音声入力の録音・処理を実現します。

- **テキスト抽出**  
  Whisperモデルを使用した高精度な音声認識により、正確なテキスト化を行います。

- **感情分析**  
  8つの基本感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）を検出・分析します。

- **感情に基づく音声合成**  
  検出された感情に合わせてパラメータを調整し、自然な音声を生成します。

- **音声エクスポート**  
  高品質なAACエンコーディング（192kbps）によるM4A形式での保存に対応しています。

## 動作環境

### システム要件

- Python 3.8以上
- CUDA対応GPUを推奨（CPUモードも利用可能）
- AIVISサーバー（localhost:10101で動作）
- FFmpeg（音声ファイル変換用）

## セットアップガイド

### 1. リポジトリのクローン

```bash
git clone [リポジトリのURL]
cd SentioVox
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
python -m spacy download ja_ginza
```

### 3. FFmpegのインストール

各環境に応じて以下の手順でインストールしてください。

#### Ubuntu/Debian環境

```bash
sudo apt-get install ffmpeg
```

#### macOS環境

```bash
brew install ffmpeg
```

#### Windows環境

1. **FFmpegのダウンロード**
   - [FFmpeg公式ダウンロードページ](https://www.ffmpeg.org/download.html)にアクセス
   - 「Windows Builds」から「ffmpeg-release-full.7z」をダウンロード

2. **インストール手順**
   - 7zファイルを展開
   - binフォルダを任意の場所（例：`C:\Program Files\FFmpeg`）に配置

3. **環境変数の設定**
   - Windowsキー + R → 「systempropertiesadvanced」
   - 環境変数 → システム環境変数のPath → 編集
   - FFmpegのbinフォルダのパスを追加
   - すべての画面でOKをクリック

4. **インストールの確認**
   ```bash
   ffmpeg -version
   ```

## プロジェクト構造

システムは以下の階層構造で整理されています：

```plaintext
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

### モジュールの役割

- **main.py**  
  システム全体のエントリーポイントとコマンドライン引数の処理を担当します。

- **analysis/**  
  テキストの感情分析と音声認識を実行する中核モジュールです。
  - `emotion.py`: 感情分析エンジン
  - `text.py`: 音声認識とテキスト分割

- **audio/**  
  音声処理の基幹機能を提供します。
  - `aivis_client.py`: AIVIS通信制御
  - `emotion_mapper.py`: 感情パラメータ変換
  - `process_manager.py`: プロセス管理
  - `processor.py`: 音声データ処理
  - `recorder.py`: 音声録音制御
  - `synthesis.py`: 音声合成実行

- **models/**  
  データ構造とシステム定数を定義します。

- **utils/**  
  共通ユーティリティ機能を提供します。

## 使用方法

### コマンドライン引数

```bash
python -m src.main [オプション]
```

利用可能なオプション：

| オプション | 説明 | 例 |
|------------|------|-----|
| `--file` | 分析対象ファイル指定 | `--file input.txt` |
| `--record` | 録音モード開始 | `--record` |
| `--duration` | 録音時間（秒） | `--duration 15` |
| `--speak` | 音声合成有効化 | `--speak` |
| `--output` | 出力ファイル指定 | `--output out.m4a` |
| `--no-play` | 再生無効化 | `--no-play` |

### 使用例

1. **テキスト分析と音声再生**
   ```bash
   python -m src.main --file input.txt --speak
   ```

2. **テキスト分析と音声保存**
   ```bash
   python -m src.main --file input.txt --speak --output output.m4a --no-play
   ```

3. **音声録音と分析**
   ```bash
   python -m src.main --record --duration 15 --speak --output recording.m4a
   ```

4. **既存音声の感情強調**
   ```bash
   python -m src.main --file input.wav --speak --output enhanced.m4a
   ```

## 音声出力仕様

### 1. リアルタイム再生

- `--speak`オプションで有効化
- 感情に基づく自動パラメータ調整
- 即時フィードバック機能

### 2. ファイルエクスポート

- 形式: M4A（AAC, 192kbps）
- 出力指定: `--output`オプション
- 自動ファイル名生成対応

### 3. 音声品質

- サンプリングレート: 24kHz
- チャンネル: モノラル
- ビットレート: 192kbps（AAC）

## 運用上の注意事項

### 録音環境の整備

音声認識の精度は録音品質に大きく依存するため、以下の点に特に注意を払ってください：

1. **環境ノイズの管理**
   - 窓やドアを閉めて外部騒音を遮断することを推奨します
   - マイクから30cm程度の距離を保ち、適切な音量で録音を行ってください

2. **録音操作の制御**
   - 録音中はCtrl+Cでいつでも安全に停止できます
   - 停止時は自動的にそれまでの録音データが保存されます
   - 予期せぬ中断時もデータ損失を防ぐため、30秒ごとに自動バックアップを作成します

3. **一時ファイルの取り扱い**
   - 録音データは一時的に`temp_recordings`フォルダに保存されます
   - 処理完了後、一時ファイルは自動的に削除されます
   - システムクラッシュ時は`cleanup.py`スクリプトで手動クリーンアップが可能です

### 感情分析の精度向上

感情分析の精度を最大限に引き出すため、以下のポイントに注意してください：

1. **テキストの品質**
   - 文末の句読点を適切に配置してください
   - 一文が長すぎる場合は、適切な長さに分割することを推奨します
   - 略語や特殊な表現は正式な表現に置き換えることで精度が向上します

2. **感情スコアの解釈**
   - スコアは0.0から1.0の範囲で出力されます
   - 0.05未満のスコアは信頼性が低いため表示されません
   - 複数の感情が検出された場合は、上位2～3個に注目することを推奨します

3. **分析結果の検証**
   - 分析結果は常にコンテキストを考慮して解釈してください
   - 予期せぬ結果が出た場合は、入力テキストの見直しを検討してください
   - 継続的な精度向上のため、フィードバックを収集しています

### 音声合成の最適化

高品質な音声合成を実現するため、以下の設定と注意点を確認してください：

1. **AIVISサーバーの設定**
   - サーバーは常時localhost:10101で稼働している必要があります
   - メモリ使用量が4GB以上の場合は自動的に警告が表示されます
   - サーバーの再起動が必要な場合は、現在の処理の完了を待ってから実行してください

2. **感情パラメータの調整**
   - パラメータは`models/voice.py`で細かく調整可能です
   - 極端なパラメータ設定は音声の品質低下を招く可能性があります
   - 新しいパラメータセットを試す場合は、必ずバックアップを作成してください

3. **出力フォーマットの管理**
   - FFmpegは最新バージョンの使用を推奨します
   - M4A出力失敗時は自動的にWAVフォーマットにフォールバックします
   - 出力ファイルのビットレートは必要に応じて`constants.py`で調整可能です

## トラブルシューティング

### 1. 音声変換に関する問題

#### 症状と対策
- **M4Aファイルが生成されない**
  1. FFmpegのインストール状態を確認：
     ```bash
     ffmpeg -version
     ```
  2. PATH環境変数に正しく登録されているか確認：
     ```bash
     echo $PATH  # Unix系
     echo %PATH% # Windows
     ```
  3. FFmpegのログを確認：
     ```bash
     python -m src.main --file input.txt --speak --output test.m4a --verbose
     ```

- **音声品質が低い**
  1. 入力音声のサンプリングレートを確認
  2. 一時的なWAVファイルの品質を検証
  3. エンコードパラメータを調整：
     ```python
     # constants.pyの該当箇所を編集
     DEFAULT_OUTPUT_SAMPLING_RATE = 24000
     DEFAULT_BITRATE = "192k"
     ```

### 2. メモリ管理の問題

#### 原因と解決策
- **処理が遅くなる・クラッシュする**
  1. メモリ使用状況の確認：
     ```python
     import psutil
     print(f"メモリ使用率: {psutil.virtual_memory().percent}%")
     ```
  2. バッチサイズの調整：
     ```python
     # constants.pyで設定
     DEFAULT_BATCH_SIZE = 4  # 必要に応じて減少
     ```
  3. 大規模入力の分割処理：
     ```python
     # 推奨される最大テキスト長
     MAX_TEXT_LENGTH = 1000  # 文字
     ```
## パフォーマンス最適化

システムは以下の最適化機能を実装しています：

- バッチ処理の自動最適化
- GPU活用による高速化
- 感情分析結果のキャッシュ
- リソース使用効率の改善
- カスタマイズ可能なパラメータ

## ライセンス

本プロジェクトはMITライセンスの下で公開されています。  
詳細は付属のLICENSEファイルをご確認ください。

## 開発者・謝辞

### 開発者
- wabisuke（初期開発）

### 謝辞
- Whisperチーム（音声認識モデル）
- SpaCyチーム（自然言語処理）
- AIVISチーム（音声合成）
- FFmpegプロジェクト（音声変換）
