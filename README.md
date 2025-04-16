# SentioVox: 感情認識音声合成システム

## 主な機能

システムは以下の包括的な機能を提供します：

- **高品質な音声録音**: 設定可能なパラメータとリアルタイムレベルモニタリングによる音声入力の録音・処理
- **テキスト抽出**: Whisperモデルを使用した高精度な音声認識
- **感情分析**: 8つの基本感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）の検出と分析
- **感情に基づく音声合成**: 検出された感情に合わせてパラメータを調整した自然な音声生成
- **音声エクスポート**: 高品質なAACエンコーディングによるM4A形式での保存
- **JSONベース会話処理**: 複数キャラクターの会話を一括して処理
- **グラフィカルユーザーインターフェース**: 直感的な操作が可能なStreamlit UI
- **統合コマンドライン**: 一元化された使いやすいコマンドラインインターフェース

### 感情パラメータマッピング

各感情は以下のパラメータに影響を与えます：

- イントネーション（抑揚）
- テンポのダイナミクス（速度変化）
- 発話速度
- ピッチ（声の高さ）
- 音量
- 音素間の間隔

感情の強度に応じて、これらのパラメータが適応的に調整されます。

## 統合コマンドラインインターフェース

SentioVoxでは、すべての機能に統一されたインターフェースでアクセスできます：

```bash
python -m src.sentiovox <サブコマンド> [オプション]
```

### サブコマンド

- **process**: 音声/テキストファイルを処理（従来のモード）
  ```bash
  python -m src.sentiovox process --file input.mp3 --speak --output
  ```

- **json**: JSONファイルを処理
  ```bash
  python -m src.sentiovox json --file dialogue.json --analyze --synthesize
  ```

詳細なオプションは各サブコマンドのヘルプで確認できます：
```bash
python -m src.sentiovox <サブコマンド> --help
```

## Streamlit UI

SentioVoxには、グラフィカルユーザーインターフェース（GUI）としてStreamlitアプリケーションが含まれています。このインターフェースでは、以下の操作が可能です：

1. **感情分析**: JSONファイルの感情分析を実行
2. **データ読み込み**: JSONフォーマットの会話データの読み込みとプレビュー
3. **音声設定**: キャラクターと話者のマッピング、感情ごとの話者スタイル設定
4. **音声合成**: 範囲指定、感情パラメータの調整、音声合成と試聴

### Streamlit UIの起動

以下のコマンドでStreamlit UIを起動できます：

```bash
# 統合コマンド
streamlit run src/ui/streamlit_app.py --server.fileWatcherType none
```

ブラウザが自動的に開き、`http://localhost:8501` でアプリケーションにアクセスできます。

### 対応するJSONフォーマット

アプリケーションは以下の形式のJSONファイルに対応しています：

```json
[
    {
        "speaker": "キャラクター名",
        "text": "セリフ内容",
        "dominant_emotion": "主要感情（オプション）",
        "emotions": {
            "感情名1": 0.5,
            "感情名2": 0.3,
            ...
        }
    },
    ...
]
```

- `speaker` と `text` は必須です
- `dominant_emotion` は主要感情を示す文字列で、オプションです
- `emotions` は感情と強度の連想配列で、オプションです

### 設定の保存と読み込み

キャラクターと話者のマッピング設定はJSONファイルとして保存でき、後で再利用できます。デフォルトでは、入力JSONファイルと同じベース名に`_settings.json`を付加した名前で保存されます。

## JSON処理コマンド

JSONファイルの処理コマンドが強化されました：

```bash
python -m src.sentiovox json --file dialogue.json [オプション]
```

### 主なオプション

| オプション | 説明 | デフォルト値 |
|---------|------|------------|
| `--file` | 処理するJSONファイル | 必須 |
| `--output` | 出力JSONファイルのパス | 自動生成 |
| `--analyze` | JSONファイルに感情分析を実行 | False |
| `--synthesize` | JSONファイルから音声合成を実行 | False |
| `--mapping` | 話者マッピング設定ファイル | なし |
| `--output-dir` | 音声ファイルの出力ディレクトリ | output |
| `--start-index` | 開始インデックス | 0 |
| `--end-index` | 終了インデックス | なし(最後まで) |

### 使用例

1. JSONファイルの感情分析を実行：
```bash
python -m src.sentiovox json --file dialogue.json --analyze
```

2. 感情分析済みJSONファイルから音声合成を実行：
```bash
python -m src.sentiovox json --file dialogue_with_emotions.json --synthesize
```

3. 特定の範囲のみを処理：
```bash
python -m src.sentiovox json --file dialogue_with_emotions.json --synthesize --start-index 5 --end-index 10
```

4. 話者マッピングを指定して音声合成を実行：
```bash
python -m src.sentiovox json --file dialogue_with_emotions.json --synthesize --mapping mapping.json
```

## システム要件

SentioVoxを実行するために必要な環境：

- Python 3.8以上
- CUDA対応GPUを推奨（CPUモードも利用可能）
- AIVISサーバー（localhost:10101で動作）
- FFmpeg（音声ファイル変換用）
- Streamlit 1.30.0以上（UIモード利用時）

Windows以外でもパラメータを変更すれば動くと思いますが、
Windows10/11でのご利用を推奨します。

## プロジェクト構造

プロジェクトは以下のような階層構造で整理されています：

```
src/
├── __init__.py
├── main.py                 # メインスクリプト
├── ui_main.py              # Streamlit UI用エントリーポイント
├── sentiovox.py            # 統合コマンドラインエントリーポイント
├── ui/                     # UI層
│   ├── __init__.py
│   └── streamlit_app.py    # Streamlit UIの実装
├── commands/               # コマンドモジュール
│   ├── __init__.py
│   └── process_json.py     # JSON処理コマンド
├── analysis/               # 分析モジュール
│   ├── __init__.py
│   ├── emotion.py          # 感情分析エンジン
│   ├── text.py             # テキスト処理エンジン
│   ├── json_dialogue.py    # JSON処理モジュール
│   └── json_emotion_processor.py # JSON感情処理
├── audio/                  # 音声処理モジュール
│   ├── __init__.py
│   ├── aivis_client.py     # AIVISクライアント
│   ├── emotion_mapper.py   # 感情-音声パラメータマッピング
│   ├── process_manager.py  # AIVISプロセス管理
│   ├── processor.py        # 音声データ処理
│   ├── recorder.py         # 音声録音
│   ├── synthesis.py        # 音声合成
│   └── json_synthesis.py   # JSON音声合成モジュール
├── models/                 # モデル定義
│   ├── __init__.py
│   ├── constants.py        # システム定数
│   └── voice.py            # 音声パラメータモデル
└── utils/                  # ユーティリティ
    ├── __init__.py
    ├── warnings.py         # 警告抑制
    └── aivis_utils.py      # AIVIS関連ユーティリティ
```

## インストール方法

以下の手順でSentioVoxをセットアップしてください：

1. リポジトリのクローン:
```bash
git clone https://github.com/wabisukecx/SentioVox.git
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

### 統合コマンドラインモード

最も簡単な使用方法は統合コマンドラインインターフェースを使用することです：

```bash
python -m src.sentiovox <サブコマンド> [オプション]
```

サブコマンドの一覧を表示するには：

```bash
python -m src.sentiovox --help
```

### 従来のコマンドラインモード

以下のコマンドライン引数をサポートしています：

| 引数 | 説明 | デフォルト値 |
|------|------|------------|
| `--file` | 分析対象の音声/テキストファイルを指定 | なし |
| `--record` | マイクからの録音を開始（秒数を引数として指定可能、例：--record 15） | 10秒 |
| `--speak` | 音声合成出力を有効化 | False |
| `--output` | 合成音声をM4Aファイルとして保存。タイムスタンプ付きで「output_20250201_123456.m4a」の形式で保存されます。 | なし |

### 使用例

以下に一般的な使用シナリオを示します：

1. Streamlit UIを起動:
```bash
python -m src.sentiovox ui
```

2. 音声ファイルを分析し、AIVIS_Engineで音声を再生、音声ファイルとして保存:
```bash
python -m src.sentiovox process --file input.mp3 --speak --output
```

3. 音声ファイルを分析し、文字起こし結果をコンソールに表示：
```bash
python -m src.main --file input.mp3
```

4. テキストファイルを分析し、音声を再生:
```bash
python -m src.main --file input.txt --speak
```

5. テキストファイルを分析し、音声ファイルとして保存:
```bash
python -m src.main --file input.txt --output
```

6. 音声を15秒間録音し、音声ファイルとして保存(ファイル名を変更):
```bash
python -m src.main --record 15 --speak --output my_recording
```

7. JSONファイルに感情分析を実行:
```bash
python -m src.sentiovox json --file dialogue.json --analyze
```

8. 感情分析済みJSONファイルから音声合成を実行:
```bash
python -m src.sentiovox json --file dialogue_with_emotions.json --synthesize
```

### Streamlit UIモード

Streamlit UIを使用する場合は以下のコマンドを実行します：

```bash
python -m src.sentiovox ui
```

ブラウザで自動的にUIが開きます。

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

### Streamlit UI関連の問題
- 正しいPythonバージョンの使用を確認
- Streamlitのインストール状態を確認
- ポート8501が利用可能かを確認
- ブラウザのキャッシュをクリア

### JSONデータ処理の問題
- JSONフォーマットが正しいか確認
- 必須フィールド（speaker, text）が存在するか確認
- 感情分析前に大きなJSONファイルを分割することを検討
- 話者マッピングファイルの形式を確認

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
- Streamlitチーム - インタラクティブなWebアプリケーションフレームワーク

## 更新履歴

### V1.1 - JSONベース会話処理とStreamlit UI導入 (2025年4月)

SentioVox V1.1では、以下の機能強化を行いました：

- **Streamlit UIの追加**: テキストベースのコマンドラインインターフェースに加え、直感的なグラフィカルインターフェースを導入
- **JSONベースの会話処理**: 複数キャラクターの会話データをJSONフォーマットで一括処理する機能
- **キャラクターと話者のマッピング**: キャラクターごとに異なる話者を割り当て可能
- **感情ごとの話者スタイル設定**: 同一キャラクターの異なる感情に対して個別の話者スタイルを適用可能
- **感情パラメータの詳細調整**: GUIでの直感的な音声パラメータ調整
- **設定の保存と再利用**: 話者マッピングと感情設定を保存して再利用可能
- **範囲指定合成**: 指定した範囲のみを合成する機能
- **合成音声のインラインプレビュー**: ブラウザ上で直接試聴可能
- **連結音声のダウンロード**: 合成した音声を連結してダウンロード可能
- **統合コマンドラインインターフェース**: `sentiovox`コマンドによる一元的な機能アクセス
- **サブコマンド構造の導入**: UI起動、テキスト/音声処理、JSON処理をサブコマンドとして整理
- **JSONベース処理の強化**: JSONデータの感情分析と音声合成機能の拡充
- **感情分析の精度向上**: 感情検出アルゴリズムの改善とパフォーマンス最適化
- **エラーハンドリングの改善**: 堅牢性の向上と詳細なエラーメッセージ
- **プロジェクト構造の整理**: より保守性の高いモジュール構成

この更新は、コマンドライン操作に慣れていないユーザーや、大量の会話データを効率的に処理したいユーザーに特に有用です。
