# SentioVox: 感情分析音声合成システム

このプロジェクトは、テキストや音声の感情分析を行い、その結果に基づいて感情豊かな音声合成を実現するシステムです。

## 機能

- 音声録音
- 音声からのテキスト抽出
- テキストの感情分析
- 感情に基づいた音声合成

## 必要条件

### システム要件
- Python 3.8以上
- CUDA対応GPUを推奨（ない場合はCPUモードで動作）

### 必須ソフトウェア
- AivisSpeech
  - AivisSpeechをインストールすることで、音声合成エンジン（AivisSpeech-Engine）も同時にインストールされます
  - インストール後、音声合成エンジンが`C:\Program Files\AivisSpeech\AivisSpeech-Engine\run.exe`に配置されます
  - 音声合成機能を使用する際は、音声合成エンジンが起動している必要があります

## インストール

1. AivisSpeechのインストール
- AivisSpeechをインストールしてください
- インストール後、音声合成エンジンが正しく配置されていることを確認してください

2. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```

3. SpaCy日本語モデルのインストール:
```bash
python -m spacy download ja_ginza
```

## 使用方法

### コマンドライン引数

- `--file`: 分析対象の音声/テキストファイルを指定
- `--record`: マイクから録音を開始
- `--duration`: 録音時間（秒）を指定
- `--speak`: 分析結果を音声合成で読み上げる

### 例

1. ファイルを分析:
```bash
python -m src.main --file input.txt
```

2. マイクから録音して分析:
```bash
python -m src.main --record --duration 15
```

3. 分析結果を音声合成で読み上げる:
```bash
python -m src.main --file input.txt --speak
```

## プロジェクト構造

```
src/
├── __init__.py
├── audio/
│   ├── __init__.py
│   ├── recorder.py     # 音声録音
│   └── synthesis.py    # 音声合成
├── analysis/
│   ├── __init__.py
│   ├── emotion.py      # 感情分析
│   └── text.py         # テキスト処理
├── models/
│   ├── __init__.py
│   ├── voice.py        # 音声パラメータ
│   └── constants.py    # 定数定義
├── utils/
│   ├── __init__.py
│   ├── warnings.py     # 警告抑制
│   └── aivis_utils.py  # AIVISユーティリティ
└── main.py             # メインスクリプト
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 注意事項

### 音声録音について
- 録音時は周囲の環境音に注意してください
- 録音中はCtrl+Cで中断可能です
- 録音データは一時ファイルとして保存され、処理後に自動的に削除されます

### 感情分析について
- 感情分析は8つの基本感情（喜び、悲しみ、期待、驚き、怒り、恐れ、嫌悪、信頼）を検出します
- 感情スコアは0〜1の範囲で表示されます
- スコアが0.05未満の感情は表示されません

### 音声合成について
- AivisSpeechが必要です
- 音声合成機能を使用する際は、音声合成エンジンが起動している必要があります
- 感情に応じて音声パラメータが自動調整されます
- 複数の感情が検出された場合は、それらが混合されます

## トラブルシューティング

### 音声合成に関する問題
1. **"AivisSpeech-Engineが見つかりません"**
   - AivisSpeechが正しくインストールされているか確認してください
   - インストールが必要な場合は、AivisSpeechをインストールしてください

2. **"音声合成エンジンは実行中ですが、応答がありません"**
   - AivisSpeechのメイン画面を開いて、音声合成エンジンを起動してください
   - ポート10101が他のプロセスで使用されていないか確認してください

3. **"音声合成エンジンの起動がタイムアウトしました"**
   - システムのリソース使用状況を確認してください
   - AivisSpeechを再起動してください

### 一般的な問題
1. **CUDA関連のエラー**
   - GPUが利用できない場合は自動的にCPUモードで動作します
   - CUDA関連の警告は自動的に抑制されます

2. **音声録音の問題**
   - マイクの接続を確認してください
   - PyAudioのインストールに問題がある場合は、OSに応じた追加ライブラリが必要な場合があります

### パフォーマンスの最適化
- バッチサイズは`models/constants.py`で調整可能です
- GPU使用時は自動的に最適化されます
- キャッシュ機構により、同じテキストの重複分析を防いでいます

## 著者

- wabisuke - 初期開発者

## 謝辞

- Whisperチーム - 音声認識モデル
- SpaCyチーム - 自然言語処理ライブラリ
- GiNZAチーム - 日本語自然言語処理ライブラリ
- AIVISチーム - 音声合成エンジン
- koshin2001 - 日本語感情分析モデル (https://huggingface.co/koshin2001/Japanese-to-emotions)
