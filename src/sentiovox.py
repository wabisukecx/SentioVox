"""SentioVox 統合コマンドエントリーポイント

このモジュールは、SentioVoxの機能を統合したコマンドラインインターフェイスを提供します。
感情分析、音声合成、Streamlit UIなどの主要機能に簡単にアクセスできます。
"""

import os
import sys
import argparse
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description='SentioVox: 感情認識音声合成システム'
    )
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # UI起動コマンド
    ui_parser = subparsers.add_parser('ui', help='Streamlit UIを起動')
    
    # 従来のコマンド（音声/テキストファイル処理）
    legacy_parser = subparsers.add_parser('process', help='音声/テキストファイルを処理')
    legacy_parser.add_argument(
        '--file',
        help='分析対象の音声/テキストファイル(.mp3, .wav, .m4a, .flac, .txt)',
        default=None
    )
    legacy_parser.add_argument(
        '--record',
        nargs='?',
        const=10,
        type=int,
        metavar='DURATION',
        help='録音時間を指定して録音を開始（デフォルト: 10秒）'
    )
    legacy_parser.add_argument(
        '--speak',
        action='store_true',
        help='音声合成による読み上げを実行'
    )
    legacy_parser.add_argument(
        '--output',
        nargs='?',
        const='output',
        metavar='FILENAME',
        help='出力音声ファイルのベース名',
        default=None
    )
    
    # JSON処理コマンド
    json_parser = subparsers.add_parser('json', help='JSONファイルを処理')
    json_parser.add_argument(
        '--file',
        help='処理するJSONファイル',
        required=True
    )
    json_parser.add_argument(
        '--output',
        help='出力JSONファイルのパス',
        default=None
    )
    json_parser.add_argument(
        '--analyze',
        action='store_true',
        help='JSONファイルに感情分析を実行'
    )
    json_parser.add_argument(
        '--synthesize',
        action='store_true',
        help='JSONファイルから音声合成を実行'
    )
    json_parser.add_argument(
        '--mapping',
        help='話者マッピング設定ファイル',
        default=None
    )
    json_parser.add_argument(
        '--output-dir',
        help='音声ファイルの出力ディレクトリ',
        default='output'
    )
    json_parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='開始インデックス'
    )
    json_parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='終了インデックス'
    )
    
    # 引数の解析
    args = parser.parse_args()
    
    # コマンドの実行
    if args.command == 'ui':
        # UI起動
        from src.ui_main import main as ui_main
        sys.argv = [sys.argv[0]]
        ui_main()
        
    elif args.command == 'process':
        # 従来の音声/テキスト処理
        from src.main import main as main_process
        sys.argv = [sys.argv[0]]
        if args.file:
            sys.argv.extend(['--file', args.file])
        if args.record is not None:
            if args.record == 10:
                sys.argv.append('--record')
            else:
                sys.argv.extend(['--record', str(args.record)])
        if args.speak:
            sys.argv.append('--speak')
        if args.output:
            if args.output == 'output':
                sys.argv.append('--output')
            else:
                sys.argv.extend(['--output', args.output])
        main_process()
        
    elif args.command == 'json':
        # JSON処理
        from src.commands.process_json import main as json_process
        sys.argv = [sys.argv[0]]
        sys.argv.extend(['--json', args.file])
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.analyze:
            sys.argv.append('--analyze')
        if args.synthesize:
            sys.argv.append('--synthesize')
        if args.mapping:
            sys.argv.extend(['--mapping', args.mapping])
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        sys.argv.extend(['--start-index', str(args.start_index)])
        if args.end_index is not None:
            sys.argv.extend(['--end-index', str(args.end_index)])
        json_process()
        
    else:
        # コマンドが指定されていない場合はヘルプを表示
        parser.print_help()

if __name__ == "__main__":
    main()