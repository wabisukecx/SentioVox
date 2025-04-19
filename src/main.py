"""SentioVox 統合エントリーポイント

このモジュールは、SentioVoxシステムの主要なエントリーポイントです。
コマンドライン引数に基づいて、Streamlit UIを起動します。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def start_streamlit_ui():
    """Streamlit UIを起動する関数"""
    # UIパスを設定
    script_path = str(Path(__file__).parent / "ui" / "streamlit_app.py")
    
    print(f"SentioVox Streamlit UIを起動: {script_path}")
    
    # Streamlitを実行（ファイルウォッチャーを無効化して効率化）
    cmd = ["streamlit", "run", script_path, "--server.fileWatcherType", "none"]
    
    try:
        if os.name == 'nt':  # Windowsの場合
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Unix系の場合
            subprocess.Popen(cmd, start_new_session=True)
        
        print("Streamlit UIが起動しました。ブラウザで http://localhost:8501 にアクセスしてください。")
        return True
    except Exception as e:
        print(f"Streamlit UIの起動に失敗しました: {e}")
        return False

def main():
    """メインエントリーポイント関数"""
    parser = argparse.ArgumentParser(description='SentioVox: 感情認識音声合成システム')
    parser.add_argument('--legacy', action='store_true', 
                      help='従来のコマンドラインモードで起動（非推奨）')
    
    args = parser.parse_args()
    
    if args.legacy:
        print("注意: 従来のコマンドラインインターフェースは非推奨です。")
    
    # いずれの場合もStreamlit UIを起動
    start_streamlit_ui()

if __name__ == "__main__":
    main()