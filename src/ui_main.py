"""SentioVox Streamlit UIのエントリーポイント"""

import os
import sys
from pathlib import Path

def main():
    """Streamlitアプリケーションのエントリーポイント"""
    # プロジェクトルートをパスに追加
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # スクリプトパスを取得
    script_path = str(Path(__file__).parent / "ui" / "streamlit_app.py")
    
    # 直接モジュールとしてstreamlitを実行
    os.system(f"streamlit run {script_path}")

if __name__ == "__main__":
    main()