"""SentioVox Streamlit UIのエントリーポイント"""

import os
import sys
from pathlib import Path

def main():
    """Streamlitアプリケーションのエントリーポイント"""
    # プロジェクトルートをパスに追加
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # 感情分析機能を含む新UIを使用
    script_path = str(Path(__file__).parent / "ui" / "streamlit_app_modified.py")
    
    print(f"Streamlit UIを起動: {script_path}")
    
    # 直接モジュールとしてstreamlitを実行
    os.system(f"streamlit run {script_path}")

if __name__ == "__main__":
    main()