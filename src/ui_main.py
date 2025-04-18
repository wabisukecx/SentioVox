import subprocess
import os
import sys
from pathlib import Path

def main():
    """Streamlitアプリケーションのエントリーポイント"""
    # プロジェクトルートをパスに追加
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # 感情分析機能を含む新UIを使用
    script_path = str(Path(__file__).parent / "ui" / "streamlit_app.py")
    
    print(f"Streamlit UIを起動: {script_path}")
    
    # バックグラウンドでStreamlitを実行
    if os.name == 'nt':  # Windowsの場合
        subprocess.Popen(["streamlit", "run", script_path, "--server.fileWatcherType", "none"], 
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Unix系の場合
        subprocess.Popen(["streamlit", "run", script_path, "--server.fileWatcherType", "none"], 
                         start_new_session=True)
    
    print("Streamlitアプリケーションが起動しました。ブラウザで http://localhost:8501 にアクセスしてください。")
    print("（このコンソールは閉じても構いません。）")

if __name__ == "__main__":
    main()