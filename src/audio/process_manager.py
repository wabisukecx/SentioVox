# src/audio/process_manager.py
"""AIVISプロセスの管理を担当するモジュール

このモジュールは、AIVISエンジンのライフサイクル管理に特化しています。
プロセスの起動、終了、および異常終了時の処理を一元的に管理します。
"""

import os
import time
import atexit
import signal
import subprocess
import psutil
import requests
from typing import Optional, Tuple
from ..models.constants import AIVIS_PATH

class AivisProcessManager:
    """AIVISプロセスを管理するシングルトンクラス
    
    このクラスは以下の責任を持ちます：
    - AIVISプロセスの起動と終了の管理
    - プロセスの異常終了時のクリーンアップ
    - シグナルハンドリングによる安全な終了処理
    
    シングルトンパターンを使用することで、システム全体で
    単一のプロセスインスタンスを保証します。
    """
    _instance = None
    _aivis_process: Optional[subprocess.Popen] = None
    
    def __new__(cls):
        """シングルトンインスタンスの生成"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初期化処理
        
        終了処理とシグナルハンドラを登録します。この処理は
        インスタンスの初回生成時のみ実行されます。
        """
        if not hasattr(self, 'initialized'):
            self.initialized = True
            atexit.register(self.cleanup)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラ：SIGTERMを受信した際の処理"""
        self.cleanup()
        
    def start_aivis(self, exe_path: str) -> bool:
        """AIVISプロセスを開始
        
        Args:
            exe_path: AIVISエンジンの実行ファイルパス
            
        Returns:
            bool: 起動の成否
        """
        try:
            self._aivis_process = subprocess.Popen(exe_path)
            return True
        except Exception as e:
            print(f"AIVISの起動に失敗しました: {str(e)}")
            return False
    
    def cleanup(self, *args):
        """AIVISプロセスの終了処理を行う
        
        プロセスとその子プロセスを含めて完全に終了させ、
        リソースのクリーンアップを確実に行います。
        """
        if self._aivis_process:
            try:
                # プロセスツリー全体を終了
                parent = psutil.Process(self._aivis_process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                parent.terminate()
                
                # プロセスの終了を待機（最大5秒）
                self._aivis_process.wait(timeout=5)
                print("AIVISエンジンを正常に終了しました。")
            except psutil.NoSuchProcess:
                print("AIVISプロセスはすでに終了しています。")
            except Exception as e:
                print(f"AIVISの終了処理中にエラーが発生しました: {str(e)}")
            finally:
                self._aivis_process = None

def ensure_aivis_server(url: str) -> Tuple[bool, str]:
    """AivisSpeech-Engineの状態を確認し、必要に応じて起動する
    
    Args:
        url: AIVISサーバーのベースURL
        
    Returns:
        Tuple[bool, str]: (成功フラグ, メッセージ)
    """
    process_manager = AivisProcessManager()
    
    try:
        response = requests.get(f"{url}/version")
        if response.status_code == 200:
            return True, "AivisSpeech-Engineに接続しました。"
        return False, "AivisSpeech-Engineが応答しません。"
    except requests.exceptions.RequestException:    
        try:
            exe_path = AIVIS_PATH
            if os.path.exists(exe_path):
                if process_manager.start_aivis(exe_path):
                    print("Aivis Engineを起動しています...")
                    time.sleep(10)  # エンジンの起動を待つ
                    
                    response = requests.get(f"{url}/version")
                    if response.status_code == 200:
                        return True, "Aivis Engineが正常に起動しました。"
                    else:
                        process_manager.cleanup()
                        return False, "Aivis Engineの起動に失敗しました。"
            else:
                return False, "Aivis Engineの実行ファイルが見つかりません。"
        except Exception as e:
            process_manager.cleanup()
            return False, f"Aivis Engineの起動中にエラーが発生しました: {str(e)}"