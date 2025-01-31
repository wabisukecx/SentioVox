import os
import time
import psutil
import requests
import subprocess
from typing import Tuple

from ..models.constants import AIVIS_PATH

def check_aivis_server(url: str, timeout: int = 5) -> bool:
    """AIVISサーバーの応答をチェック"""
    try:
        response = requests.get(f"{url}/version", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

def find_aivis_process() -> bool:
    """AivisSpeech-Engineプロセスが実行中かチェック"""
    for proc in psutil.process_iter(['name', 'exe']):
        try:
            if proc.info['exe'] and proc.info['exe'].lower() == AIVIS_PATH.lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def ensure_aivis_server(url: str) -> Tuple[bool, str]:
    """AIVISサーバーの状態を確認
    
    Returns:
        Tuple[bool, str]: (成功したかどうか, メッセージ)
    """
    # サーバーが既に応答可能か確認
    if check_aivis_server(url):
        return True, "AIVISサーバーは既に起動しています"
        
    # インストールの確認
    if not os.path.exists(AIVIS_PATH):
        return False, f"AivisSpeech-Engineが見つかりません。次のパスを確認してください: {AIVIS_PATH}"
        
    # プロセスの確認
    if find_aivis_process():
        return False, "AivisSpeech-Engineは実行中ですが、応答がありません。手動での確認が必要です"
        
    # サーバー起動を試行
    try:
        subprocess.Popen([AIVIS_PATH])
        
        # 起動完了を待機
        for _ in range(30):  # 最大30秒待機
            if check_aivis_server(url):
                return True, "AivisSpeech-Engineを正常に起動しました"
            time.sleep(1)
            
        return False, "AivisSpeech-Engineの起動がタイムアウトしました"
        
    except Exception as e:
        return False, f"AivisSpeech-Engineの起動中にエラーが発生しました: {str(e)}"