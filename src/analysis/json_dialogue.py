"""JSONフォーマットの会話データを処理するモジュール

このモジュールは、SentioVoxシステムにJSONフォーマットの会話データ処理機能を
追加します。キャラクターと感情のマッピング、データ検証、会話の処理を行います。
"""

from typing import Dict, List, Tuple, Optional, Set


class JsonDialogueProcessor:
    """JSONフォーマットの会話データを処理するクラス
    
    このクラスは以下の責任を持ちます：
    - JSONファイルの読み込みと検証
    - キャラクターと話者のマッピング管理
    - 感情情報の抽出と処理
    """
    
    def __init__(self):
        """初期化処理"""
        pass
    
    def validate_json_format(self, data: List[Dict]) -> bool:
        """JSONデータのフォーマット検証
        
        Args:
            data: 検証するJSONデータ
            
        Returns:
            bool: フォーマットが有効な場合はTrue
        """
        required_fields = ["speaker", "text"]
        
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not all(field in item for field in required_fields):
                return False
        
        return True
    
    def extract_characters_and_emotions(
        self,
        data: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        """会話データからキャラクターと感情を抽出
        
        Args:
            data: 会話データ
            
        Returns:
            Tuple[List[str], List[str]]: キャラクター名と感情のリスト
        """
        characters: Set[str] = set()
        emotions: Set[str] = set()
        
        for item in data:
            # キャラクター名を抽出
            if "speaker" in item:
                characters.add(item["speaker"])
            
            # 主要感情を抽出
            if "dominant_emotion" in item and item["dominant_emotion"]:
                emotions.add(item["dominant_emotion"])
            
            # 感情スコアから感情を抽出
            if "emotions" in item and isinstance(item["emotions"], dict):
                for emotion in item["emotions"].keys():
                    emotions.add(emotion)
        
        return sorted(list(characters)), sorted(list(emotions))
    
    def get_dialogue_segment(
        self,
        data: List[Dict],
        start_index: int,
        end_index: int
    ) -> List[Dict]:
        """指定範囲の会話セグメントを取得
        
        Args:
            data: 会話データ
            start_index: 開始インデックス
            end_index: 終了インデックス
            
        Returns:
            List[Dict]: 指定範囲の会話データ
        """
        if not data:
            return []
        
        start = max(0, start_index)
        end = min(len(data) - 1, end_index)
        
        return data[start:end+1]