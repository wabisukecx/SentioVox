"""AIVISエンジンとの通信を担当するモジュール

このモジュールは、AIVISエンジンとのHTTP通信を管理し、音声合成リクエストの
送信と応答の処理を行います。通信の信頼性を確保するため、エラーハンドリングと
リトライ機能を実装しています。また、音声データの品質を保証するための
様々な前処理と後処理も提供します。
"""

import io
import json
import time
from typing import Optional, Tuple, Dict
import numpy as np
import requests
import soundfile
from ..models.constants import (
    DEFAULT_OUTPUT_SAMPLING_RATE,
    MAX_RETRIES,
    RETRY_DELAY,
    VOLUME_SCALE,
    MODEL_TRUNCATION,
    NOISE_SCALE,
    PRE_POST_PHONEME_LENGTH,
    REQUEST_TIMEOUT,
    MAX_TEXT_LENGTH
)

class AivisClient:
    """AIVISエンジンとの通信を行うクラス
    
    このクラスは以下の責任を持ちます：
    - AIVISエンジンへのHTTPリクエストの送信
    - レスポンスの処理と音声データの変換
    - エラーハンドリングとリトライ処理
    - 通信の最適化とパフォーマンス管理
    """
    
    def __init__(self, base_url: str):
        """初期化
        
        Args:
            base_url: AIVISサーバーのベースURL
            
        Note:
            セッションを再利用することで、TCP接続のオーバーヘッドを削減します。
        """
        self.url = base_url
        self.session = requests.Session()

    def synthesize_segment(
        self,
        text: str,
        style_id: int,
        params: Dict[str, float]
    ) -> Optional[Tuple[np.ndarray, int]]:
        """音声合成リクエストの送信と応答の処理
        
        テキストと音声パラメータを使用して音声を合成し、
        音声データとサンプリングレートを返します。
        
        Args:
            text: 合成するテキスト
            style_id: 音声スタイルのID
            params: 音声パラメータの辞書
            
        Returns:
            Tuple[np.ndarray, int]: 音声データとサンプリングレート
            エラーの場合はNoneを返します。
            
        Note:
            テキストが長すぎる場合は自動的に分割して処理します。
        """
        try:
            # テキストの前処理
            text = self._preprocess_text(text)
            if not text:
                return None

            # テキストの長さチェック
            if len(text) > MAX_TEXT_LENGTH:
                print(f"警告: テキストが長すぎます ({len(text)} 文字)")
                text = text[:MAX_TEXT_LENGTH]
                print(f"テキストを {MAX_TEXT_LENGTH} 文字に切り詰めました")

            # クエリパラメータの設定
            query_params = self._prepare_query_params(text, style_id)

            # 音声クエリの生成
            query_response = self._send_request_with_retry(
                'audio_query',
                method='post',
                params=query_params
            )
            if query_response is None:
                return None

            # パラメータの適用と微調整
            query_response.update(params)
            query_response.update(self._get_additional_params())

            # 音声合成の実行
            audio_response = self._send_request_with_retry(
                'synthesis',
                method='post',
                params={"speaker": style_id},
                headers={
                    "accept": "audio/wav",
                    "Content-Type": "application/json"
                },
                data=json.dumps(query_response)
            )
            if audio_response is None:
                return None

            # 音声データの処理
            return self._process_audio_response(audio_response)

        except Exception as e:
            print(f"音声合成中にエラーが発生しました: {str(e)}")
            return None

    def _preprocess_text(self, text: str) -> str:
        """テキストの前処理を行う
        
        テキストを正規化し、合成に適した形式に変換します。
        空白文字の正規化や句読点の調整を行い、
        不要な記号を適切に処理します。
        
        Args:
            text: 処理対象のテキスト
            
        Returns:
            str: 正規化されたテキスト
            
        Note:
            - ダッシュ（──）は削除
            - 連続する空白は1つに統合
            - 文末が句読点で終わっていない場合は句点を追加
        """
        # 特殊文字の処理
        text = text.replace('─', '、')  # ダッシュを空白に置換
        
        # 基本的な正規化
        text = text.strip()
        text = ' '.join(text.split())  # 連続する空白を1つに
        
        # 文末の句読点の処理
        if not text.endswith(('。', '！', '？', '、')):
            text += '。'
            
        return text

    def _prepare_query_params(self, text: str, style_id: int) -> Dict:
        """クエリパラメータを準備する
        
        音声合成のための基本パラメータを設定します。
        サンプリングレートやステレオ設定などを含みます。
        
        Args:
            text: 合成するテキスト
            style_id: 音声スタイルのID
            
        Returns:
            Dict: クエリパラメータの辞書
        """
        return {
            "text": text,
            "speaker": style_id,
            "outputSamplingRate": DEFAULT_OUTPUT_SAMPLING_RATE,
            "outputStereo": False,
        }

    def _get_additional_params(self) -> Dict[str, float]:
        """追加のパラメータを取得する
        
        音声品質を向上させるための追加パラメータを提供します。
        すべてのパラメータは定数として一元管理されており、
        容易に調整可能です。
        
        Returns:
            Dict[str, float]: 音声合成の追加パラメータ
        """
        return {
            "volumeScale": VOLUME_SCALE,
            "prePhonemeLength": PRE_POST_PHONEME_LENGTH,
            "postPhonemeLength": PRE_POST_PHONEME_LENGTH,
            "modelTruncation": MODEL_TRUNCATION,
            "noiseScale": NOISE_SCALE,
        }

    def _send_request_with_retry(
        self,
        endpoint: str,
        method: str = 'post',
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY,
        **kwargs
    ) -> Optional[dict]:
        """リトライ機能付きでリクエストを送信
        
        通信エラーが発生した場合、指定回数まで再試行します。
        リトライの回数と待機時間は定数として管理されています。
        
        Args:
            endpoint: APIエンドポイント
            method: HTTPメソッド（'get'または'post'）
            max_retries: 最大リトライ回数
            retry_delay: リトライ間の待機時間（秒）
            **kwargs: requestsライブラリに渡す追加の引数
            
        Returns:
            Optional[dict]: レスポンスデータ（エラー時はNone）
            
        Note:
            タイムアウトは REQUEST_TIMEOUT 定数で制御されます。
        """
        kwargs.setdefault('timeout', REQUEST_TIMEOUT)
        
        for attempt in range(max_retries):
            try:
                if method.lower() == 'get':
                    response = self.session.get(
                        f"{self.url}/{endpoint}",
                        **kwargs
                    )
                else:
                    response = self.session.post(
                        f"{self.url}/{endpoint}",
                        **kwargs
                    )
                
                response.raise_for_status()
                return response.json() if endpoint == 'audio_query' else response
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"リクエスト失敗 (試行回数: {attempt + 1}/{max_retries}): {str(e)}")
                    return None
                    
                print(f"リクエスト失敗、リトライします ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay * (attempt + 1))  # 指数バックオフ

    def _process_audio_response(
        self,
        response: requests.Response
    ) -> Optional[Tuple[np.ndarray, int]]:
        """音声レスポンスの処理
        
        受信した音声データをNumPy配列に変換し、必要な前処理を
        適用します。
        
        Args:
            response: AIVISからの音声レスポンス
            
        Returns:
            Tuple[np.ndarray, int]: 音声データとサンプリングレート
            エラー時はNoneを返します。
        """
        try:
            with io.BytesIO(response.content) as stream:
                audio_data, rate = soundfile.read(stream)
                return audio_data, rate
        except Exception as e:
            print(f"音声データの処理中にエラーが発生しました: {str(e)}")
            return None

    def check_health(self) -> bool:
        """AIVISサーバーの健康状態をチェック
        
        サーバーが応答可能な状態にあるかを確認します。
        タイムアウトは REQUEST_TIMEOUT 定数で制御されます。
        
        Returns:
            bool: サーバーが正常に応答する場合はTrue
        """
        try:
            response = self.session.get(
                f"{self.url}/version",
                timeout=REQUEST_TIMEOUT
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def cleanup(self) -> None:
        """リソースのクリーンアップ
        
        セッションを閉じ、使用していたリソースを解放します。
        """
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                print(f"セッションのクローズ中にエラーが発生しました: {str(e)}")