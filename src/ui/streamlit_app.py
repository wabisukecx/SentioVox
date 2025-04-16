import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 絶対インポートに変更
import streamlit as st
import json
import requests
import base64
import pandas as pd
from io import BytesIO
from typing import Optional, Dict, List, Tuple

from src.analysis.json_dialogue import JsonDialogueProcessor
from src.audio.json_synthesis import JsonSynthesisAdapter
from src.models.constants import AIVIS_BASE_URL

# アプリのタイトル設定
st.title("SentioVox 音声合成ツール")

# JSONファイルの選択と読み込み
def load_json_data(file_path=None):
    if file_path is None:
        # ファイルアップロードの処理
        uploaded_file = st.file_uploader("会話データのJSONファイルをアップロード", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                return data, uploaded_file.name
            except Exception as e:
                st.error(f"JSONデータの読み込みに失敗しました: {e}")
                return None, None
        else:
            return None, None
    else:
        # 直接ファイルパスから読み込む
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f), os.path.basename(file_path)
        except Exception as e:
            st.error(f"JSONデータの読み込みに失敗しました: {e}")
            return None, None

# 話者（speaker）情報を取得する関数
@st.cache_data(ttl=600)  # 10分キャッシュ
def get_speakers():
    try:
        response = requests.get(f"{AIVIS_BASE_URL}/speakers")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"スピーカー情報の取得に失敗しました: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"API接続エラー: {e}")
        return []

# JSONデータのフォーマット検証
def validate_json_format(data):
    required_fields = ["speaker", "text"]
    
    if not isinstance(data, list):
        st.error("JSONデータはリスト形式である必要があります")
        return False
    
    for item in data:
        if not all(field in item for field in required_fields):
            st.error(f"必須フィールドが不足しています: {required_fields}")
            return False
    
    return True

# 設定ファイル名を生成
def get_settings_filename(json_filename):
    if not json_filename:
        return "default_settings.json"
    base_name = os.path.splitext(json_filename)[0]
    return f"{base_name}_settings.json"

# キャラクターの話者設定が変更されたときに実行される関数
def character_speaker_changed(character, speaker_id):
    # キャラクターの基本話者IDを更新
    st.session_state.settings["character_mapping"][character] = speaker_id
    
    # そのキャラクターの感情マッピングも全て同じ話者に更新
    if character in st.session_state.settings["emotion_mapping"]:
        for emotion in st.session_state.settings["emotion_mapping"][character]:
            st.session_state.settings["emotion_mapping"][character][emotion] = speaker_id

# タブを作成
tab1, tab2, tab3 = st.tabs(["データ読み込み", "音声設定", "音声合成"])

with tab1:
    st.header("会話データの読み込み")
    
    # JSONファイルのフォーマット説明
    with st.expander("対応JSONフォーマットについて"):
        st.markdown("""
        ## 必要なJSONフォーマット
        
        アップロードするJSONファイルは以下の形式である必要があります：
        ```json
        [
            {
                "speaker": "キャラクター名",
                "text": "セリフ内容",
                "dominant_emotion": "主要感情（オプション）",
                "emotions": {
                    "感情名1": 0.5,
                    "感情名2": 0.3,
                    ...
                }
            },
            ...
        ]
        ```
        
        - `speaker` と `text` は必須です
        - `dominant_emotion` は主要感情を示す文字列で、オプションです
        - `emotions` は感情と強度の連想配列で、オプションです
        """)
    
    # JSONデータの読み込み
    json_data, json_filename = load_json_data()
    
    if json_data and validate_json_format(json_data):
        st.success(f"JSONデータを正常に読み込みました: {len(json_data)}件の会話")
        
        # データを全て表示
        st.subheader("データプレビュー")
        preview_df = pd.DataFrame([
            {
                "Index": i,
                "Character": item["speaker"],
                "Text": item["text"],
                "Emotion": item.get("dominant_emotion", "")
            }
            for i, item in enumerate(json_data)
        ])
        st.dataframe(preview_df, use_container_width=True, height=400)
        
        # キャラクターと感情の概要
        characters = sorted(list(set([item["speaker"] for item in json_data])))
        emotions = sorted(list(set([item.get("dominant_emotion", "") for item in json_data if "dominant_emotion" in item])))
        
        st.subheader("データ概要")
        col1, col2 = st.columns(2)
        with col1:
            st.write("登場人物一覧:")
            st.write(", ".join(characters))
        with col2:
            st.write("感情一覧:")
            st.write(", ".join([e for e in emotions if e]))  # 空文字列を除外
        
        # セッションにデータを保存
        st.session_state.json_data = json_data
        st.session_state.json_filename = json_filename
        st.session_state.characters = characters
        st.session_state.emotions = emotions
    else:
        if json_data:  # データは読み込めたけど検証に失敗した
            st.error("JSONデータの形式が正しくありません。「対応JSONフォーマットについて」を確認してください。")
        # セッションデータをクリア
        if 'json_data' in st.session_state:
            del st.session_state.json_data

with tab2:
    st.header("音声設定")
    
    if 'json_data' not in st.session_state or not st.session_state.json_data:
        st.warning("まず「データ読み込み」タブでJSONデータを読み込んでください。")
        st.stop()
    
    # 話者情報を取得
    speakers = get_speakers()
    if not speakers:
        st.error("話者情報が取得できませんでした。AivisSpeech APIが起動していることを確認してください。")
        st.stop()
    
    # セッション設定の初期化
    if 'settings' not in st.session_state:
        settings_filename = get_settings_filename(st.session_state.json_filename)
        st.session_state.settings = {
            "character_mapping": {},
            "emotion_mapping": {}
        }
        # 設定ファイルが存在するか確認して読み込む
        if os.path.exists(settings_filename):
            try:
                with open(settings_filename, 'r', encoding='utf-8') as f:
                    st.session_state.settings = json.load(f)
                st.info(f"既存の設定を {settings_filename} から読み込みました。")
            except Exception as e:
                st.warning(f"設定ファイルの読み込みに失敗しました: {e}")
    
    # スタイルオプションを作成
    style_options = {}
    style_options_by_id = {}
    for speaker in speakers:
        for style in speaker["styles"]:
            option_text = f"{speaker['name']} - {style['name']} (ID: {style['id']})"
            style_options[option_text] = style['id']
            style_options_by_id[style['id']] = option_text
    
    # キャラクターごとに話者を設定
    st.subheader("キャラクターと話者のマッピング")
    
    for character in st.session_state.characters:
        with st.expander(f"{character}の設定"):
            # デフォルト話者の選択
            default_index = 0
            if character in st.session_state.settings["character_mapping"]:
                speaker_id = st.session_state.settings["character_mapping"][character]
                # 選択されたspeaker_idに対応するoption_textのインデックスを探す
                for i, (option_text, style_id) in enumerate(style_options.items()):
                    if style_id == speaker_id:
                        default_index = i
                        break
            
            selected_default = st.selectbox(
                f"{character}のデフォルト話者",
                options=list(style_options.keys()),
                index=default_index,
                key=f"default_{character}",
                on_change=character_speaker_changed,
                args=(character, style_options[list(style_options.keys())[default_index]])
            )
            
            # 選択された話者のIDを保存し、関連する感情話者も更新
            selected_id = style_options[selected_default]
            character_speaker_changed(character, selected_id)
            
            # 感情ごとの話者設定
            if st.session_state.emotions:  # 感情が抽出されている場合
                use_emotion = st.checkbox(
                    f"{character}の感情ごとに異なる話者/スタイルを設定する", 
                    key=f"use_emotion_{character}"
                )
                
                if use_emotion:
                    # このキャラクターの感情マッピングを初期化
                    if character not in st.session_state.settings["emotion_mapping"]:
                        st.session_state.settings["emotion_mapping"][character] = {}
                    
                    # 各感情に話者を割り当てる
                    for emotion in [e for e in st.session_state.emotions if e]:  # 空文字列を除外
                        # デフォルトインデックスを検索
                        emotion_default_index = 0
                        if (character in st.session_state.settings["emotion_mapping"] and 
                            emotion in st.session_state.settings["emotion_mapping"][character]):
                            emotion_speaker_id = st.session_state.settings["emotion_mapping"][character][emotion]
                            for i, (option_text, style_id) in enumerate(style_options.items()):
                                if style_id == emotion_speaker_id:
                                    emotion_default_index = i
                                    break
                        
                        selected_emotion = st.selectbox(
                            f"{character}の「{emotion}」時の話者/スタイル",
                            options=list(style_options.keys()),
                            index=emotion_default_index,
                            key=f"emotion_{character}_{emotion}"
                        )
                        
                        # 選択された話者のIDを保存
                        selected_emotion_id = style_options[selected_emotion]
                        st.session_state.settings["emotion_mapping"][character][emotion] = selected_emotion_id
    
    # 設定の保存と読み込み
    st.subheader("設定の保存と読み込み")
    
    settings_filename = get_settings_filename(st.session_state.json_filename)
    
    col1, col2 = st.columns(2)
    with col1:
        # 保存
        custom_save_filename = st.text_input("保存するファイル名", settings_filename)
        if st.button("設定を保存"):
            try:
                with open(custom_save_filename, 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.settings, f, ensure_ascii=False, indent=4)
                st.success(f"設定を {custom_save_filename} に保存しました。")
            except Exception as e:
                st.error(f"設定の保存に失敗しました: {e}")
    
    with col2:
        # 読み込み
        custom_load_filename = st.text_input("読み込むファイル名", settings_filename)
        if st.button("設定を読み込む"):
            try:
                with open(custom_load_filename, 'r', encoding='utf-8') as f:
                    st.session_state.settings = json.load(f)
                st.success(f"設定を {custom_load_filename} から読み込みました。")
                st.rerun()  # UIを更新するために再実行
            except FileNotFoundError:
                st.error(f"ファイル {custom_load_filename} が見つかりません。")
            except json.JSONDecodeError:
                st.error(f"ファイル {custom_load_filename} のJSONフォーマットが無効です。")
            except Exception as e:
                st.error(f"設定の読み込みに失敗しました: {e}")

with tab3:
    st.header("音声合成")
    
    if 'json_data' not in st.session_state or not st.session_state.json_data:
        st.warning("まず「データ読み込み」タブでJSONデータを読み込んでください。")
        st.stop()
    
    if not st.session_state.settings.get("character_mapping"):
        st.warning("「音声設定」タブでキャラクターと話者のマッピングを設定してください。")
        st.stop()
    
    # 音声合成パラメータの設定
    st.subheader("合成パラメータ")
    
    # 合成範囲の設定
    col1, col2 = st.columns(2)
    with col1:
        start_index = st.number_input(
            "開始インデックス", 
            min_value=0, 
            max_value=len(st.session_state.json_data)-1, 
            value=0
        )
    
    with col2:
        end_index = st.number_input(
            "終了インデックス", 
            min_value=start_index, 
            max_value=len(st.session_state.json_data)-1, 
            value=min(start_index+5, len(st.session_state.json_data)-1)
        )
    
    # 感情によるパラメータ調整の設定
    st.subheader("感情によるパラメータ調整")
    use_emotion_params = st.checkbox("感情に基づいてパラメータを自動調整", value=True)
    
    if use_emotion_params:
        # 各感情のパラメータ調整
        st.write("感情ごとのパラメータ調整：")
        
        if 'emotion_params' not in st.session_state:
            st.session_state.emotion_params = {
                "喜び": {"speedScale": 1.15, "pitchScale": 0.05, "intonationScale": 1.0, "volumeScale": 1.0},
                "悲しみ": {"speedScale": 0.9, "pitchScale": -0.05, "intonationScale": 0.9, "volumeScale": 0.9},
                "怒り": {"speedScale": 1.1, "pitchScale": 0.0, "intonationScale": 1.3, "volumeScale": 1.2},
                "恐れ": {"speedScale": 1.05, "pitchScale": 0.0, "intonationScale": 0.8, "volumeScale": 0.9},
                "期待": {"speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0, "volumeScale": 1.0},
                "驚き": {"speedScale": 1.2, "pitchScale": 0.05, "intonationScale": 1.2, "volumeScale": 1.1},
                "信頼": {"speedScale": 0.95, "pitchScale": 0.0, "intonationScale": 0.9, "volumeScale": 0.95},
                "嫌悪": {"speedScale": 1.05, "pitchScale": -0.02, "intonationScale": 1.1, "volumeScale": 1.0},
            }
        
        # 感情の選択
        emotions_to_edit = st.session_state.emotions or ["喜び", "悲しみ", "怒り", "恐れ", "期待", "驚き", "信頼", "嫌悪"]
        emotions_to_edit = [e for e in emotions_to_edit if e]  # 空文字列を除外
        
        # 感情タブ（ネストされたエクスパンダーの代わりにタブを使用）
        if emotions_to_edit:
            emotion_tabs = st.tabs(emotions_to_edit)
            
            for i, emotion in enumerate(emotions_to_edit):
                with emotion_tabs[i]:
                    if emotion not in st.session_state.emotion_params:
                        st.session_state.emotion_params[emotion] = {
                            "speedScale": 1.0, "pitchScale": 0.0, 
                            "intonationScale": 1.0, "volumeScale": 1.0
                        }
                    
                    params = st.session_state.emotion_params[emotion]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        params["speedScale"] = st.slider(
                            "話速 (speedScale)", 
                            min_value=0.5, max_value=2.0, value=params["speedScale"], 
                            step=0.05, key=f"speed_{emotion}"
                        )
                        params["pitchScale"] = st.slider(
                            "音高 (pitchScale)", 
                            min_value=-0.15, max_value=0.15, value=params["pitchScale"], 
                            step=0.01, key=f"pitch_{emotion}"
                        )
                    
                    with col2:
                        params["intonationScale"] = st.slider(
                            "抑揚 (intonationScale)", 
                            min_value=0.0, max_value=2.0, value=params["intonationScale"], 
                            step=0.05, key=f"intonation_{emotion}"
                        )
                        params["volumeScale"] = st.slider(
                            "音量 (volumeScale)", 
                            min_value=0.0, max_value=2.0, value=params["volumeScale"], 
                            step=0.05, key=f"volume_{emotion}"
                        )
    
    # 音声合成の実行
    if st.button("選択した範囲を合成"):
        # 進捗表示用
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 音声ファイルを格納するリスト
        audio_files = []
        
        # 選択された範囲の音声を合成
        data_to_process = st.session_state.json_data[start_index:end_index+1]
        total_items = len(data_to_process)
        
        for i, dialogue in enumerate(data_to_process):
            progress = i / total_items
            progress_bar.progress(progress)
            
            character = dialogue["speaker"]
            text = dialogue["text"]
            emotion = dialogue.get("dominant_emotion", "")  # 感情がない場合は空文字列
            
            truncated_text = text[:30] + ("..." if len(text) > 30 else "")
            emotion_text = f" ({emotion})" if emotion else ""
            status_text.text(f"合成中 ({i+1}/{total_items}): {character}「{truncated_text}」{emotion_text}")
            
            # 話者IDを取得
            speaker_id = None
            if character in st.session_state.settings["character_mapping"]:
                # デフォルトのスピーカーID
                speaker_id = st.session_state.settings["character_mapping"][character]
                
                # 感情に対応するスタイルIDがあれば使用
                if (emotion and 
                    character in st.session_state.settings["emotion_mapping"] and 
                    emotion in st.session_state.settings["emotion_mapping"][character]):
                    speaker_id = st.session_state.settings["emotion_mapping"][character][emotion]
            
            if speaker_id is not None:
                try:
                    # 音声クエリを作成
                    response = requests.post(
                        f"{AIVIS_BASE_URL}/audio_query",
                        params={"text": text, "speaker": speaker_id}
                    )
                    
                    if response.status_code == 200:
                        query = response.json()
                        
                        # 感情に基づいてパラメータを調整
                        if use_emotion_params and emotion and emotion in st.session_state.emotion_params:
                            params = st.session_state.emotion_params[emotion]
                            
                            # 各パラメータの調整（上限・下限を考慮）
                            query["speedScale"] = max(0.5, min(2.0, query["speedScale"] * params["speedScale"]))
                            query["pitchScale"] = max(-0.15, min(0.15, query["pitchScale"] + params["pitchScale"]))
                            query["intonationScale"] = max(0.0, min(2.0, query["intonationScale"] * params["intonationScale"]))
                            query["volumeScale"] = max(0.0, min(2.0, query["volumeScale"] * params["volumeScale"]))
                        
                        # 音声合成
                        synth_response = requests.post(
                            f"{AIVIS_BASE_URL}/synthesis",
                            headers={"Content-Type": "application/json"},
                            params={"speaker": speaker_id},
                            json=query
                        )
                        
                        if synth_response.status_code == 200:
                            audio_data = synth_response.content
                            audio_files.append({
                                "index": start_index + i,
                                "character": character,
                                "text": text,
                                "emotion": emotion,
                                "speaker_id": speaker_id,
                                "audio_data": audio_data
                            })
                        else:
                            st.error(f"音声合成に失敗しました: {synth_response.status_code}")
                    else:
                        st.error(f"音声クエリの作成に失敗しました: {response.status_code}")
                except Exception as e:
                    st.error(f"処理中にエラーが発生しました: {e}")
            else:
                st.warning(f"{character}の話者が設定されていません。")
        
        progress_bar.progress(1.0)
        status_text.text("合成完了！")
        
        # 合成した音声を表示
        if audio_files:
            st.subheader("合成された音声")
            
            # テーブル表示形式で音声を表示（表示モード切替なし）
            for audio_item in audio_files:
                emotion_text = f" ({audio_item['emotion']})" if audio_item['emotion'] else ""
                speaker_text = ""
                if audio_item['speaker_id'] in style_options_by_id:
                    speaker_text = f" - {style_options_by_id[audio_item['speaker_id']]}"
                
                st.write(f"#{audio_item['index']} - {audio_item['character']}{emotion_text}{speaker_text}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(audio_item['text'])
                with col2:
                    st.audio(audio_item['audio_data'], format="audio/wav")
                st.divider()
            
            # すべての音声を連結してダウンロード
            try:
                # Base64でエンコードした音声データのリストを作成
                encoded_waves = [base64.b64encode(item["audio_data"]).decode('utf-8') for item in audio_files]
                
                # APIを使って音声を連結
                response = requests.post(
                    f"{AIVIS_BASE_URL}/connect_waves",
                    json=encoded_waves
                )
                
                if response.status_code == 200:
                    output_filename = f"{os.path.splitext(st.session_state.json_filename)[0]}_{start_index}-{end_index}.wav"
                    st.download_button(
                        label="連結された音声をダウンロード",
                        data=response.content,
                        file_name=output_filename,
                        mime="audio/wav"
                    )
                else:
                    st.error(f"音声の連結に失敗しました: {response.status_code}")
            except Exception as e:
                st.error(f"音声の連結中にエラーが発生しました: {e}")
        else:
            st.warning("合成された音声がありません。")