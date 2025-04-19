"""SentioVox Streamlit UIの拡張版

このモジュールはJSON処理と感情分析および従来のテキスト・音声処理機能を
統合したStreamlit UIを提供します。SentioVox V1.1の機能拡張に対応しています。
"""

import os
import sys
from pathlib import Path

# --- Begin monkey-patch for asyncio.get_running_loop ---
import asyncio
_old_get_running_loop = asyncio.get_running_loop

def safe_get_running_loop():
    try:
        return _old_get_running_loop()
    except RuntimeError:
        class _DummyLoop:
            def is_running(self):
                return False
        return _DummyLoop()

asyncio.get_running_loop = safe_get_running_loop
# --- End asyncio patch ---

# --- Begin monkey-patch for signal.signal to avoid "main thread" errors ---
import signal
_old_signal = signal.signal

def safe_signal(sig, handler):
    try:
        return _old_signal(sig, handler)
    except (ValueError, OSError):
        return None

signal.signal = safe_signal
# --- End signal patch ---

# --- Disable PyTorch JIT and override torch.classes path to avoid watcher errors ---
os.environ['PYTORCH_JIT'] = '0'
import torch
torch.classes.__path__ = []
# -------------------------------------------------------------

# Windows でのイベントループポリシー設定
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 絶対インポートに変更
import streamlit as st
import json
import time
import traceback
import requests
import base64
import pandas as pd
from io import BytesIO
from typing import Optional, Dict, List, Tuple

# ヘルパー関数の定義

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


def has_emotion_data(data):
    return all("emotions" in item and "dominant_emotion" in item for item in data)


def get_settings_filename(json_filename):
    if not json_filename:
        return "default_settings.json"
    base_name = os.path.splitext(json_filename)[0]
    if '_with_emotions' not in base_name:
        base_name += '_with_emotions'
    return f"{base_name}_settings.json"


def get_emotions_filename(json_filename):
    if not json_filename:
        return "output_with_emotions.json"
    base_name = os.path.splitext(json_filename)[0]
    if '_with_emotions' in base_name:
        return f"{base_name}.json"
    else:
        return f"{base_name}_with_emotions.json"


def character_speaker_changed(character, speaker_id):
    st.session_state.settings["character_mapping"][character] = speaker_id
    if character in st.session_state.settings["emotion_mapping"]:
        for emotion in st.session_state.settings["emotion_mapping"][character]:
            st.session_state.settings["emotion_mapping"][character][emotion] = speaker_id


@st.cache_data(ttl=600)
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


def load_json_data(file_path=None, key=None):
    if file_path is None:
        uploaded_file = st.file_uploader("会話データのJSONファイルをアップロード", type=["json"], key=key)
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
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f), os.path.basename(file_path)
        except Exception as e:
            st.error(f"JSONデータの読み込みに失敗しました: {e}")
            return None, None


@st.cache_resource
def get_sentiovox_components():
    try:
        from src.models.constants import AIVIS_BASE_URL
        from src.analysis.json_dialogue import JsonDialogueProcessor
        from src.analysis.json_emotion_processor import JsonEmotionProcessor
        from src.audio.json_synthesis import JsonSynthesisAdapter
        from src.analysis.emotion import EmotionAnalyzer
        from src.analysis.text import TextProcessor
        from src.audio.synthesis import AivisAdapter
        return {
            'AIVIS_BASE_URL': AIVIS_BASE_URL,
            'JsonDialogueProcessor': JsonDialogueProcessor,
            'JsonEmotionProcessor': JsonEmotionProcessor,
            'JsonSynthesisAdapter': JsonSynthesisAdapter,
            'EmotionAnalyzer': EmotionAnalyzer,
            'TextProcessor': TextProcessor,
            'AivisAdapter': AivisAdapter
        }
    except Exception as e:
        st.error(f"SentioVoxコンポーネントのロード中にエラーが発生しました: {str(e)}")
        st.error(traceback.format_exc())
        return None

components = get_sentiovox_components()
if not components:
    st.error("必要なコンポーネントをロードできませんでした。環境設定を確認してください。")
    st.stop()

AIVIS_BASE_URL = components['AIVIS_BASE_URL']
JsonDialogueProcessor = components['JsonDialogueProcessor']
JsonEmotionProcessor = components['JsonEmotionProcessor']
JsonSynthesisAdapter = components['JsonSynthesisAdapter']
EmotionAnalyzer = components['EmotionAnalyzer']
TextProcessor = components['TextProcessor']
AivisAdapter = components['AivisAdapter']


def handle_legacy_processing(file_path=None, should_speak=False, output_path=None):
    if not file_path:
        st.error("ファイルが指定されていません。")
        return
    file_extension = Path(file_path).suffix.lower()
    if file_extension in ['.mp3', '.wav', '.m4a', '.flac']:
        process_audio_file(file_path, should_speak, output_path)
    elif file_extension == '.txt':
        process_text_file(file_path, should_speak, output_path)
    else:
        st.error(f"未対応のファイル形式です: {file_extension}")


def process_audio_file(file_path, should_speak, output_path):
    try:
        text_processor = TextProcessor()
        emotion_analyzer = EmotionAnalyzer()
        st.info("音声認識を実行中...")
        segments = text_processor.segment_audio(str(file_path))
        if not segments:
            st.error("テキストを抽出できませんでした。")
            return
        st.subheader("抽出されたテキスト:")
        for i, segment in enumerate(segments):
            st.write(f"{i+1}: {segment}")
        st.info("感情分析を実行中...")
        emotion_scores = emotion_analyzer.analyze_emotions(segments)
        st.subheader("感情分析結果:")
        emotion_data = []
        for i, (text, scores) in enumerate(zip(segments, emotion_scores)):
            dominant_idx = scores.argmax()
            dominant_emotion = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"][dominant_idx]
            dominant_score = float(scores[dominant_idx])
            emotion_data.append({
                "セグメント": i+1,
                "テキスト": text,
                "主要感情": dominant_emotion,
                "スコア": f"{dominant_score:.3f}"
            })
        emotion_df = pd.DataFrame(emotion_data)
        st.dataframe(emotion_df, use_container_width=True)
        if should_speak or output_path:
            st.info("音声合成を準備中...")
            adapter = AivisAdapter()
            output_file = adapter.speak_continuous(
                segments,
                emotion_scores,
                save_path=output_path,
                play_audio=should_speak
            )
            if output_file:
                st.success(f"音声合成が完了しました: {output_file}")
                with open(output_file, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/m4a")
                st.download_button(
                    label="音声ファイルをダウンロード",
                    data=audio_bytes,
                    file_name=os.path.basename(output_file),
                    mime="audio/m4a"
                )
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {str(e)}")
        st.error(traceback.format_exc())


def process_text_file(file_path, should_speak, output_path):
    try:
        text_processor = TextProcessor()
        emotion_analyzer = EmotionAnalyzer()
        st.info("テキストファイルを読み込み中...")
        segments = text_processor.segment_text(str(file_path))
        if not segments:
            st.error("テキストを分割できませんでした。")
            return
        st.subheader("分割されたテキスト:")
        for i, segment in enumerate(segments):
            st.write(f"{i+1}: {segment}")
        st.info("感情分析を実行中...")
        emotion_scores = emotion_analyzer.analyze_emotions(segments)
        st.subheader("感情分析結果:")
        emotion_data = []
        for i, (text, scores) in enumerate(zip(segments, emotion_scores)):
            dominant_idx = scores.argmax()
            dominant_emotion = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"][dominant_idx]
            dominant_score = float(scores[dominant_idx])
            emotion_data.append({
                "セグメント": i+1,
                "テキスト": text,
                "主要感情": dominant_emotion,
                "スコア": f"{dominant_score:.3f}"
            })
        emotion_df = pd.DataFrame(emotion_data)
        st.dataframe(emotion_df, use_container_width=True)
        if should_speak or output_path:
            st.info("音声合成を準備中...")
            adapter = AivisAdapter()
            output_file = adapter.speak_continuous(
                segments,
                emotion_scores,
                save_path=output_path,
                play_audio=should_speak
            )
            if output_file:
                st.success(f"音声合成が完了しました: {output_file}")
                with open(output_file, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/m4a")
                st.download_button(
                    label="音声ファイルをダウンロード",
                    data=audio_bytes,
                    file_name=os.path.basename(output_file),
                    mime="audio/m4a"
                )
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {str(e)}")
        st.error(traceback.format_exc())

# アプリのタイトル設定
st.title("SentioVox 音声合成ツール")

# サイドバーでモードを選択
app_mode = st.sidebar.selectbox(
    "処理モードを選択",
    ["JSONデータ処理", "単一ファイル処理"]
)

if app_mode == "単一ファイル処理":
    st.header("テキスト/音声ファイル処理")
    
    st.info("""
    このモードでは、音声ファイル(.mp3, .wav, .m4a, .flac)またはテキストファイル(.txt)を
    処理し、感情分析と音声合成を行います。
    
    1. 処理するファイルをアップロード
    2. 音声合成と出力オプションを選択
    3. 処理を実行
    """)
    
    uploaded_file = st.file_uploader("処理するファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "txt"])
    
    if uploaded_file is not None:
        # 一時ファイルとして保存
        temp_file = Path(f"temp_{uploaded_file.name}")
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"ファイルがアップロードされました: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            should_speak = st.checkbox("音声合成を実行", value=True)
        with col2:
            should_output = st.checkbox("音声ファイルを保存", value=True)
        
        output_path = None
        if should_output:
            output_basename = st.text_input("出力ファイル名", value="output")
            output_path = f"{output_basename}.m4a"
        
        if st.button("処理を開始"):
            handle_legacy_processing(temp_file, should_speak, output_path)
            
            # 一時ファイルの削除
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

else:  # JSONデータ処理モード
    # 元のStreamlit UIのタブを作成
    tab1, tab2, tab3, tab4 = st.tabs(["感情分析", "データ読み込み", "音声設定", "音声合成"])
    
    # ここから元のUIと同じ実装
    with tab1:
        st.header("感情分析")
        
        # 感情分析のワークフロー説明
        st.info("""
        このタブでは、会話データに感情分析を追加します。ワークフロー：
        1. 未処理の会話JSONファイル（speakerとtextのみ）をアップロード
        2. 感情分析を実行し、*_with_emotions.jsonファイルを生成（自動保存）
        3. 生成されたファイルを「データ読み込み」タブで読み込みます
        
        すでに *_with_emotions.json と *_with_emotions_settings.json を持っている場合は、
        「データ読み込み」タブから直接ファイルを読み込んで音声合成を開始できます。
        """)
        
        # JSONファイルのフォーマット説明
        with st.expander("会話JSONフォーマットについて"):
            st.markdown("""
            ## 入力JSONフォーマット
            
            アップロードするJSONファイルは以下の形式である必要があります：
            ```json
            [
                {
                    "speaker": "キャラクター名",
                    "text": "セリフ内容"
                },
                ...
            ]
            ```
            
            - `speaker` と `text` は必須です
            - 感情分析後のファイルには `dominant_emotion` と `emotions` フィールドが追加されます
            """)
        
        # JSONデータの読み込み - タブ1用の一意のキーを使用
        uploaded_file = st.file_uploader("会話JSONファイルをアップロード", type=["json"], key="emotion_analysis_uploader")
        
        json_data = None
        json_filename = None
        
        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)
                json_filename = uploaded_file.name
            except Exception as e:
                st.error(f"JSONデータの読み込みに失敗しました: {e}")
        
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
            st.dataframe(preview_df, use_container_width=True, height=300)
            
            # 感情情報が含まれているかチェック
            has_emotions = has_emotion_data(json_data)
            
            if has_emotions:
                st.success("このJSONデータには既に感情情報が含まれています。別タブの「データ読み込み」から読み込んでください。")
                
                # 感情分布を表示
                st.subheader("感情分布")
                emotion_counts = {}
                for item in json_data:
                    emotion = item.get("dominant_emotion", "")
                    if emotion:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                emotion_df = pd.DataFrame({
                    "感情": list(emotion_counts.keys()),
                    "回数": list(emotion_counts.values())
                })
                st.bar_chart(emotion_df, x="感情", y="回数")
                
            else:
                st.warning("このJSONデータには感情情報が含まれていません。感情分析を実行します。")
                
                run_analysis = st.button("感情分析を実行", key="tab1_run_emotion_analysis")
                
                if run_analysis:
                    try:
                        # プログレスバーとステータスメッセージの表示
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("感情分析を開始しています...")
                        
                        data_to_analyze = json_data.copy()
                        
                        def update_progress(current, total):
                            progress = float(current) / float(total)
                            progress_bar.progress(progress)
                            status_text.text(f"感情分析中... ({current}/{total} 完了)")
                        
                        emotion_processor = JsonEmotionProcessor()
                        
                        total_items = len(data_to_analyze)
                        for i in range(0, total_items, max(1, total_items // 10)):
                            update_progress(i, total_items)
                            time.sleep(0.1)
                        
                        analyzed_data = emotion_processor.process_json_data(data_to_analyze)
                        
                        progress_bar.progress(1.0)
                        status_text.text("感情分析が完了しました！")
                        
                        # セッションステートに保存
                        st.session_state.json_data = analyzed_data
                        
                        st.success(f"{len(analyzed_data)}件のデータの感情分析が完了しました。")
                        
                        # 感情分析結果を自動で保存する
                        default_output_file = get_emotions_filename(json_filename)
                        with open(default_output_file, 'w', encoding='utf-8') as f:
                            json.dump(analyzed_data, f, ensure_ascii=False, indent=2)
                        st.success(f"感情分析結果を自動で {default_output_file} に保存しました。")
                        
                        # 感情分布を表示
                        st.subheader("感情分布")
                        emotion_counts = {}
                        for item in analyzed_data:
                            emotion = item.get("dominant_emotion", "")
                            if emotion:
                                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        emotion_df = pd.DataFrame({
                            "感情": list(emotion_counts.keys()),
                            "回数": list(emotion_counts.values())
                        })
                        st.bar_chart(emotion_df, x="感情", y="回数")
                        
                    except Exception as e:
                        st.error(f"感情分析中にエラーが発生しました: {str(e)}")
                        st.error("詳細エラー情報: " + traceback.format_exc())
        
        elif json_data:
            st.error("JSONデータの形式が正しくありません。会話JSONフォーマットをご確認ください。")

    # タブ2: データ読み込み
    with tab2:
        st.header("データ読み込み")
        st.info("""
        このタブでは、感情分析済みのJSONファイル（*_with_emotions.json）を読み込みます。
        感情分析がまだのファイルは、まず「感情分析」タブで処理してください。
        """)
        
        json_data, json_filename = load_json_data(key="data_load_uploader")
        
        if json_data and validate_json_format(json_data):
            has_emotions = has_emotion_data(json_data)
            if not has_emotions:
                st.error("このファイルには感情分析情報が含まれていません。まず「感情分析」タブで実行してください。")
                st.stop()
            
            st.success(f"感情分析済みJSONデータを正常に読み込みました: {len(json_data)}件の会話")
            st.subheader("データプレビュー")
            preview_df = pd.DataFrame([
                {"Index": i, "Character": item["speaker"], "Text": item["text"], "Emotion": item.get("dominant_emotion", "")}
                for i, item in enumerate(json_data)
            ])
            st.dataframe(preview_df, use_container_width=True, height=400)
            
            characters = sorted(list(set([item["speaker"] for item in json_data])))
            emotions = sorted(list(set([item.get("dominant_emotion", "") for item in json_data if item.get("dominant_emotion", "")])))
            
            st.subheader("データ概要")
            col1, col2 = st.columns(2)
            with col1:
                st.write("登場人物一覧:")
                st.write(", ".join(characters))
            with col2:
                st.write("感情一覧:")
                st.write(", ".join(emotions))
            
            st.session_state.json_data = json_data
            st.session_state.json_filename = json_filename
            st.session_state.characters = characters
            st.session_state.emotions = emotions
            
            st.subheader("感情分布")
            emotion_counts = {}
            for item in json_data:
                emotion = item.get("dominant_emotion", "")
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            emotion_df = pd.DataFrame({
                "感情": list(emotion_counts.keys()),
                "回数": list(emotion_counts.values())
            })
            st.bar_chart(emotion_df, x="感情", y="回数")
            
            st.info("データ読み込み完了。次に「音声設定」タブで話者設定をしてください。")
                
        else:
            if json_data:
                st.error("JSONデータの形式が正しくありません。")

    # タブ3: 音声設定
    with tab3:
        st.header("音声設定")
        
        if 'json_data' not in st.session_state or not st.session_state.json_data:
            st.warning("まず「データ読み込み」タブで感情分析済みJSONデータを読み込んでください。")
            st.stop()
        
        if 'json_filename' not in st.session_state or not st.session_state.json_filename:
            st.warning("JSONファイルが読み込まれていません。まず「データ読み込み」タブでデータを読み込んでください。")
            st.stop()
        
        has_emotions = has_emotion_data(st.session_state.json_data)
        if not has_emotions:
            st.warning("データに感情分析情報が含まれていません。感情分析を実行してください。")
            st.stop()
        
        speakers = get_speakers()
        if not speakers:
            st.error("話者情報が取得できませんでした。AivisSpeech APIをご確認ください。")
            st.stop()
        
        settings_filename = get_settings_filename(st.session_state.json_filename)
        if 'settings' not in st.session_state:
            st.session_state.settings = {"character_mapping": {}, "emotion_mapping": {}}
            if os.path.exists(settings_filename):
                try:
                    with open(settings_filename, 'r', encoding='utf-8') as f:
                        st.session_state.settings = json.load(f)
                    st.info(f"既存の設定を {settings_filename} から読み込みました。")
                except Exception as e:
                    st.warning(f"設定ファイルの読み込みに失敗しました: {e}")
        
        style_options = {}
        style_options_by_id = {}
        for speaker in speakers:
            for style in speaker["styles"]:
                option_text = f"{speaker['name']} - {style['name']} (ID: {style['id']})"
                style_options[option_text] = style['id']
                style_options_by_id[style['id']] = option_text
        
        st.subheader("キャラクターと話者のマッピング")
        for character in st.session_state.characters:
            with st.expander(f"{character}の設定"):
                default_index = 0
                if character in st.session_state.settings["character_mapping"]:
                    speaker_id = st.session_state.settings["character_mapping"][character]
                    for i, (option_text, style_id) in enumerate(style_options.items()):
                        if style_id == speaker_id:
                            default_index = i
                            break
                
                selected_default = st.selectbox(
                    f"{character}のデフォルト話者",
                    options=list(style_options.keys()),
                    index=default_index,
                    key=f"tab3_default_{character}",
                    on_change=character_speaker_changed,
                    args=(character, style_options[list(style_options.keys())[default_index]])
                )
                
                selected_id = style_options[selected_default]
                character_speaker_changed(character, selected_id)
                
                if st.session_state.emotions:
                    use_emotion = st.checkbox(
                        f"{character}の感情ごとに異なる話者/スタイルを設定する", 
                        key=f"tab3_use_emotion_{character}"
                    )
                    
                    if use_emotion:
                        if character not in st.session_state.settings["emotion_mapping"]:
                            st.session_state.settings["emotion_mapping"][character] = {}
                        
                        for emotion in [e for e in st.session_state.emotions if e]:
                            emotion_default_index = 0
                            if character in st.session_state.settings["emotion_mapping"] and emotion in st.session_state.settings["emotion_mapping"][character]:
                                emotion_speaker_id = st.session_state.settings["emotion_mapping"][character][emotion]
                                for i, (option_text, style_id) in enumerate(style_options.items()):
                                    if style_id == emotion_speaker_id:
                                        emotion_default_index = i
                                        break
                            
                            selected_emotion = st.selectbox(
                                f"{character}の「{emotion}」時の話者/スタイル",
                                options=list(style_options.keys()),
                                index=emotion_default_index,
                                key=f"tab3_emotion_{character}_{emotion}"
                            )
                            
                            selected_emotion_id = style_options[selected_emotion]
                            st.session_state.settings["emotion_mapping"][character][emotion] = selected_emotion_id
        
        st.subheader("設定の保存と読み込み")
        settings_filename = get_settings_filename(st.session_state.json_filename)
        col1, col2 = st.columns(2)
        with col1:
            custom_save_filename = st.text_input("保存するファイル名", settings_filename, key="tab3_settings_save_filename")
            if st.button("設定を保存", key="tab3_save_settings"):
                try:
                    with open(custom_save_filename, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.settings, f, ensure_ascii=False, indent=4)
                    st.success(f"設定を {custom_save_filename} に保存しました。")
                    st.info("設定が保存されました。「音声合成」タブで音声を生成してください。")
                except Exception as e:
                    st.error(f"設定の保存に失敗しました: {e}")
        
        with col2:
            custom_load_filename = st.text_input("読み込むファイル名", settings_filename, key="tab3_settings_load_filename")
            if st.button("設定を読み込む", key="tab3_load_settings"):
                try:
                    with open(custom_load_filename, 'r', encoding='utf-8') as f:
                        st.session_state.settings = json.load(f)
                    st.success(f"設定を {custom_load_filename} から読み込みました。")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"ファイル {custom_load_filename} が見つかりません。")
                except json.JSONDecodeError:
                    st.error(f"ファイル {custom_load_filename} のJSONフォーマットが無効です。")
                except Exception as e:
                    st.error(f"設定の読み込みに失敗しました: {e}")

    # タブ4: 音声合成
    with tab4:
        st.header("音声合成")
        
        if 'json_data' not in st.session_state or not st.session_state.json_data:
            st.warning("まず「データ読み込み」タブで感情分析済みJSONデータを読み込んでください。")
            st.stop()
        
        has_emotions = has_emotion_data(st.session_state.json_data)
        if not has_emotions:
            st.warning("データに感情分析情報が含まれていません。感情分析を実行してください。")
            st.stop()
        
        if not st.session_state.get("settings", {}).get("character_mapping"):
            st.warning("「音声設定」タブでキャラクターと話者のマッピングを設定してください。")
            st.stop()
        
        st.subheader("合成パラメータ")
        col1, col2 = st.columns(2)
        with col1:
            start_index = st.number_input("開始インデックス", min_value=0, max_value=len(st.session_state.json_data)-1, value=0, key="tab4_start_index")
        with col2:
            end_index = st.number_input("終了インデックス", min_value=start_index, max_value=len(st.session_state.json_data)-1, value=min(start_index+5, len(st.session_state.json_data)-1), key="tab4_end_index")
        
        st.subheader("感情によるパラメータ調整")
        use_emotion_params = st.checkbox("感情に基づいてパラメータを自動調整", value=True, key="tab4_use_emotion_params")
        
        if use_emotion_params:
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
                    "中立": {"speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0, "volumeScale": 1.0},
                }
            
            emotions_to_edit = st.session_state.emotions or ["喜び", "悲しみ", "怒り", "恐れ", "期待", "驚き", "信頼", "嫌悪", "中立"]
            emotions_to_edit = [e for e in emotions_to_edit if e]
            
            if emotions_to_edit:
                emotion_tabs = st.tabs(emotions_to_edit)
                for i, emotion in enumerate(emotions_to_edit):
                    with emotion_tabs[i]:
                        if emotion not in st.session_state.emotion_params:
                            st.session_state.emotion_params[emotion] = {"speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0, "volumeScale": 1.0}
                        
                        params = st.session_state.emotion_params[emotion]
                        col1, col2 = st.columns(2)
                        with col1:
                            params["speedScale"] = st.slider("話速 (speedScale)", min_value=0.5, max_value=2.0, value=params["speedScale"], step=0.05, key=f"tab4_speed_{emotion}")
                            params["pitchScale"] = st.slider("音高 (pitchScale)", min_value=-0.15, max_value=0.15, value=params["pitchScale"], step=0.01, key=f"tab4_pitch_{emotion}")
                        with col2:
                            params["intonationScale"] = st.slider("抑揚 (intonationScale)", min_value=0.0, max_value=2.0, value=params["intonationScale"], step=0.05, key=f"tab4_intonation_{emotion}")
                            params["volumeScale"] = st.slider("音量 (volumeScale)", min_value=0.0, max_value=2.0, value=params["volumeScale"], step=0.05, key=f"tab4_volume_{emotion}")
        
        if st.button("選択した範囲を合成", key="tab4_synthesize_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            audio_files = []
            data_to_process = st.session_state.json_data[start_index:end_index+1]
            synthesizer = JsonSynthesisAdapter()
            
            def update_progress(progress, current, total, dialogue):
                progress_bar.progress(progress)
                if dialogue:
                    character = dialogue["speaker"]
                    text = dialogue["text"]
                    emotion = dialogue.get("dominant_emotion", "")
                    truncated_text = text[:30] + ("..." if len(text) > 30 else "")
                    emotion_text = f" ({emotion})" if emotion else ""
                    status_text.text(f"合成中 ({current+1}/{total}): {character}「{truncated_text}」{emotion_text}")
            
            audio_results = synthesizer.synthesize_dialogue(
                data_to_process,
                st.session_state.settings["character_mapping"],
                st.session_state.settings["emotion_mapping"],
                st.session_state.emotion_params if use_emotion_params else None,
                progress_callback=update_progress
            )
            
            progress_bar.progress(1.0)
            status_text.text("合成完了！")
            
            if audio_results:
                st.subheader("合成された音声")
                for audio_item in audio_results:
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
                
                combined_audio = synthesizer.connect_audio_files(audio_results)
                if combined_audio:
                    output_filename = f"{os.path.splitext(st.session_state.json_filename)[0]}_{start_index}-{end_index}.wav"
                    st.download_button(label="連結された音声をダウンロード", data=combined_audio, file_name=output_filename, mime="audio/wav", key="tab4_download_button")
            else:
                st.warning("合成された音声がありません。")

def main():
    """メインエントリーポイント関数"""
    # 引数からファイルパスを取得
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            should_speak = "--speak" in sys.argv
            
            if "--output" in sys.argv:
                output_idx = sys.argv.index("--output")
                if output_idx + 1 < len(sys.argv) and not sys.argv[output_idx + 1].startswith("--"):
                    output_path = sys.argv[output_idx + 1]
                else:
                    output_path = "output.m4a"
            else:
                output_path = None
                
            handle_legacy_processing(file_path, should_speak, output_path)

if __name__ == "__main__":
    main()