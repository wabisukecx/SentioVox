"""JSONファイルを処理するコマンドラインツール

このモジュールは、コマンドラインからJSONファイルを処理するための
ユーティリティを提供します。speakerとtextを含むJSONファイルを読み込み、
感情分析を実行し、音声合成を行います。
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.json_emotion_processor import JsonEmotionProcessor
from src.audio.json_synthesis import JsonSynthesisAdapter


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description='SentioVox: JSONファイルの処理と音声合成'
    )
    
    parser.add_argument(
        '--json',
        required=True,
        help='処理するJSONファイル'
    )
    parser.add_argument(
        '--output',
        help='出力JSONファイルのパス',
        default=None
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='JSONファイルに感情分析を実行'
    )
    parser.add_argument(
        '--synthesize',
        action='store_true',
        help='JSONファイルから音声合成を実行'
    )
    parser.add_argument(
        '--mapping',
        help='話者マッピング設定ファイル',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        help='音声ファイルの出力ディレクトリ',
        default='output'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='開始インデックス'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='終了インデックス'
    )
    
    args = parser.parse_args()
    return args


def main() -> None:
    """メインエントリーポイント"""
    args = parse_arguments()

    try:
        # JSONファイルの読み込み
        try:
            with open(args.json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print(f"{len(json_data)}件の会話データを読み込みました")
        except Exception as e:
            print(f"JSONファイルの読み込みに失敗しました: {str(e)}")
            return
        
        # 感情分析の実行
        processed_file = args.json
        if args.analyze:
            print(f"JSONファイルに対して感情分析を実行します...")
            
            processor = JsonEmotionProcessor()
            processed_file = processor.process_json_file(
                args.json,
                args.output
            )
            
            # 処理されたJSONファイルを再読み込み
            with open(processed_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            # サンプル結果の表示
            processor.analyze_sample(json_data)
        
        # 音声合成の実行
        if args.synthesize:
            print(f"JSONデータに対して音声合成を行います...")
            
            # 話者マッピングの読み込み
            character_mapping = {}
            emotion_mapping = {}
            
            if args.mapping:
                try:
                    with open(args.mapping, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                    
                    character_mapping = mapping_data.get("character_mapping", {})
                    emotion_mapping = mapping_data.get("emotion_mapping", {})
                    
                    print(f"話者マッピングを {args.mapping} から読み込みました")
                except Exception as e:
                    print(f"マッピングファイルの読み込みに失敗しました: {str(e)}")
            
            # マッピングがない場合はインタラクティブに作成
            if not character_mapping:
                print("話者マッピングが見つかりません。インタラクティブに作成します...")
                
                # キャラクター一覧の取得
                characters = set(item["speaker"] for item in json_data)
                
                # AIVISサーバーから話者情報を取得
                synthesizer = JsonSynthesisAdapter()
                speakers = synthesizer.get_speakers()
                
                if not speakers:
                    print("AivisSpeech APIが応答しません。デフォルトのマッピングを使用します。")
                    # デフォルトの話者ID
                    default_speaker_id = 888753761
                    for character in characters:
                        character_mapping[character] = default_speaker_id
                else:
                    # 利用可能な話者リストを表示
                    print("\n利用可能な話者:")
                    speaker_id_map = {}
                    index = 1
                    for speaker in speakers:
                        print(f"{index}. {speaker['name']}")
                        for style in speaker['styles']:
                            print(f"   {index}.{style['id'] % 100}: {style['name']} (ID: {style['id']})")
                            speaker_id_map[f"{index}.{style['id'] % 100}"] = style['id']
                        index += 1
                    
                    # 各キャラクターに話者を割り当て
                    for character in characters:
                        valid_choice = False
                        while not valid_choice:
                            try:
                                choice = input(f"\n{character}の話者を選択してください (例: 1.1): ")
                                if choice in speaker_id_map:
                                    character_mapping[character] = speaker_id_map[choice]
                                    valid_choice = True
                                else:
                                    print("無効な選択です。もう一度入力してください。")
                            except KeyboardInterrupt:
                                print("\n中断されました。デフォルトの話者を使用します。")
                                character_mapping[character] = 888753761
                                valid_choice = True
                            except Exception:
                                print("無効な入力です。もう一度入力してください。")
                
                # マッピングを保存
                mapping_filename = os.path.splitext(args.json)[0] + "_mapping.json"
                try:
                    with open(mapping_filename, 'w', encoding='utf-8') as f:
                        json.dump({"character_mapping": character_mapping, "emotion_mapping": {}}, f, ensure_ascii=False, indent=4)
                    print(f"話者マッピングを {mapping_filename} に保存しました")
                except Exception as e:
                    print(f"マッピングの保存に失敗しました: {str(e)}")
            
            # 感情パラメータの設定
            emotion_params = {
                "喜び": {"speedScale": 1.15, "pitchScale": 0.05, "intonationScale": 1.2, "volumeScale": 1.1},
                "悲しみ": {"speedScale": 0.9, "pitchScale": -0.05, "intonationScale": 0.9, "volumeScale": 0.9},
                "怒り": {"speedScale": 1.1, "pitchScale": 0.0, "intonationScale": 1.3, "volumeScale": 1.2},
                "恐れ": {"speedScale": 1.05, "pitchScale": 0.0, "intonationScale": 0.8, "volumeScale": 0.9},
                "期待": {"speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0, "volumeScale": 1.0},
                "驚き": {"speedScale": 1.2, "pitchScale": 0.05, "intonationScale": 1.2, "volumeScale": 1.1},
                "信頼": {"speedScale": 0.95, "pitchScale": 0.0, "intonationScale": 0.9, "volumeScale": 0.95},
                "嫌悪": {"speedScale": 1.05, "pitchScale": -0.02, "intonationScale": 1.1, "volumeScale": 1.0},
                "中立": {"speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0, "volumeScale": 1.0}
            }
            
            # 合成範囲の設定
            start_index = args.start_index
            end_index = args.end_index if args.end_index is not None else len(json_data) - 1
            
            # 有効な範囲に調整
            start_index = max(0, start_index)
            end_index = min(len(json_data) - 1, end_index)
            
            # 進捗表示関数
            def print_progress(progress, current, total, dialogue):
                if dialogue:
                    character = dialogue["speaker"]
                    text = dialogue["text"][:30] + ("..." if len(dialogue["text"]) > 30 else "")
                    emotion = dialogue.get("dominant_emotion", "")
                    emotion_text = f" ({emotion})" if emotion else ""
                    
                    print(f"\r合成中 ({current+1}/{total}): {character}「{text}」{emotion_text}", end="")
            
            # 音声合成の実行
            synthesizer = JsonSynthesisAdapter()
            
            print(f"\n{start_index}から{end_index}までの{end_index-start_index+1}件のセグメントを合成します...")
            
            audio_results = synthesizer.synthesize_dialogue(
                json_data,
                character_mapping,
                emotion_mapping,
                emotion_params,
                start_index=start_index,
                end_index=end_index,
                progress_callback=print_progress
            )
            
            print("\n音声合成が完了しました")
            
            if audio_results:
                # 音声ファイルの保存
                os.makedirs(args.output_dir, exist_ok=True)
                saved_files = synthesizer.save_audio_files(audio_results, args.output_dir)
                print(f"{len(saved_files)}個の音声ファイルを {args.output_dir} に保存しました")
                
                # 連結音声の保存
                combined_audio = synthesizer.connect_audio_files(audio_results)
                if combined_audio:
                    combined_filename = os.path.join(
                        args.output_dir,
                        f"{Path(args.json).stem}_combined.wav"
                    )
                    
                    with open(combined_filename, 'wb') as f:
                        f.write(combined_audio)
                    
                    print(f"連結音声を {combined_filename} に保存しました")
            else:
                print("音声合成結果がありません")

    except KeyboardInterrupt:
        print("\n処理が中断されました")
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()