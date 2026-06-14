import os
import sys
import argparse
import yaml
import subprocess
import soundfile as sf
import sounddevice as sd
from datetime import datetime

# 音声合成エンジンモジュールをインポート
from tts_engine import create_tts_engine

# ---------------------------------------------------------
# デスクトップ通知を送信する関数 (notify-send)
# ---------------------------------------------------------
def send_notification(title: str, message: str):
    """
    Linuxのシステム通知 (notify-send) を使って、画面に進捗を表示します。
    ショートカットキーでバックグラウンド実行している際に、動作状況を把握しやすくします。
    """
    try:
        # -a はアプリケーション名を設定するオプションです
        subprocess.run(["notify-send", "-a", "音声合成システム", title, message], check=False)
    except Exception:
        # notify-send がインストールされていない、または CUI 環境などの場合は無視します
        pass

# ---------------------------------------------------------
# クリップボードからテキストを取得する関数
# ---------------------------------------------------------
def get_clipboard_text() -> str:
    """
    Linux (X11 / Wayland) 環境において、最も堅牢な方法でクリップボードのテキストを取得します。
    """
    # 1. xclip (X11用) の利用を試みる
    try:
        result = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                               capture_output=True, text=True, check=True)
        text = result.stdout.strip()
        if text:
            return text
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2. wl-paste (Wayland用) の利用を試みる
    try:
        result = subprocess.run(["wl-paste", "--no-newline"],
                               capture_output=True, text=True, check=True)
        text = result.stdout.strip()
        if text:
            return text
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3. Pythonの pyperclip ライブラリの利用を試みる
    try:
        import pyperclip
        text = pyperclip.paste().strip()
        return text
    except Exception:
        pass

    return ""

# ---------------------------------------------------------
# メイン実行処理
# ---------------------------------------------------------
def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Hugging Faceモデルを使用したクリップボードテキスト音声合成システム")
    parser.add_argument("--text", type=str, default=None, help="合成するテキスト（指定がない場合はクリップボードから取得）")
    parser.add_argument("--file", "-f", type=str, default=None, help="合成するテキストファイルのパス（長文の読み込み用）")
    args = parser.parse_args()

    # 1. 設定ファイル (config.yaml) の読み込み
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"設定ファイルが見つかりません: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 音声合成対象のテキストを取得
    # 引数 --text がある場合
    if args.text:
        text = args.text
        print(f"引数からテキストを取得しました: \"{text[:30]}...\"")
    # 引数 --file がある場合
    elif args.file:
        if not os.path.exists(args.file):
            print(f"指定されたファイルが見つかりません: {args.file}")
            send_notification("音声合成エラー", f"ファイルが見つかりません:\n{args.file}")
            sys.exit(1)
        try:
            # 指定されたテキストファイルをUTF-8で読み込みます
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"ファイル \"{args.file}\" からテキストを取得しました。文字数: {len(text)}文字")
        except Exception as e:
            error_msg = f"ファイルの読み込みに失敗しました: {e}"
            print(error_msg, file=sys.stderr)
            send_notification("音声合成エラー", error_msg)
            sys.exit(1)
    # 引数の指定がない場合は、クリップボードから取得
    else:
        text = get_clipboard_text()
        if not text:
            print("クリップボードにテキストがありません。処理を終了します。")
            send_notification("音声合成エラー", "クリップボードが空です。")
            sys.exit(0)
        print(f"クリップボードからテキストを取得しました: \"{text[:30]}...\"")

    # 通知: 音声合成の開始
    send_notification("音声合成", f"音声を生成しています...\n対象テキスト: \"{text[:20]}...\"")

    try:
        # 3. 音声合成エンジンの構築と実行
        # 設定ファイルに基づいて適切なモデルをロードし、メモリ上で音声を生成します
        engine = create_tts_engine(config)
        audio, sample_rate = engine.synthesize(text)

        # 4. 音声の再生 (設定で有効な場合、かつデータが存在する場合)
        if config.get("auto_play", True) and len(audio) > 0:
            print("音声を再生しています...")
            # 一時ファイルを作成せず、メモリ上の NumPy 配列を直接オーディオデバイスに流し込みます
            sd.play(audio, sample_rate)
            sd.wait()  # 再生が完了するまで処理を待機します
            print("再生が完了しました。")

        # 5. 特定フォルダへの音声保存
        output_dir = config.get("output_dir", "./output")
        # 出力先ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)

        # ファイル名を日時から生成 (例: tts_20260614_143000.wav)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)

        # soundfile を使用して NumPy 配列を WAV ファイルとして書き出し
        sf.write(filepath, audio, sample_rate)
        print(f"音声を保存しました: {filepath}")

        # 通知: 成功
        send_notification("音声合成完了", f"音声ファイルを保存しました:\n{filename}")

    except Exception as e:
        error_msg = f"音声合成中にエラーが発生しました: {e}"
        print(error_msg, file=sys.stderr)
        send_notification("音声合成エラー", error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
