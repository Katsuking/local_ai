import os
import sys
import glob
import ctypes.util
import argparse

# ==============================================================================
# 1. 起動前ブートストラップ処理 (CUDA 共有ライブラリ of 自動検出と動的リンク)
# ==============================================================================
# ctranslate2 が必要とする libcublas.so.12 がシステム内のどこにあるかを自動検索します。
# プロセス開始後に os.environ で LD_LIBRARY_PATH を書き換えても動的リンカは認識しないため、
# 検出されたパスを環境変数に追加した上で os.execve() で自分自身を再起動します。


def find_cuda_library_path():
  """
  Locates the directory containing the libcublas shared library.
  Supports various CUDA versions (libcublas.so.11, .12, .13, etc.).
  """
  # If the system can already find libcublas, no need to add custom path
  if ctypes.util.find_library("cublas"):
    return None

  # Standard candidate directories for CUDA installations
  candidates = [
      "/usr/local/lib/ollama/cuda_v12",     # Ollama (CUDA 12)
      "/usr/local/cuda-13/lib64",            # Official CUDA Toolkit 13
      "/usr/local/cuda-12/lib64",            # Official CUDA Toolkit 12
      "/usr/local/cuda-11/lib64",            # Official CUDA Toolkit 11
      "/usr/local/cuda/lib64",               # Symbolic link
      "/usr/lib/x86_64-linux-gnu",           # Standard apt path
      "/usr/lib64",                          # Other distros
      "/opt/cuda/targets/x86_64-linux/lib",  # Arch Linux, etc.
  ]

  for path in candidates:
    if os.path.isdir(path):
      # Look for any libcublas.so files in this directory
      files = glob.glob(os.path.join(path, "libcublas.so*"))
      if files:
        return path

  # Fallback to system-wide recursive search if not found in candidate paths
  search_dirs = ["/usr/local", "/opt", "/usr/lib"]
  for s_dir in search_dirs:
    if os.path.isdir(s_dir):
      matches = glob.glob(os.path.join(
          s_dir, "**/libcublas.so*"), recursive=True)
      if matches:
        return os.path.dirname(matches[0])

  return None


# 動的リンクパスの判定とプロセスの再起動実行
cuda_lib_dir = find_cuda_library_path()
if cuda_lib_dir:
  current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
  if cuda_lib_dir not in current_ld_path.split(":"):
    new_ld_path = f"{cuda_lib_dir}:{current_ld_path}" if current_ld_path else cuda_lib_dir
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    print(f"[音声入力] CUDA ライブラリ検出パスを追加して再起動します: {cuda_lib_dir}")
    try:
      # 環境変数を更新した状態でプロセスを起動し直します
      os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

    except Exception as e:
      print(f"[音声入力] 動的リンク追加後の再起動に失敗しました: {e}")

# ==============================================================================
# 2. 必要なライブラリのインポート (環境変数が反映された状態で安全にロードされます)
# ==============================================================================
import time
import threading
import subprocess
import fcntl  # 二重起動防止のファイルロック用に追加
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from faster_whisper import WhisperModel

# ==========================================
# 2.5 二重起動チェック (排他ロックの取得)
# ==========================================
lock_file = None


def ensure_single_instance():
  """
  二重起動を防止するため、ロックファイルを作成して排他ロックを取得します。
  既に別プロセスがロックを保持している場合は、即座に終了します。
  """
  global lock_file
  lock_path = os.path.expanduser("~/.voice_input.lock")
  try:
    lock_file = open(lock_path, "w")
    # flock でノンブロッキング排他ロックを取得
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except IOError:
    print("[音声入力] ⚠️ 既に別の音声入力プロセスが起動しています。終了します。")
    # 重複起動の場合はデスクトップ通知を送り、ユーザーに知らせます
    try:
      subprocess.run(["notify-send", "-t", "3000", "音声入力", "⚠️ 既に起動しています"])
    except Exception:
      pass
    sys.exit(0)


# 重複起動をチェック
ensure_single_instance()


# ==========================================
# 設定パラメータと引数処理
# ==========================================
# コマンドライン引数を解析し、モデルサイズを指定できるようにします
parser = argparse.ArgumentParser(description="F8キーによるインメモリ音声入力ツール")
parser.add_argument(
    "--model",
    type=str,
    default="medium",
    choices=["tiny", "base", "small", "medium", "large-v3"],
    help="使用するWhisperモデルのサイズを指定します (デフォルト: medium)"
)
# 他のライブラリ（pynput等）の引数と競合しないように parse_known_args を使用します
args, unknown = parser.parse_known_args()

MODEL_SIZE = args.model       # コマンドライン引数からモデル名を取得 (指定がない場合は medium)
DEVICE = "cuda"               # GPU (CUDA) 加速を有効化
COMPUTE_TYPE = "float16"      # 半精度浮動小数点 (float16) を使用し、処理の高速化と VRAM 節約を両立
SAMPLE_RATE = 16000           # Whisperモデルが要求する標準サンプリングレート (16kHz)



# ==========================================
# グローバル状態管理
# ==========================================
is_recording = False          # 現在録音中かどうかのフラグ
audio_buffer = []             # 録音データを一時的にメモリ上に保持するリスト (一時ファイルは作成しない)
stream = None                 # sounddevice の録音ストリームオブジェクト
model = None                  # WhisperModel インスタンス
keyboard_controller = keyboard.Controller()  # キー入力をシミュレートするためのコントローラー


def send_notification(title, message):
  """
  Ubuntu のデスクトップ通知 (notify-send) を送信して、
  バックグラウンド動作時にも音声入力の状態が視覚的にわかるようにします。
  """
  try:
    subprocess.run(["notify-send", "-t", "2000", title, message])
  except Exception as e:
    print(f"通知の送信に失敗しました: {e}")


def audio_callback(indata, frames, time_info, status):
  """
  sounddevice の入力ストリームから呼び出されるコールバック関数。
  取得した音声データをそのままメモリ上のリストに追加します。
  """
  if status:
    print(f"音声入力警告: {status}")
  if is_recording:
    audio_buffer.append(indata.copy())


def start_recording():
  """
  録音処理を開始します。
  """
  global is_recording, audio_buffer, stream
  audio_buffer = []
  is_recording = True

  send_notification("音声入力", "🔴 録音中...")
  print("\n[音声入力] 録音を開始しました。")

  # 16kHz モノラル、float32 形式でマイクから入力を取得
  stream = sd.InputStream(
      samplerate=SAMPLE_RATE,
      channels=1,
      dtype='float32',
      callback=audio_callback
  )
  stream.start()


def stop_and_transcribe():
  """
  録音を停止し、非同期で文字起こしおよび入力エミュレーション処理を開始します。
  """
  global is_recording, stream
  is_recording = False

  print("[音声入力] 録音を停止しました。文字起こしを開始します。")
  send_notification("音声入力", "⏳ 認識中...")

  if stream:
    stream.stop()
    stream.close()
    stream = None

  # 文字起こしと入力処理は UI やリスナーをブロックしないよう、別スレッドで非同期実行します
  threading.Thread(target=process_audio).start()


def process_audio():
  """
  メモリ上の音声データを結合し、Whisper でテキスト化してアクティブウィンドウに貼り付けます。
  """
  global audio_buffer
  if not audio_buffer:
    print("[音声入力] 録音データが空です。")
    send_notification("音声入力", "❌ 音声データがありません")
    return

  # メモリ上の音声バッファを結合して平坦な NumPy 配列にします (ファイル書き出しなし)
  audio_data = np.concatenate(audio_buffer, axis=0).flatten()


  try:
    start_time = time.time()

    # Whisperモデルで文字起こしを実行 (日本語を指定、beam_sizeは標準の5)
    segments, info = model.transcribe(audio_data, beam_size=5, language="ja")
    text = "".join([segment.text for segment in segments])

    elapsed_time = time.time() - start_time
    print(f"[音声入力] 処理時間: {elapsed_time:.2f}秒")
    print(f"[音声入力] 認識結果: {text}")

    if text.strip():
      paste_text(text)
    else:
      send_notification("音声入力", "⚠️ 音声を認識できませんでした")

  except Exception as e:
    print(f"[音声入力] 文字起こしエラー: {e}")
    send_notification("音声入力", f"❌ エラーが発生しました: {e}")


def paste_text(text):
  """
  文字起こし結果をクリップボードにコピーし、
  Ctrl+V を送信してアクティブウィンドウに入力します。
  """
  # 元のクリップボードの値を退避して後で復元したい場合は、以下の行のコメントアウトを解除します
  # original_clipboard = pyperclip.paste()

  # 認識されたテキストをクリップボードに設定（これにより文字起こし結果がクリップボードに残ります）
  pyperclip.copy(text)

  # アプリケーションがフォーカスやクリップボード変更を検知するための短いウェイト
  time.sleep(0.1)

  # Ctrl + V キーの送信をシミュレート
  with keyboard_controller.pressed(keyboard.Key.ctrl):
    keyboard_controller.press('v')
    keyboard_controller.release('v')

  # 元のクリップボードの値を復元したい場合は、以下の行のコメントアウトを解除します
  # time.sleep(0.2)
  # pyperclip.copy(original_clipboard)

  send_notification("音声入力", f"✓ 入力完了: \"{text[:15]}...\"")


def on_press(key):
  """
  キーが押されたときに呼び出されるリスナーコールバック。
  F8キーが押された場合に、録音の開始/終了を切り替えます。
  """
  global is_recording
  # F8 キーの入力を判定
  if key == keyboard.Key.f8:
    if not is_recording:
      start_recording()
    else:
      stop_and_transcribe()


def main():
  """
  メインエントリーポイント。モデルを初期化し、グローバルキーボードリスナーを起動します。
  """
  global model

  # 使用する CUDA ライブラリパスを特定してターミナルに出力
  used_path = None
  ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
  for p in ld_paths:
    if p and (("cuda" in p) or ("nvidia" in p) or ("ollama" in p)):
      used_path = p
      break
  if used_path:
    print(f"[音声入力] 使用する CUDA ライブラリパス: {used_path}")
  else:
    print("[音声入力] システム標準の CUDA ライブラリパスを使用します。")

  print(f"[音声入力] Whisper モデル ({MODEL_SIZE}) をロードしています...")
  send_notification("音声入力", "🚀 モデルをロード中...")

  # モデルのロード (これには初回のダウンロードやGPUロードで数十秒かかる場合があります)
  model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

  print("[音声入力] 準備完了! F8キーを押して音声入力を開始/停止してください。")
  send_notification("音声入力", "🟢 準備完了 (F8キーで開始/停止)")

  # キーボード入力を監視するリスナーを起動し、メインスレッドで待機します
  with keyboard.Listener(on_press=on_press) as listener:
    listener.join()


if __name__ == "__main__":
  main()
