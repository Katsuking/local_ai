#!/bin/bash

# 音声合成システムのセットアップスクリプト
#
# このスクリプトは以下の処理を自動で行います：
# 1. システムに必要な espeak-ng コマンドの存在チェック
# 2. Python 仮想環境 (venv) の作成
# 3. 仮想環境内での pip のアップデート
# 4. GPU (RTX 3060) 向けの CUDA 12.1 対応 PyTorch のインストール
# 5. requirements.txt に記載された依存パッケージのインストール

# エラーが発生した時点でスクリプトの実行を中断する設定
set -e

echo "=== 音声合成システム セットアップ開始 ==="

# 1. espeak-ng がインストールされているか確認
# ※ Kokoro-82M などの多言語モデルがテキストを音素解析する際に必要です。
if ! command -v espeak-ng &> /dev/null; then
    echo "【警告】espeak-ng がインストールされていません。"
    echo "音声合成（音素変換）を行うために espeak-ng がシステムに必要です。"
    echo "セットアップを続行するには、まず以下のコマンドをターミナルで実行してインストールしてください："
    echo ""
    echo "    sudo apt update && sudo apt install -y espeak-ng"
    echo ""
    # インストールされていない場合は、エラー終了してユーザーにインストールを促す
    exit 1
fi

# 2. プロジェクトディレクトリ配下に仮想環境 (venv) を作成
if [ ! -d "venv" ]; then
    echo "Python 仮想環境 (venv) を作成しています..."
    python3 -m venv venv
else
    echo "Python 仮想環境 (venv) は既に存在するため、作成をスキップします。"
fi

# 3. 仮想環境をアクティベート
source venv/bin/activate

# 4. pipを最新化
echo "pip を最新バージョンにアップグレードしています..."
pip install --upgrade pip

# 5. CUDA対応の PyTorch をインストール
# ※ RTX 3060 の性能を活用するため、公式の CUDA 12.1 対応バイナリを明示的に指定してインストールします。
echo "RTX 3060 GPU 向けに CUDA 12.1 対応の PyTorch をインストールしています..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 6. その他の依存関係（sounddevice, pyperclip, kokoro等）のインストール
echo "requirements.txt から必要な Python パッケージをインストールしています..."
pip install -r requirements.txt

# 7. 日本語形態素解析用の UniDic 辞書データのダウンロード
# ※ 日本語の漢字やひらがなの読み分けを行うために辞書データが必要です。
echo "日本語解析用の UniDic 辞書データをダウンロードしています..."
python3 -m unidic download

echo "=== セットアップが完了しました ==="
echo "これで Python 仮想環境の準備は完了です。"

