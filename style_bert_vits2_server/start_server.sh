#!/bin/bash
# ==============================================================================
# StyleBertVITS2 APIサーバー 起動・セットアップスクリプト (GitHubクローン版)
#
# このスクリプトは以下の処理を行います：
# 1. 公式リポジトリをサブフォルダにクローンします
# 2. クローンしたフォルダ内で Python 仮想環境 (venv) を作成します
# 3. CUDA 12.1 対応 PyTorch および Torchaudio、その他依存パッケージをインストールします
# 4. 公式の初期化スクリプトを実行し、必要アセットとデフォルトモデル (つくよみちゃん) を取得します
# 5. APIサーバーをバックグラウンドで起動します (ポート 5000)
# 6. 起動完了をログ監視し、通知を送信します
# ==============================================================================

# エラーが発生したら即時終了
set -e

# スクリプトの置かれているディレクトリに移動
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== StyleBertVITS2 サーバーセットアップ＆起動開始 ==="

# 1. 公式リポジトリのクローン
if [ ! -d "style-bert-vits2" ]; then
    echo "StyleBertVITS2 の公式リポジトリをクローンしています..."
    git clone https://github.com/litagin02/style-bert-vits2.git
else
    echo "StyleBertVITS2 リポジトリは既にクローンされています。"
fi

# クローンしたディレクトリに移動
cd style-bert-vits2

# 2. 仮想環境 (venv) の作成
if [ ! -d "venv" ]; then
    echo "仮想環境 (venv) を作成しています..."
    python3 -m venv venv
else
    echo "仮想環境 (venv) は既に存在します。"
fi

# 仮想環境をアクティベート
source venv/bin/activate

# 3. pipの最新化と依存パッケージのインストール
echo "pip を最新バージョンにアップグレードしています..."
pip install --upgrade pip

# ※ RTX 3060 で高速に動作させるため、CUDA 12.1 対応 PyTorch / Torchaudio を明示的に指定してインストールします
INSTALL_TORCH=0
if ! python3 -c "import torch" &> /dev/null; then
    INSTALL_TORCH=1
fi
if ! python3 -c "import torchaudio" &> /dev/null; then
    INSTALL_TORCH=1
fi

if [ $INSTALL_TORCH -eq 1 ]; then
    echo "CUDA 12.1 対応の PyTorch および Torchaudio をインストールしています..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "PyTorch および Torchaudio は既にインストールされています。"
fi

# 依存パッケージ群のインストール (カスタムされた requirements.txt を利用します)
echo "依存パッケージをインストールしています..."
pip install -r requirements.txt

# 4. 初期アセットとデフォルトモデル（つくよみちゃん）のダウンロード
# initialize.py は既に存在するファイルをスキップするため、起動時に毎回チェックとして実行しても安全です。
# これにより、必要な BERT モデルや辞書データが不足している場合に確実にダウンロードされます。
echo "初期化スクリプトを実行し、モデルアセットと辞書データをチェック・取得しています..."
python3 initialize.py



# 5. サーバーの多重起動防止
if lsof -i :5000 &> /dev/null; then
    echo "【警告】すでにポート 5000 でプロセスが動作しています。起動をスキップします。"
    notify-send -a "StyleBertVITS2" "サーバー起動警告" "すでにサーバーが動作している可能性があります。"
    exit 0
fi

# 6. APIサーバーのバックグラウンド起動
echo "API サーバーをバックグラウンドで起動しています..."
# server_fastapi.py を実行し、ログを server.log に書き出しながらバックグラウンド化します
nohup python3 server_fastapi.py > server.log 2>&1 &

# 7. サーバーの起動完了を監視 (最大 90 秒待機)
echo "サーバーの起動完了を待機しています..."
SUCCESS=0
for i in {1..90}; do
    sleep 2
    # Uvicorn または本家カスタムログの起動完了を示すログがあるか確認
    if grep -q "Uvicorn running on\|server listen:" server.log; then
        echo "=== サーバーの起動が完了しました ==="
        notify-send -a "StyleBertVITS2" "サーバー起動完了" "APIサーバーが正常に起動しました (ポート 5000)"
        SUCCESS=1
        break
    fi
    # プロセスが途中で死んでいないか確認
    if ! pgrep -f "server_fastapi.py" > /dev/null; then
        echo "エラー: サーバープロセスが異常終了しました。server.log を確認してください。"
        cat server.log
        exit 1
    fi
    echo -n "."
done

if [ $SUCCESS -eq 0 ]; then
    echo "警告: 90秒以内にサーバーの起動完了ログが確認できませんでした。ログを確認してください。"
    tail -n 20 server.log
fi
