#!/bin/bash
# ==============================================================================
# StyleBertVITS2 APIサーバー 停止スクリプト
#
# このスクリプトは以下の処理を行います：
# 1. バックグラウンドで動作している APIサーバー のプロセスID (PID) を検索
# 2. プロセスが存在する場合、シグナルを送って安全に終了させます
# 3. 終了確認後、デスクトップ通知を送信してメモリ解放をユーザーに報告します
# ==============================================================================

# スクリプトの置かれているディレクトリに移動
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== StyleBertVITS2 サーバー停止処理 ==="

# 1. API サーバーのプロセスID (PID) を取得
# server_fastapi.py で動いているものを特定します
PID=$(pgrep -f "server_fastapi.py" || true)

if [ -z "$PID" ]; then
    echo "API サーバーは起動していません。"
    notify-send -a "StyleBertVITS2" "サーバー停止" "サーバーは既に停止しています。"
    exit 0
fi

echo "動作中のサーバープロセス (PID: $PID) を終了しています..."

# 2. 安全な終了シグナル (SIGTERM) を送信
kill $PID

# 3. プロセスの完全終了を待機 (最大10秒)
SUCCESS=0
for i in {1..10}; do
    sleep 1
    if ! pgrep -f "server_fastapi.py" > /dev/null; then
        echo "=== サーバーは正常に停止しました ==="
        notify-send -a "StyleBertVITS2" "サーバー停止完了" "APIサーバーを停止し、GPUメモリ (VRAM) を解放しました。"
        SUCCESS=1
        break
    fi
    echo -n "."
done

# 4. もし安全に終了しなかった場合は強制終了 (SIGKILL)
if [ $SUCCESS -eq 0 ]; then
    echo "警告: プロセスが正常に終了しなかったため、強制終了します..."
    kill -9 $PID
    notify-send -a "StyleBertVITS2" "サーバー強制停止" "APIサーバーを強制終了しました。"
fi
