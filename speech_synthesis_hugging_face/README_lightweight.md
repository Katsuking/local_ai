# StyleBertVITS2 専用構成におけるクライアント環境の軽量化手順

このドキュメントでは、音声合成エンジンとして **StyleBertVITS2（APIサーバー）のみを使用する** 場合に、クライアント側（`speech_synthesis_hugging_face`）の Python 仮想環境（`venv`）や依存関係を大幅に軽量化する手順について説明します。

---

## 軽量化できる理由

`StyleBertVITS2` を使用する場合、テキストの解析（形態素解析や音素変換）および音声波形の生成といった高負荷な AI 処理は、すべて別ディレクトリで動作する **StyleBertVITS2 サーバー側** で行われます。

そのため、クライアント側（本ディレクトリ）は以下の処理だけを担当するシンプルな「中継システム」となります。
1. ショートカットキー経由でクリップボードのテキストを取得する
2. サーバーに HTTP リクエストでテキストを送信する
3. 返ってきた音声データをスピーカーから再生する

結果として、**ローカル推論用の巨大なライブラリ（PyTorchなど）や形態素解析用の辞書データが不要** になります。

---

## 具体的な軽量化手順

### 1. `requirements.txt` の書き換え

`requirements.txt` を以下のように書き換えて、不要なパッケージ（`transformers`, `kokoro`, `misaki` 等）を削除し、通信用の `requests` を追加します。

```text
# 依存ライブラリ一覧 (StyleBertVITS2 クライアント専用)
numpy
soundfile
sounddevice
pyyaml
pyperclip
requests
```

### 2. `setup.sh` の簡略化

`setup.sh` 内の以下の処理は不要になるため、削除またはコメントアウトします。
* `espeak-ng` の存在チェック（17〜28行目付近）
* `PyTorch (torch)` のインストール（45〜49行目付近）
* `UniDic` 辞書のダウンロード（54〜58行目付近）

#### 簡略化後の `setup.sh` の例：
```bash
#!/bin/bash
set -e

echo "=== StyleBertVITS2 専用クライアント セットアップ開始 ==="

# 仮想環境の作成
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip

# 軽量化した requirements.txt からインストール
pip install -r requirements.txt

echo "=== セットアップ完了 ==="
```

### 3. 既存の仮想環境を再構築する場合

すでに重いライブラリをインストールしてしまった既存の `venv` がある場合は、一度ディレクトリごと削除してセットアップし直すのが最も確実で簡単です。

```bash
# 1. 既存の venv を削除
rm -rf venv

# 2. requirements.txt と setup.sh を上記の内容に書き換える

# 3. 再セットアップを実行
./setup.sh
```
