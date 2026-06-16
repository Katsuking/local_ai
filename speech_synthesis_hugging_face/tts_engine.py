import abc
import numpy as np
import torch
import os

# ---------------------------------------------------------
# 音声合成エンジンの共通インターフェース（抽象基底クラス）
# 新しいモデルを追加する場合は、このクラスを継承して実装します。
# ---------------------------------------------------------
class BaseTTSEngine(abc.ABC):
    @abc.abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        テキストから音声を合成し、(音声波形配列, サンプリングレート) のタプルを返します。
        ディスクへの一時ファイルの書き込みは行わず、すべてメモリ上のNumPy配列として処理します。

        Args:
            text (str): 合成する対象の日本語テキスト

        Returns:
            tuple[np.ndarray, int]: (float32型の1次元音声波形配列, サンプリングレート(Hz))
        """
        pass

# ---------------------------------------------------------
# Kokoro-82M 音声合成エンジンの実装
# ---------------------------------------------------------
class KokoroEngine(BaseTTSEngine):
    def __init__(self, repo_id: str, voice: str, speed: float):
        """
        Kokoro-82M エンジンを初期化します。

        Args:
            repo_id (str): Hugging FaceのリポジトリID (例: hexgrad/Kokoro-82M)
            voice (str): 使用する話者 (例: jf_alpha)
            speed (float): 発話速度の倍率 (例: 1.0)
        """
        self.voice = voice
        self.speed = speed
        self.sample_rate = 24000  # Kokoro-82M のデフォルトサンプリングレートは 24kHz です

        # GPU (CUDA) が利用可能か確認し、デバイスを設定します
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Kokoro] 使用デバイス: {self.device.upper()}")
        if self.device == "cuda":
            print(f"[Kokoro] GPU名: {torch.cuda.get_device_name(0)}")

        # Kokoroの KPipeline を初期化 (日本語 'j' を指定)
        # 内部で自動的に必要なモデル（hexgrad/Kokoro-82Mなど）がダウンロードまたはキャッシュからロードされます。
        from kokoro import KPipeline
        print(f"[Kokoro] モデルパイプラインを初期化中 (リポジトリ: {repo_id})...")
        self.pipeline = KPipeline(lang_code='j', device=self.device)
        print("[Kokoro] 初期化が完了しました。")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        テキストから音声を合成し、メモリ上のNumPy配列として結合して返します。
        """
        if not text.strip():
            return np.array([], dtype=np.float32), self.sample_rate

        print(f"[Kokoro] 音声合成を開始します: \"{text[:20]}...\"")

        # パイプラインを実行して音声を生成 (ジェネレータ形式で複数文に分割されて返されます)
        # split_pattern は改行で文を分割するように設定
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r'\n+'
        )

        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                # audio は float32型 の 1次元 NumPy 配列
                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("音声合成で音声データが生成されませんでした。")

        # メモリ上で複数の音声チャンクを1つの配列に結合 (一時ファイルは作成しません)
        full_audio = np.concatenate(audio_chunks)
        print(f"[Kokoro] 音声合成が完了しました。データ長: {len(full_audio)} サンプル")

        return full_audio, self.sample_rate

# ---------------------------------------------------------
# StyleBertVITS2 音声合成エンジンの実装 (ローカルAPIサーバー連携)
# ---------------------------------------------------------
class StyleBertVITS2Engine(BaseTTSEngine):
    def __init__(self, url: str, speaker_id: int, speed: float):
        """
        StyleBertVITS2 エンジンを初期化します。
        
        Args:
            url (str): APIサーバーの音声合成エンドポイント (例: http://localhost:5000/voice)
            speaker_id (int): 話者ID (デフォルトのつくよみちゃんは 0)
            speed (float): 発話速度の倍率
        """
        self.url = url
        self.speaker_id = speaker_id
        self.speed = speed

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        ローカルAPIサーバーにリクエストを送り、メモリ上で音声データをNumPy配列にデコードして返します。
        文字数制限（100文字）を回避するため、長いテキストは適切に分割して合成し、最後に結合します。
        """
        if not text.strip():
            return np.array([], dtype=np.float32), 24000

        import requests
        import io
        import soundfile as sf

        print(f"[StyleBertVITS2] サーバー接続先: {self.url}")
        print(f"[StyleBertVITS2] 音声合成を開始します（全体文字数: {len(text)}文字）")

        # ----------------------------------------------------------------------
        # 新機能: テキストを最大80文字程度の短い文に分割するヘルパー関数
        # StyleBertVITS2のAPI仕様（最大100文字制限）を回避するための処理です。
        # ----------------------------------------------------------------------
        def split_text_for_api(input_text: str, max_len: int = 80) -> list[str]:
            chunks = []
            # まずは改行コードで大まかに分割します
            lines = input_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 既に制限文字数以下の場合はそのまま追加
                if len(line) <= max_len:
                    chunks.append(line)
                    continue
                # 制限を超える場合は句点（。）で分割します
                sub_lines = line.split('。')
                for i, sub_line in enumerate(sub_lines):
                    if i == len(sub_lines) - 1 and not sub_line.strip():
                        continue
                    part = sub_line.strip()
                    if i < len(sub_lines) - 1:
                        part += '。'
                    if not part:
                        continue
                    # 句点で分割しても制限を超える場合は、さらに読点（、）で分割します
                    if len(part) <= max_len:
                        chunks.append(part)
                    else:
                        sub_parts = part.split('、')
                        current = ""
                        for j, p in enumerate(sub_parts):
                            if j < len(sub_parts) - 1:
                                p += '、'
                            if len(current) + len(p) <= max_len:
                                current += p
                            else:
                                if current:
                                    chunks.append(current)
                                current = p
                        if current:
                            chunks.append(current)
            return chunks

        # テキストを短い文に分割
        text_chunks = split_text_for_api(text)
        audio_chunks = []
        sample_rate = 24000  # デフォルト値

        # ----------------------------------------------------------------------
        # 分割したテキストごとにAPIを呼び出し、音声を合成します
        # ----------------------------------------------------------------------
        for idx, chunk in enumerate(text_chunks):
            print(f"[StyleBertVITS2] 部分合成中 ({idx + 1}/{len(text_chunks)}): \"{chunk[:15]}...\"")
            params = {
                "text": chunk,
                "speaker_id": self.speaker_id,
                "length": 1.0 / self.speed,
                "language": "JP"
            }

            try:
                # サーバーに音声合成をリクエスト (タイムアウトは60秒)
                response = requests.get(self.url, params=params, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"StyleBertVITS2 サーバーへの通信に失敗しました。サーバーが起動しているか確認してください。\nエラー詳細: {e}"
                )

            # メモリ上でWAVバイナリをNumPy配列に変換し、一時リストに格納
            chunk_audio, sr = sf.read(io.BytesIO(response.content))
            audio_chunks.append(chunk_audio)
            sample_rate = sr

        if not audio_chunks:
            raise RuntimeError("音声合成で音声データが生成されませんでした。")

        # ----------------------------------------------------------------------
        # すべての部分音声をメモリ上で一つに結合します
        # ----------------------------------------------------------------------
        audio_data = np.concatenate(audio_chunks)
        print(f"[StyleBertVITS2] 合成および結合完了。サンプリングレート: {sample_rate}Hz, 総データ長: {len(audio_data)} サンプル")

        return audio_data, sample_rate

# ---------------------------------------------------------
# 音声合成エンジンのファクトリ関数
# 設定ファイル (config.yaml) に応じて適切なエンジンをインスタンス化します。
# ---------------------------------------------------------
def create_tts_engine(config: dict) -> BaseTTSEngine:
    active_model = config.get("active_model", "kokoro")
    model_settings = config.get("models", {}).get(active_model, {})

    if active_model == "kokoro":
        repo_id = model_settings.get("repo_id", "hexgrad/Kokoro-82M")
        voice = model_settings.get("voice", "jf_alpha")
        speed = model_settings.get("speed", 1.0)
        return KokoroEngine(repo_id=repo_id, voice=voice, speed=speed)
    elif active_model == "style_bert_vits2":
        url = model_settings.get("url", "http://localhost:5000/voice")
        speaker_id = model_settings.get("speaker_id", 0)
        speed = model_settings.get("speed", 1.0)
        return StyleBertVITS2Engine(url=url, speaker_id=speaker_id, speed=speed)
    else:
        raise ValueError(f"サポートされていないモデルタイプです: {active_model}")
