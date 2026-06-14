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
    else:
        raise ValueError(f"サポートされていないモデルタイプです: {active_model}")
