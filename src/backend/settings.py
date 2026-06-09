from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SKATING_",
        extra="ignore",
    )

    # backend
    max_upload_size_mb: int = 2048          # полное видео программы — крупный файл
    temp_dir: Path = Path("tmp/uploads")
    jobs_db: Path = Path("data/jobs.db")    # SQLite-хранилище задач анализа

    # чекпойнты пайплайна
    verifier_checkpoint_dir: Path = Path("checkpoints_verifier")  # ансамбль seed_*/best.pt
    b3_checkpoint: Path = Path("checkpoints_pose_b3_final/seed_45/best.pt")
    b3_meta: Path = Path("data/b3_meta.json")

    # порог verifier'а: выше → меньше ложных срабатываний детектора
    verifier_threshold: float = 0.6

    # устройство вычислений: cuda (GPU-сервер) / cpu
    device: str = "cuda"

    # YOLO через ONNXRuntime ускоряет CPU; на GPU быстрее обычный .pt → false
    use_onnx: bool = False

    # Заглушка для разработки фронта без моделей и чекпойнтов.
    use_dummy_predictor: bool = False

    def torch_device(self) -> torch.device:
        return torch.device(self.device)


settings = Settings()
