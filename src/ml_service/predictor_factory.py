"""Фабрика пайплайна анализа прыжков.

Реальный пайплайн (JumpAnalysisPipeline) импортируется лениво — чтобы
dummy-режим работал без тяжёлых ML-зависимостей и чекпойнтов.
"""

from src.backend.settings import Settings
from src.ml_service.contracts import DetectedJump, RawPrediction


class DummyPipeline:
    """Заглушка для разработки фронта без GPU и чекпойнтов."""

    def analyze_video(self, video_path: str) -> list[DetectedJump]:
        return [
            DetectedJump(12.4, 13.9, 0.98, RawPrediction(5, 1, 0, 0, 2.5)),
            DetectedJump(31.1, 32.8, 0.95, RawPrediction(0, 2, 1, 0, 3.1)),
            DetectedJump(47.6, 49.2, 0.91, RawPrediction(1, 2, 0, 1, 2.9)),
        ]


def create_pipeline(settings: Settings):
    """Возвращает объект с методом analyze_video(video_path) -> list[DetectedJump]."""
    if settings.use_dummy_predictor:
        return DummyPipeline()

    from src.ml_service.jump_pipeline import JumpAnalysisPipeline

    return JumpAnalysisPipeline(
        verifier_checkpoint_dir=settings.verifier_checkpoint_dir,
        b3_checkpoint=settings.b3_checkpoint,
        b3_meta=settings.b3_meta,
        device=settings.torch_device(),
        verifier_threshold=settings.verifier_threshold,
    )
