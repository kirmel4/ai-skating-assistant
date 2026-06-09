"""Контракты ML-сервиса.

RawPrediction — сырое предсказание классификатора по одному прыжку.
DetectedJump — найденный в видео прыжок: таймкоды + предсказание.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RawPrediction:
    """Предсказание B3-классификатора по одному окну прыжка (id классов)."""

    jump_type_id: int
    rotation_id: int
    underrotation_id: int
    fall_id: int
    rotations_value: float | None = None


@dataclass(frozen=True)
class DetectedJump:
    """Найденный и классифицированный прыжок из полного видео."""

    start_sec: float
    end_sec: float
    detection_confidence: float   # p_jump от verifier'а
    prediction: RawPrediction
