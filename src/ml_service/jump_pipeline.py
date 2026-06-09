"""Полный пайплайн анализа прыжков для сервиса.

Полное видео программы → детекция (rotation + verifier) → N окон прыжков →
B3-классификатор по каждому окну → список прыжков с типом/оборотами/таймкодами.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch

from src.ml_service.b3_predictor import B3JumpClassifier
from src.ml_service.contracts import DetectedJump
from src.ml_service.jump_detector import JumpDetector
from src.ml_service.pose_extraction import CropPoseExtractor


class JumpAnalysisPipeline:
    """Связывает три этапа: детекция → crop-pose → классификация."""

    def __init__(
        self,
        verifier_checkpoint_dir: Path,
        b3_checkpoint: Path,
        b3_meta: Path,
        device: torch.device,
        verifier_threshold: float = 0.6,
    ):
        self.device = device
        self.detector = JumpDetector(verifier_checkpoint_dir, device, threshold=verifier_threshold)
        self.pose_extractor = CropPoseExtractor(device)
        self.classifier = B3JumpClassifier(b3_checkpoint, b3_meta, device)

    def analyze_video(self, video_path: str) -> list[DetectedJump]:
        """Полное видео → список классифицированных прыжков."""
        video_path = str(video_path)

        # этап 1: где прыжки
        detections = self.detector.detect(video_path)
        if not detections:
            return []

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # этап 2-3: crop-pose + классификация по каждому окну
        results: list[DetectedJump] = []
        for start_sec, end_sec, p_jump in detections:
            kpts = self.pose_extractor.extract_window(
                video_path, start_sec, end_sec, src_fps, total,
            )
            prediction = self.classifier.classify(kpts)
            results.append(DetectedJump(
                start_sec=start_sec,
                end_sec=end_sec,
                detection_confidence=p_jump,
                prediction=prediction,
            ))
        return results
