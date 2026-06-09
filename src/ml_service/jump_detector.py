"""Детекция прыжков в полном видео — этап 1 пайплайна сервиса.

rotation-эвристика → кандидаты → verifier → слияние пересекающихся окон.

YOLO-pose прогоняется по видео ОДИН раз: rotation-проход отдаёт и сигнал
вращения, и покадровые keypoints. Окно каждого кандидата для verifier'а
ресемплится из этого кэша (без повторного прогона YOLO) — оптимизация #1.

Verifier — ансамбль из 3 сидов (усреднение вероятностей): одиночный чекпойнт
уверенно ошибается на отдельных не-прыжках, ансамбль их сглаживает.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.detect_jumps import (
    POSE_IMGSZ,
    POSE_MODEL,
    VERIFY_MERGE_GAP,
    compute_angular_speed,
    detect_airborne_candidates,
    detect_candidates,
    extract_rotation_signal,
    merge_candidate_lists,
    merge_verified,
)
from src.ml_service.yolo_loader import load_yolo
from scripts.train_verifier import (
    DEFAULT_THRESHOLD as VERIFIER_THRESHOLD,
    FEATURE_DIM as VERIFIER_FEATURE_DIM,
    NUM_FRAMES as VERIFIER_NUM_FRAMES,
    SEEDS as VERIFIER_SEEDS,
    SMOOTH_SIGMA as VERIFIER_SMOOTH,
    TEMPORAL_HIDDEN_DIM as VERIFIER_HIDDEN,
    WINDOW_SEC as VERIFIER_WINDOW_SEC,
    VerifierModel,
    extract_features_f4 as extract_verifier_features,
)


def _window_kpts_from_cache(times, keypoints, center_sec: float) -> torch.Tensor:
    """Окно verifier'а (VERIFIER_NUM_FRAMES кадров) из кэша покадровых keypoints
    rotation-прохода. Индексный ресемпл — как при обучении verifier'а."""
    half = VERIFIER_WINDOW_SEC / 2.0
    mask = (times >= center_sec - half) & (times <= center_sec + half)
    win = keypoints[mask]
    if len(win) == 0:
        return torch.zeros(VERIFIER_NUM_FRAMES, 17, 3, dtype=torch.float32)
    idx = np.linspace(0, len(win) - 1, VERIFIER_NUM_FRAMES).round().astype(int)
    return torch.from_numpy(win[idx]).float()


class JumpDetector:
    """Находит окна прыжков: rotation-эвристика + verifier-ансамбль."""

    def __init__(
        self,
        verifier_checkpoint_dir: Path,
        device: torch.device,
        threshold: float = VERIFIER_THRESHOLD,
    ):
        self.device = device
        self.threshold = threshold

        ckpt_dir = Path(verifier_checkpoint_dir)
        ckpt_paths = [ckpt_dir / f"seed_{s}" / "best.pt" for s in VERIFIER_SEEDS]
        ckpt_paths = [p for p in ckpt_paths if p.is_file()]
        if not ckpt_paths:
            raise FileNotFoundError(f"Verifier чекпойнты не найдены в {ckpt_dir}")

        self.verifiers = []
        for p in ckpt_paths:
            model = VerifierModel(VERIFIER_FEATURE_DIM, VERIFIER_HIDDEN).to(device)
            ckpt = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.verifiers.append(model)
        print(f"[detector] verifier-ансамбль: {len(self.verifiers)} чекпойнтов")

        self.pose = load_yolo(POSE_MODEL, device, imgsz=POSE_IMGSZ)

    @torch.no_grad()
    def detect(self, video_path: str) -> list[tuple[float, float, float]]:
        """Полное видео → [(start_sec, end_sec, p_jump)] финальных детекций."""
        video_path = str(video_path)

        # 1. rotation-проход: ОДИН прогон YOLO даёт и сигнал, и покадровые keypoints
        times, angles, keypoints = extract_rotation_signal(
            video_path, 0.0, 0.0, self.device, model=self.pose,
        )
        if len(times) == 0:
            return []
        ang_speed = compute_angular_speed(times, angles)
        rot_candidates = detect_candidates(times, ang_speed)
        air_candidates = detect_airborne_candidates(times, keypoints)
        candidates = merge_candidate_lists(rot_candidates, air_candidates)
        print(f"[detector] кандидаты: rotation={len(rot_candidates)}, "
              f"airborne={len(air_candidates)}, объединено={len(candidates)}")
        if not candidates:
            return []

        # 2. verifier — окно кандидата ресемплится из кэша keypoints (без YOLO)
        verified: list[tuple[float, float, float]] = []
        for cs, ce in candidates:
            kpts = _window_kpts_from_cache(times, keypoints, (cs + ce) / 2.0)
            feats = extract_verifier_features(kpts, smooth_sigma=VERIFIER_SMOOTH).unsqueeze(0).to(self.device)
            # ансамбль — усреднение вероятностей по сидам
            probs = [torch.softmax(m(feats), dim=-1)[0, 1].item() for m in self.verifiers]
            verified.append((cs, ce, sum(probs) / len(probs)))

        # 3. фильтр по порогу + слияние пересекающихся окон одного прыжка
        verified_sorted = sorted(verified, key=lambda x: x[2], reverse=True)
        print(f"[detector] verifier по {len(verified)} кандидатам (порог {self.threshold}):")
        for cs, ce, p in verified_sorted:
            mark = "PASS" if p >= self.threshold else " -- "
            print(f"  {mark}  {cs:8.1f}-{ce:8.1f}s  p_jump={p:.3f}")
        kept = [(cs, ce, p) for cs, ce, p in verified if p >= self.threshold]
        merged = merge_verified(kept, VERIFY_MERGE_GAP)
        print(f"[detector] прошло порог: {len(kept)}, финальных детекций после слияния: {len(merged)}")
        return merged
