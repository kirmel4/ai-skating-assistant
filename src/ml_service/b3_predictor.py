"""B3-классификатор прыжков для инференса.

B3PoseModel: pose F4-фичи (96-dim) → TCN+Transformer → 4 головы
(тип прыжка / обороты / недокрут / падение) + регрессия числа оборотов.

По требованию — один чекпойнт (seed 45 — лучший по score), без ансамбля.

num_classes для оборотов и недокрута читаются из data/b3_meta.json
(скрипт scripts/export_b3_meta.py), т.к. в самом чекпойнте их нет.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.train_pose_b3_final import (
    FEATURE_DIM,
    SMOOTH_SIGMA,
    TEMPORAL_HIDDEN_DIM,
    B3PoseModel,
    extract_features_f4,
)
from src.ml_service.contracts import RawPrediction


class B3JumpClassifier:
    """Классификатор одного прыжка по keypoints окна. Один чекпойнт B3."""

    def __init__(self, checkpoint_path: Path, meta_path: Path, device: torch.device):
        self.device = device

        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        # B3PoseModel ждёт num_classes с ключами jump/rot/under/fall
        self.num_classes = {k: int(v) for k, v in meta["num_classes"].items()}

        self.model = B3PoseModel(
            feature_dim=FEATURE_DIM,
            hidden_dim=TEMPORAL_HIDDEN_DIM,
            num_classes=self.num_classes,
        ).to(device)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"B3 чекпойнт не найден: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def classify(self, kpts: torch.Tensor) -> RawPrediction:
        """kpts: (T, 17, 3) — keypoints одного окна прыжка (CPU-тензор).

        extract_features_f4 частично работает через numpy (unwrap),
        поэтому keypoints должны быть на CPU; на device уходят уже фичи.
        """
        feats = extract_features_f4(kpts, smooth_sigma=SMOOTH_SIGMA)
        feats = feats.unsqueeze(0).to(self.device)  # (1, T, 96)

        out = self.model(feats)

        return RawPrediction(
            jump_type_id=int(out["jump"].argmax(dim=1).item()),
            rotation_id=int(out["rot"].argmax(dim=1).item()),
            underrotation_id=int(out["under"].argmax(dim=1).item()),
            fall_id=int(out["fall"].argmax(dim=1).item()),
            rotations_value=float(out["rot_reg"].reshape(-1)[0].item()),
        )
