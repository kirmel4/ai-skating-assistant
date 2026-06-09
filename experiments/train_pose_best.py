"""B5 super-combo: лучшие preprocessing winners из ablation v3, собранные в одну модель.

  target_fps = 16    (B1/B3 winner — wider window даёт +jump)
  bbox_padding = 0.30 (B2/B3/B4 — больше контекста ноги/конька)
  bbox_smooth σ = 2.0 (B2/B3 — убирает фейковую velocity из дрожащего bbox)
  pose_model = rtmw   (B4 — 6 foot keypoints для под/Lutz/Flip)

Multi-seed (5 seeds) → ensemble. Ожидаемо ближе всего к теоретическому
per-task потолку ~0.873.

Использование:
    pip install rtmlib onnxruntime-gpu scipy
    python scripts/train_pose_best.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from scripts.train_dinov_2_temporal import (
    make_underrotation_map,
    map_fall,
    normalize_underrotation_value,
)
# Переиспользуем всю инфраструктуру из ablation v3:
#  - run_config (тренировка + ensemble)
#  - feature extractors (RTMW версия используется автоматически по pose_model)
#  - cache path функции
import experiments.train_pose_ablation_v3 as v3
from experiments.train_pose_ablation_v3 import NUM_FRAMES, NUM_JUMP_CLASSES
from src.config import DataLoaderConfig, VideoConfig


# === Override v3 модульных констант под "best" режим ===
v3.CHECKPOINT_DIR = Path("checkpoints_pose_best")
v3.CHECKPOINT_DIR.mkdir(exist_ok=True)
v3.SEEDS = [42, 43, 44, 45, 46]   # 5 seeds для стабильности
v3.EPOCHS = 60                     # 60 эпох как в final


BEST_CONFIG = {
    "name": "B5_super_combo",
    "target_fps": 16.0,        # wider window: 64/16 = 4 сек контекста
    "bbox_padding": 0.30,      # больше контекста ноги/конька
    "bbox_smooth_sigma": 2.0,  # сглаживание bbox по времени
    "pose_model": "rtmw",      # 23 keypoints (17 body + 6 foot)
}


def main():
    print(f"=== B5 super-combo: 5 seeds × {v3.EPOCHS} epochs ===")
    print(f"Config: {BEST_CONFIG}")
    print(f"Device: {v3.DEVICE}")

    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=25.0, image_size=v3.IMAGE_SIZE, return_meta=False)
    data_config = DataLoaderConfig(batch_size=v3.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])
    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}
    inv_rotation_map = {i: v for v, i in rotation_map.items()}
    under_values = df_clips["underrotation"].apply(normalize_underrotation_value).values
    underrotation_map = make_underrotation_map(under_values)

    print(f"Dataset: {len(df_clips)} jumps")

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    rotation_floats = np.array([float(inv_rotation_map[c]) for c in rotation_labels], dtype=np.float32)
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    num_classes = {
        "jump": NUM_JUMP_CLASSES, "rot": len(rotation_map),
        "under": len(underrotation_map), "fall": 2,
    }

    result = v3.run_config(
        BEST_CONFIG, df_clips, jump_labels, rotation_labels, rotation_floats,
        underrotation_labels, fall_labels, num_classes,
    )

    if result is None:
        print("\n✗ B5 не запустился — проверь установку rtmlib")
        return

    # === финальная сводка с ссылкой на ablation v3 ===
    print(f"\n\n{'=' * 80}")
    print("B5 SUPER-COMBO RESULT")
    print('=' * 80)
    print(f"  mean_score   = {result['mean_score']:.3f} ± {result['std_score']:.3f}")
    print(f"  ens_score    = {result['ens_score']:.3f}")
    print(f"  ens_jump_f1  = {result['ens_jump_f1']:.3f}")
    print(f"  ens_rot_f1   = {result['ens_rot_f1']:.3f}")
    print(f"  ens_under_f1 = {result['ens_under_f1']:.3f}")
    print(f"  ens_fall_f1  = {result['ens_fall_f1']:.3f}")

    print(f"\nReference (from ablation v3 best per task):")
    print(f"  jump:  B3 = 0.911")
    print(f"  rot:   B2 = 0.794")
    print(f"  under: B4 = 0.906")
    print(f"  fall:  B1/B3 = 0.973")
    print(f"  per-task upper bound score = 0.873")

    pd.DataFrame([result]).to_csv(v3.CHECKPOINT_DIR / "result.csv", index=False)
    print(f"\nCSV: {v3.CHECKPOINT_DIR / 'result.csv'}")
    print(f"Per-seed: {v3.CHECKPOINT_DIR / BEST_CONFIG['name'] / 'per_seed.csv'}")


if __name__ == "__main__":
    main()
