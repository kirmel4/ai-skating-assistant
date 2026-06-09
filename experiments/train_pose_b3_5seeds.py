"""B3 с 5 seeds — fallback если RTMW не работает.

  target_fps = 16    (wider window, +jump)
  bbox_padding = 0.30
  bbox_smooth σ = 2.0
  pose_model = yolov8   (без RTMW — точно работает)

5 seeds для устойчивого ensemble. В отличие от ablation v3 (3 seeds)
даёт более стабильную оценку.

Использование:
    python scripts/train_pose_b3_5seeds.py
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
import experiments.train_pose_ablation_v3 as v3
from experiments.train_pose_ablation_v3 import NUM_FRAMES, NUM_JUMP_CLASSES
from src.config import DataLoaderConfig, VideoConfig


# === override v3 module-level constants ===
v3.CHECKPOINT_DIR = Path("checkpoints_pose_b3_5seeds")
v3.CHECKPOINT_DIR.mkdir(exist_ok=True)
v3.SEEDS = [42, 43, 44, 45, 46]
v3.EPOCHS = 60


B3_CONFIG = {
    "name": "B3_yolov8_5seeds",
    "target_fps": 16.0,
    "bbox_padding": 0.30,
    "bbox_smooth_sigma": 2.0,
    "pose_model": "yolov8",
}


def main():
    print(f"=== B3 5 seeds × {v3.EPOCHS} epochs (yolov8, без RTMW) ===")
    print(f"Config: {B3_CONFIG}")
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
        B3_CONFIG, df_clips, jump_labels, rotation_labels, rotation_floats,
        underrotation_labels, fall_labels, num_classes,
    )

    if result is None:
        print("\n✗ Что-то пошло не так")
        return

    print(f"\n\n{'=' * 80}")
    print("B3 (yolov8, 5 seeds) RESULT")
    print('=' * 80)
    print(f"  mean_score   = {result['mean_score']:.3f} ± {result['std_score']:.3f}")
    print(f"  ens_score    = {result['ens_score']:.3f}")
    print(f"  ens_jump_f1  = {result['ens_jump_f1']:.3f}")
    print(f"  ens_rot_f1   = {result['ens_rot_f1']:.3f}")
    print(f"  ens_under_f1 = {result['ens_under_f1']:.3f}")
    print(f"  ens_fall_f1  = {result['ens_fall_f1']:.3f}")

    print(f"\nReference (ablation v3, 3 seeds):")
    print(f"  B3:  score=0.847  jump=0.911  rot=0.725  under=0.896  fall=0.973")

    pd.DataFrame([result]).to_csv(v3.CHECKPOINT_DIR / "result.csv", index=False)
    print(f"\nCSV: {v3.CHECKPOINT_DIR / 'result.csv'}")


if __name__ == "__main__":
    main()
