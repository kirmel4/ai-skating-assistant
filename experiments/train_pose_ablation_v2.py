"""Ablation v2: новый baseline — hybrid backbone (TCN+Transformer). Проверяем какие
фичи дополнительно к нему помогают.

Гипотеза от v1: главный прирост дал не velocity/acceleration сам по себе, а связка
"динамические фичи + TCN перед Transformer". Теперь проверяем, нужны ли вообще фичи
поверх TCN, или достаточно базового A + hybrid.

Configs (все с hybrid backbone):
  F1_hybrid_only        — A features (61), TCN+Transformer, без других фич
  F2_+vel               — F1 + velocity (95)
  F3_+cumrot            — F2 + cum_rotation (96)
  F4_+smooth            — F3 + smoothing σ=1.0
  F5_+hflip             — F4 + horizontal flip aug
  F6_+accel  (≈ E из v1)  — F5 + acceleration (130)

Каждая дельта = вклад ровно одной фичи поверх предыдущего конфига.

Использование:
    python scripts/train_pose_ablation_v2.py

Результаты:
    checkpoints_pose_ablation_v2/results.csv
    checkpoints_pose_ablation_v2/{config_name}/best.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from scripts.train_dinov_2_temporal import (
    MultiTaskDataset,
    make_underrotation_map,
    map_fall,
    multitask_loss,
    normalize_underrotation_value,
)
from scripts.train_pose_ablation import (
    AblationKeypointsBaseDataset,
    AblationModel,
    augment_features_temporal,
    eval_epoch,
    feature_dim_for_cfg,
    train_epoch,
)
from scripts.train_pose_temporal import (
    _keypoints_dir_for_config,
    precompute_pose_keypoints,
)
from scripts.train_videomae_phase1 import CropClipDataset, detect_skater_bboxes
from src.config import DataLoaderConfig, VideoConfig


# ============================================================
# Config (те же что в v1 — для честного сравнения)
# ============================================================

DEVICE_ID = int(os.environ.get("POSE_CUDA_DEVICE", "1"))
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints_pose_ablation_v2")
CHECKPOINT_DIR.mkdir(exist_ok=True)

NUM_FRAMES = 64
IMAGE_SIZE = 224
TARGET_FPS = 25.0
EPOCHS = 60
BATCH_SIZE = 32
LR_TEMPORAL = 5e-4
LR_HEADS = 1e-3
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.05
USE_AMP = True
TEMPORAL_HIDDEN_DIM = 256

NUM_JUMP_CLASSES = len(LABEL_MAP)


# ============================================================
# Configs: каждый добавляет одну фичу к предыдущему. Все используют hybrid.
# ============================================================

CONFIGS = [
    {
        "name": "F1_hybrid_only",
        "use_velocity": False, "use_acceleration": False, "use_cum_rotation": False,
        "smooth_sigma": 0.0, "use_hflip": False,
        "temporal_backbone": "hybrid",
    },
    {
        "name": "F2_+vel",
        "use_velocity": True, "use_acceleration": False, "use_cum_rotation": False,
        "smooth_sigma": 0.0, "use_hflip": False,
        "temporal_backbone": "hybrid",
    },
    {
        "name": "F3_+cumrot",
        "use_velocity": True, "use_acceleration": False, "use_cum_rotation": True,
        "smooth_sigma": 0.0, "use_hflip": False,
        "temporal_backbone": "hybrid",
    },
    {
        "name": "F4_+smooth",
        "use_velocity": True, "use_acceleration": False, "use_cum_rotation": True,
        "smooth_sigma": 1.0, "use_hflip": False,
        "temporal_backbone": "hybrid",
    },
    {
        "name": "F5_+hflip",
        "use_velocity": True, "use_acceleration": False, "use_cum_rotation": True,
        "smooth_sigma": 1.0, "use_hflip": True,
        "temporal_backbone": "hybrid",
    },
    {
        "name": "F6_+accel",
        "use_velocity": True, "use_acceleration": True, "use_cum_rotation": True,
        "smooth_sigma": 1.0, "use_hflip": True,
        "temporal_backbone": "hybrid",
    },
]


# ============================================================
# Run experiment (копия из v1, но с CHECKPOINT_DIR этого скрипта)
# ============================================================

def run_experiment(cfg, df_clips, kpts_dir, jump_labels, rotation_labels, underrotation_labels, fall_labels, num_classes):
    name = cfg["name"]
    feat_dim = feature_dim_for_cfg(cfg)

    print(f"\n{'=' * 70}")
    print(f"Experiment: {name}  (feature_dim={feat_dim})")
    print(f"  velocity={cfg['use_velocity']}  accel={cfg['use_acceleration']}  cum_rot={cfg['use_cum_rotation']}")
    print(f"  smooth_σ={cfg['smooth_sigma']}  hflip={cfg['use_hflip']}  backbone={cfg['temporal_backbone']}")
    print('=' * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    train_base = AblationKeypointsBaseDataset(df_clips, kpts_dir, cfg, augment_hflip=cfg["use_hflip"])
    val_base = AblationKeypointsBaseDataset(df_clips, kpts_dir, cfg, augment_hflip=False)
    train_dataset = MultiTaskDataset(train_base, jump_labels, rotation_labels, underrotation_labels, fall_labels)
    val_dataset = MultiTaskDataset(val_base, jump_labels, rotation_labels, underrotation_labels, fall_labels)

    train_jl = jump_labels[train_idx]
    counts = np.maximum(np.bincount(train_jl, minlength=NUM_JUMP_CLASSES), 1)
    weights = torch.tensor((1.0 / counts)[train_jl], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    model = AblationModel(
        feature_dim=feat_dim, hidden_dim=TEMPORAL_HIDDEN_DIM,
        num_classes=num_classes, temporal_backbone=cfg["temporal_backbone"],
    ).to(DEVICE)

    temporal_params = list(model.input_proj.parameters()) + list(model.temporal.parameters())
    head_params = []
    for n, p in model.named_parameters():
        if not (n.startswith("input_proj") or n.startswith("temporal.")):
            head_params.append(p)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW([
        {"params": temporal_params, "lr": LR_TEMPORAL, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEADS, "weight_decay": WEIGHT_DECAY},
    ])
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")

    best = {"score": -1.0, "epoch": 0, "f1s": None, "loss": None}
    cfg_dir = CHECKPOINT_DIR / name
    cfg_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
        f1s = eval_epoch(model, val_loader)
        score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]

        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "f1s": dict(f1s), "loss": loss}
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "f1s": f1s,
                "score": score,
                "epoch": epoch,
            }, cfg_dir / "best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch}/{EPOCHS}] loss={loss:.4f}  jump={f1s['jump']:.3f}  rot={f1s['rot']:.3f}  under={f1s['under']:.3f}  fall={f1s['fall']:.3f}")

    print(f"  ► BEST @ epoch {best['epoch']}:  score={best['score']:.3f}")
    print(f"     jump={best['f1s']['jump']:.3f}  rot={best['f1s']['rot']:.3f}  under={best['f1s']['under']:.3f}  fall={best['f1s']['fall']:.3f}")

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()

    return {
        "config": name,
        "feature_dim": feat_dim,
        "trainable_params": trainable,
        "backbone": cfg["temporal_backbone"],
        "best_epoch": best["epoch"],
        "score": round(best["score"], 4),
        "jump_f1": round(best["f1s"]["jump"], 4),
        "rot_f1": round(best["f1s"]["rot"], 4),
        "under_f1": round(best["f1s"]["under"], 4),
        "fall_f1": round(best["f1s"]["fall"], 4),
    }


# ============================================================
# Main
# ============================================================

def main():
    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=TARGET_FPS, image_size=IMAGE_SIZE, return_meta=False)
    data_config = DataLoaderConfig(batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])
    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}
    under_values = df_clips["underrotation"].apply(normalize_underrotation_value).values
    underrotation_map = make_underrotation_map(under_values)

    print(f"Dataset: {len(df_clips)} jumps")
    print(f"Device: {DEVICE}")

    # Кэши уже должны быть готовы из v1 — переиспользуются
    bbox_cache_path = _REPO_ROOT / "data" / f"skater_bboxes_n{NUM_FRAMES}.json"
    bboxes = detect_skater_bboxes(
        df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        device=DEVICE, cache_path=bbox_cache_path,
    )
    print(f"Skater bboxes: {len(bboxes)}/{len(df_clips)}")
    frame_dataset = CropClipDataset(
        df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
    )

    kpts_dir = _keypoints_dir_for_config()
    precompute_pose_keypoints(frame_dataset, kpts_dir, DEVICE)

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    num_classes = {
        "jump": NUM_JUMP_CLASSES,
        "rot": len(rotation_map),
        "under": len(underrotation_map),
        "fall": 2,
    }

    results = []
    for cfg in CONFIGS:
        result = run_experiment(
            cfg, df_clips, kpts_dir, jump_labels, rotation_labels,
            underrotation_labels, fall_labels, num_classes,
        )
        results.append(result)
        pd.DataFrame(results).to_csv(CHECKPOINT_DIR / "results.csv", index=False)

    # === финальная сводка ===
    print(f"\n\n{'=' * 90}")
    print("ABLATION V2 RESULTS  (hybrid backbone, инкрементально по фичам)")
    print('=' * 90)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    if len(df_results) > 1:
        print(f"\n{'=' * 90}")
        print("DELTAS — вклад каждой фичи отдельно")
        print('=' * 90)
        for i in range(1, len(df_results)):
            prev, curr = df_results.iloc[i - 1], df_results.iloc[i]
            print(f"\n{curr['config']}  (vs {prev['config']}):")
            for k in ("score", "jump_f1", "rot_f1", "under_f1", "fall_f1"):
                d = curr[k] - prev[k]
                sign = "+" if d >= 0 else ""
                print(f"  {k:12s}  {curr[k]:.3f}  ({sign}{d:.3f})")

    print(f"\nCSV: {CHECKPOINT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
