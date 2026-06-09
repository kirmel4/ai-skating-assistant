"""Ablation study на pose-features. 5 конфигов, каждый добавляет одну фичу к предыдущему.

Цель: изолированно оценить вклад каждого изменения относительно baseline.

Configs:
  A_baseline         — 61 фича, transformer (текущий best)
  B_velocity         — A + velocity Δ(x,y) per keypoint (+34, dim=95)
  C_acceleration     — B + acceleration Δ²(x,y) (+34, dim=129)
  D_full_features    — C + cum_rotation + smoothing σ=1.0 + hflip aug (+1, dim=130)
  E_hybrid_backbone  — D + TCN перед Transformer

Использует кэшированные сырые keypoints из data/pose_keypoints_n64_crop/ — feature extraction
делается на лету. YOLOv8-pose precompute запускается один раз (если кэша ещё нет).

Каждая тренировка ~3-5 минут (60 эпох на cached features). Суммарно ~15-25 минут.

Выход:
  checkpoints_pose_ablation/results.csv
  checkpoints_pose_ablation/{config_name}/best.pt

Использование:
    python scripts/train_pose_ablation.py
"""

from __future__ import annotations

import math
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
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from scripts.train_dinov_2_temporal import (
    MultiTaskDataset,
    TaskAttentionHead,
    TemporalHybrid,
    TemporalTCN,
    TemporalTransformer,
    make_underrotation_map,
    map_fall,
    multitask_loss,
    normalize_underrotation_value,
)
from scripts.train_pose_temporal import (
    KP,
    _joint_cos,
    _keypoints_dir_for_config,
    _kpt_cache_path,
    _line_sincos,
    precompute_pose_keypoints,
)
from scripts.train_videomae_phase1 import CropClipDataset, detect_skater_bboxes
from src.config import DataLoaderConfig, VideoConfig


# ============================================================
# Config
# ============================================================

DEVICE_ID = int(os.environ.get("POSE_CUDA_DEVICE", "1"))
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints_pose_ablation")
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
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.05
HEAD_DROPOUT = 0.35
USE_AMP = True
TEMPORAL_HIDDEN_DIM = 256
MAX_TEMPORAL_ROLL = 8

NUM_JUMP_CLASSES = len(LABEL_MAP)


CONFIGS = [
    {
        "name": "A_baseline",
        "use_velocity": False, "use_acceleration": False, "use_cum_rotation": False,
        "smooth_sigma": 0.0, "use_hflip": False,
        "temporal_backbone": "transformer",
    },
    {
        "name": "B_velocity",
        "use_velocity": True, "use_acceleration": False, "use_cum_rotation": False,
        "smooth_sigma": 0.0, "use_hflip": False,
        "temporal_backbone": "transformer",
    },
    {
        "name": "C_acceleration",
        "use_velocity": True, "use_acceleration": True, "use_cum_rotation": False,
        "smooth_sigma": 0.0, "use_hflip": False,
        "temporal_backbone": "transformer",
    },
    {
        "name": "D_full_features",
        "use_velocity": True, "use_acceleration": True, "use_cum_rotation": True,
        "smooth_sigma": 1.0, "use_hflip": True,
        "temporal_backbone": "transformer",
    },
    {
        "name": "E_hybrid_backbone",
        "use_velocity": True, "use_acceleration": True, "use_cum_rotation": True,
        "smooth_sigma": 1.0, "use_hflip": True,
        "temporal_backbone": "hybrid",
    },
]


# ============================================================
# Helpers
# ============================================================

KEYPOINT_LR_PAIRS = [
    (KP["L_eye"], KP["R_eye"]),
    (KP["L_ear"], KP["R_ear"]),
    (KP["L_shoulder"], KP["R_shoulder"]),
    (KP["L_elbow"], KP["R_elbow"]),
    (KP["L_wrist"], KP["R_wrist"]),
    (KP["L_hip"], KP["R_hip"]),
    (KP["L_knee"], KP["R_knee"]),
    (KP["L_ankle"], KP["R_ankle"]),
]


def _smooth_keypoints(kpts: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return kpts
    from scipy.ndimage import gaussian_filter1d
    kpts = kpts.clone()
    xy_smoothed = gaussian_filter1d(kpts[..., :2].numpy(), sigma=sigma, axis=0, mode="nearest")
    kpts[..., :2] = torch.from_numpy(xy_smoothed)
    return kpts


def _hflip_keypoints(kpts: torch.Tensor) -> torch.Tensor:
    kpts = kpts.clone()
    kpts[..., 0] = -kpts[..., 0]
    for L, R in KEYPOINT_LR_PAIRS:
        tmp = kpts[:, L].clone()
        kpts[:, L] = kpts[:, R]
        kpts[:, R] = tmp
    return kpts


def feature_dim_for_cfg(cfg: dict) -> int:
    F = 61
    if cfg["use_velocity"]:
        F += 34
    if cfg["use_acceleration"]:
        F += 34
    if cfg["use_cum_rotation"]:
        F += 1
    return F


def extract_pose_features(kpts: torch.Tensor, cfg: dict) -> torch.Tensor:
    if cfg["smooth_sigma"] > 0:
        kpts = _smooth_keypoints(kpts, cfg["smooth_sigma"])

    xy = kpts[..., :2]
    conf = kpts[..., 2]

    midhip = (xy[:, KP["L_hip"]] + xy[:, KP["R_hip"]]) / 2
    midshoulder = (xy[:, KP["L_shoulder"]] + xy[:, KP["R_shoulder"]]) / 2
    torso_len = torch.linalg.norm(midshoulder - midhip, dim=-1, keepdim=True).clamp(min=1e-3)
    xy_norm = (xy - midhip.unsqueeze(1)) / torso_len.unsqueeze(1)

    shoulder_sc = _line_sincos(xy[:, KP["L_shoulder"]], xy[:, KP["R_shoulder"]])
    hip_sc = _line_sincos(xy[:, KP["L_hip"]], xy[:, KP["R_hip"]])
    spine_sc = _line_sincos(midhip, midshoulder)

    L_knee = _joint_cos(xy[:, KP["L_hip"]], xy[:, KP["L_knee"]], xy[:, KP["L_ankle"]])
    R_knee = _joint_cos(xy[:, KP["R_hip"]], xy[:, KP["R_knee"]], xy[:, KP["R_ankle"]])
    L_elbow = _joint_cos(xy[:, KP["L_shoulder"]], xy[:, KP["L_elbow"]], xy[:, KP["L_wrist"]])
    R_elbow = _joint_cos(xy[:, KP["R_shoulder"]], xy[:, KP["R_elbow"]], xy[:, KP["R_wrist"]])

    feats_list = [
        xy_norm.flatten(1),  # 34
        conf,                # 17
        shoulder_sc,         # 2
        hip_sc,              # 2
        spine_sc,            # 2
        L_knee, R_knee,      # 2
        L_elbow, R_elbow,    # 2
    ]

    xy_velocity = None
    if cfg["use_velocity"] or cfg["use_acceleration"]:
        xy_velocity = torch.zeros_like(xy_norm)
        xy_velocity[1:] = xy_norm[1:] - xy_norm[:-1]

    if cfg["use_velocity"]:
        feats_list.append(xy_velocity.flatten(1))  # +34

    if cfg["use_acceleration"]:
        xy_accel = torch.zeros_like(xy_norm)
        xy_accel[1:] = xy_velocity[1:] - xy_velocity[:-1]
        feats_list.append(xy_accel.flatten(1))     # +34

    if cfg["use_cum_rotation"]:
        sh_dx = xy[:, KP["R_shoulder"], 0] - xy[:, KP["L_shoulder"], 0]
        sh_dy = xy[:, KP["R_shoulder"], 1] - xy[:, KP["L_shoulder"], 1]
        sh_angle = torch.atan2(sh_dy, sh_dx)
        sh_unwrapped = torch.from_numpy(np.unwrap(sh_angle.numpy())).float()
        cum_rotation_abs = (sh_unwrapped - sh_unwrapped[0:1]).abs() / (2 * math.pi)
        feats_list.append(cum_rotation_abs.unsqueeze(-1))  # +1

    return torch.cat(feats_list, dim=-1)


# ============================================================
# Dataset / Model / Train / Eval
# ============================================================

class AblationKeypointsBaseDataset(Dataset):
    def __init__(self, df, kpts_dir: Path, cfg: dict, augment_hflip: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.kpts_dir = kpts_dir
        self.cfg = cfg
        self.augment_hflip = augment_hflip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kpts = torch.load(_kpt_cache_path(row, self.kpts_dir), weights_only=True)
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            kpts = _hflip_keypoints(kpts)
        feats = extract_pose_features(kpts, self.cfg)
        return feats, 0


class AblationModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: dict, temporal_backbone: str):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if temporal_backbone == "transformer":
            self.temporal = TemporalTransformer(dim=hidden_dim, num_frames=NUM_FRAMES)
        elif temporal_backbone == "tcn":
            self.temporal = TemporalTCN(dim=hidden_dim, num_frames=NUM_FRAMES)
        elif temporal_backbone == "hybrid":
            self.temporal = TemporalHybrid(dim=hidden_dim, num_frames=NUM_FRAMES)
        else:
            raise ValueError(temporal_backbone)

        self.jump_head = TaskAttentionHead(hidden_dim, num_classes["jump"], HEAD_DROPOUT)
        self.rotation_head = TaskAttentionHead(hidden_dim, num_classes["rot"], HEAD_DROPOUT)
        self.underrotation_head = TaskAttentionHead(hidden_dim, num_classes["under"], HEAD_DROPOUT)
        self.fall_head = TaskAttentionHead(hidden_dim, num_classes["fall"], HEAD_DROPOUT)

    def forward(self, x):
        x = self.input_proj(x)
        seq = self.temporal(x)
        j_out, _ = self.jump_head(seq)
        r_out, _ = self.rotation_head(seq)
        u_out, _ = self.underrotation_head(seq)
        f_out, _ = self.fall_head(seq)
        return {"jump": j_out, "rot": r_out, "under": u_out, "fall": f_out}


def augment_features_temporal(x: torch.Tensor) -> torch.Tensor:
    """temporal_roll ±MAX_TEMPORAL_ROLL — единственная аугментация в feature space."""
    shifts = torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (x.shape[0],))
    if (shifts != 0).any():
        x = x.clone()
        for i, shift in enumerate(shifts.tolist()):
            if shift != 0:
                x[i] = x[i].roll(shifts=shift, dims=0)
    return x


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, num_classes: dict):
    model.train()
    total_loss, total = 0.0, 0
    for features, j, r, u, f in loader:
        features = augment_features_temporal(features).to(DEVICE, non_blocking=True)
        j = j.to(DEVICE, non_blocking=True)
        r = r.to(DEVICE, non_blocking=True)
        u = u.to(DEVICE, non_blocking=True)
        f = f.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(features)
            loss = multitask_loss(outputs, (j, r, u, f), criterion, num_classes)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(j)
        total += len(j)
    return total_loss / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    preds = {k: [] for k in ("jump", "rot", "under", "fall")}
    trues = {k: [] for k in ("jump", "rot", "under", "fall")}
    for features, j, r, u, f in loader:
        features = features.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(features)
        preds["jump"].extend(outputs["jump"].argmax(1).cpu().numpy())
        preds["rot"].extend(outputs["rot"].argmax(1).cpu().numpy())
        preds["under"].extend(outputs["under"].argmax(1).cpu().numpy())
        preds["fall"].extend(outputs["fall"].argmax(1).cpu().numpy())
        trues["jump"].extend(j.numpy())
        trues["rot"].extend(r.numpy())
        trues["under"].extend(u.numpy())
        trues["fall"].extend(f.numpy())
    return {k: f1_score(trues[k], preds[k], average="macro", zero_division=0) for k in preds}


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

    # cleanup перед следующим экспериментом
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

    # YOLO bbox + pose precompute (один раз, кэшируется)
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
        # сохраняем после каждого, чтобы не потерять при сбое
        pd.DataFrame(results).to_csv(CHECKPOINT_DIR / "results.csv", index=False)

    # === финальная сводка ===
    print(f"\n\n{'=' * 90}")
    print("ABLATION RESULTS")
    print('=' * 90)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # дельты между конфигами
    if len(df_results) > 1:
        print(f"\n{'=' * 90}")
        print("DELTAS vs previous config")
        print('=' * 90)
        for i in range(1, len(df_results)):
            prev, curr = df_results.iloc[i - 1], df_results.iloc[i]
            print(f"\n{curr['config']}  (vs {prev['config']}):")
            for k in ("score", "jump_f1", "rot_f1", "under_f1", "fall_f1"):
                d = curr[k] - prev[k]
                sign = "+" if d >= 0 else ""
                print(f"  {k:12s}  {curr[k]:.3f}  ({sign}{d:.3f})")

    print(f"\nCSV: {CHECKPOINT_DIR / 'results.csv'}")
    print(f"Чекпоинты: {CHECKPOINT_DIR}/{{config_name}}/best.pt")


if __name__ == "__main__":
    main()
