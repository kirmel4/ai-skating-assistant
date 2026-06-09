"""Dual-stream pose training: static vs dynamic feature branches.

Гипотеза: F4 лучше для jump (статика без шума), F6 лучше для rot/under (динамика).
Делаем две параллельные ветки → каждая голова берёт ту, что любит.

Architecture:
    Static features (61): xy_norm + conf + angles + joint_cos     ← геометрия
        ↓ static_proj → TCN+Transformer → (B, T, 256) s_seq

    Dynamic features (69): velocity + acceleration + cum_rotation  ← производные
        ↓ dynamic_proj → TCN+Transformer → (B, T, 256) d_seq

    Routing (hard split):
        jump_head    ← s_seq                  (любит чистую геометрию, как F4)
        rot_head     ← concat(s_seq, d_seq)
        under_head   ← concat(s_seq, d_seq)
        fall_head    ← concat(s_seq, d_seq)

Smoothing σ=1.0 на сырые keypoints (помогает обоим потокам).
Hflip отключён (в v2 ухудшал результат).

Использование:
    python scripts/train_pose_dualstream.py
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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from scripts.train_dinov_2_temporal import (
    TaskAttentionHead,
    TemporalHybrid,
    make_underrotation_map,
    map_fall,
    multitask_loss,
    normalize_underrotation_value,
)
from scripts.train_pose_ablation import (
    KEYPOINT_LR_PAIRS,
    _hflip_keypoints,
    _smooth_keypoints,
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

CHECKPOINT_DIR = Path("checkpoints_pose_dualstream")
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

# Из ablation v2: smoothing помогает (F4 > F3), hflip мешает (F4 > F5)
SMOOTH_SIGMA = 1.0
USE_HFLIP = False

STATIC_DIM = 61
DYNAMIC_DIM = 69  # velocity (34) + acceleration (34) + cum_rotation (1)

NUM_JUMP_CLASSES = len(LABEL_MAP)


# ============================================================
# Feature extractors (split на 2 потока)
# ============================================================

def extract_static_features(kpts: torch.Tensor) -> torch.Tensor:
    """Геометрия без производных: 61 фича."""
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

    return torch.cat([
        xy_norm.flatten(1),  # 34
        conf,                # 17
        shoulder_sc, hip_sc, spine_sc,  # 6
        L_knee, R_knee, L_elbow, R_elbow,  # 4
    ], dim=-1)  # 61


def extract_dynamic_features(kpts: torch.Tensor) -> torch.Tensor:
    """Производные и кумулятивная ротация: 69 фич."""
    xy = kpts[..., :2]
    midhip = (xy[:, KP["L_hip"]] + xy[:, KP["R_hip"]]) / 2
    midshoulder = (xy[:, KP["L_shoulder"]] + xy[:, KP["R_shoulder"]]) / 2
    torso_len = torch.linalg.norm(midshoulder - midhip, dim=-1, keepdim=True).clamp(min=1e-3)
    xy_norm = (xy - midhip.unsqueeze(1)) / torso_len.unsqueeze(1)

    xy_velocity = torch.zeros_like(xy_norm)
    xy_velocity[1:] = xy_norm[1:] - xy_norm[:-1]

    xy_accel = torch.zeros_like(xy_norm)
    xy_accel[1:] = xy_velocity[1:] - xy_velocity[:-1]

    sh_dx = xy[:, KP["R_shoulder"], 0] - xy[:, KP["L_shoulder"], 0]
    sh_dy = xy[:, KP["R_shoulder"], 1] - xy[:, KP["L_shoulder"], 1]
    sh_angle = torch.atan2(sh_dy, sh_dx)
    sh_unwrapped = torch.from_numpy(np.unwrap(sh_angle.numpy())).float()
    cum_rotation_abs = (sh_unwrapped - sh_unwrapped[0:1]).abs() / (2 * math.pi)

    return torch.cat([
        xy_velocity.flatten(1),         # 34
        xy_accel.flatten(1),            # 34
        cum_rotation_abs.unsqueeze(-1), # 1
    ], dim=-1)  # 69


# ============================================================
# Dataset
# ============================================================

class DualStreamDataset(Dataset):
    """Возвращает (static, dynamic, jump, rot, under, fall).
    Smoothing/hflip применяются к сырым keypoints, обе ветки видят одни и те же
    обработанные kpts."""

    def __init__(
        self,
        df,
        kpts_dir: Path,
        jump_labels: np.ndarray,
        rotation_labels: np.ndarray,
        underrotation_labels: np.ndarray,
        fall_labels: np.ndarray,
        smooth_sigma: float = 0.0,
        augment_hflip: bool = False,
        augment_temporal_roll: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.kpts_dir = kpts_dir
        self.smooth_sigma = smooth_sigma
        self.augment_hflip = augment_hflip
        self.augment_temporal_roll = augment_temporal_roll

        self.jump = torch.tensor(jump_labels, dtype=torch.long)
        self.rotation = torch.tensor(rotation_labels, dtype=torch.long)
        self.underrotation = torch.tensor(underrotation_labels, dtype=torch.long)
        self.fall = torch.tensor(fall_labels, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kpts = torch.load(_kpt_cache_path(row, self.kpts_dir), weights_only=True)  # (T, 17, 3)

        # Аугментация на сырых keypoints — обе ветки увидят согласованную версию.
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            kpts = _hflip_keypoints(kpts)

        if self.augment_temporal_roll and MAX_TEMPORAL_ROLL > 0:
            shift = torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (1,)).item()
            if shift != 0:
                kpts = kpts.roll(shifts=shift, dims=0)

        if self.smooth_sigma > 0:
            kpts = _smooth_keypoints(kpts, self.smooth_sigma)

        static_feats = extract_static_features(kpts)    # (T, 61)
        dynamic_feats = extract_dynamic_features(kpts)  # (T, 69)

        return (
            static_feats, dynamic_feats,
            self.jump[idx], self.rotation[idx], self.underrotation[idx], self.fall[idx],
        )


# ============================================================
# Model
# ============================================================

class DualStreamPoseModel(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        hidden_dim: int,
        num_classes: dict,
        num_frames: int,
    ):
        super().__init__()
        # Две параллельные ветки
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.static_temporal = TemporalHybrid(dim=hidden_dim, num_frames=num_frames)

        self.dynamic_proj = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.dynamic_temporal = TemporalHybrid(dim=hidden_dim, num_frames=num_frames)

        # Hard routing: jump видит только static; rot/under/fall видят concat
        self.jump_head = TaskAttentionHead(hidden_dim, num_classes["jump"], HEAD_DROPOUT)
        self.rotation_head = TaskAttentionHead(hidden_dim * 2, num_classes["rot"], HEAD_DROPOUT)
        self.underrotation_head = TaskAttentionHead(hidden_dim * 2, num_classes["under"], HEAD_DROPOUT)
        self.fall_head = TaskAttentionHead(hidden_dim * 2, num_classes["fall"], HEAD_DROPOUT)

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor):
        s = self.static_proj(static_x)
        s_seq = self.static_temporal(s)         # (B, T, hidden_dim)

        d = self.dynamic_proj(dynamic_x)
        d_seq = self.dynamic_temporal(d)        # (B, T, hidden_dim)

        combined = torch.cat([s_seq, d_seq], dim=-1)  # (B, T, hidden_dim*2)

        j_out, j_w = self.jump_head(s_seq)              # ← static only
        r_out, r_w = self.rotation_head(combined)
        u_out, u_w = self.underrotation_head(combined)
        f_out, f_w = self.fall_head(combined)

        return {
            "jump": j_out, "rot": r_out, "under": u_out, "fall": f_out,
            "attn": {"jump": j_w, "rot": r_w, "under": u_w, "fall": f_w},
        }


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, num_classes):
    model.train()
    total_loss, total = 0.0, 0
    for static, dynamic, j, r, u, f in loader:
        static = static.to(DEVICE, non_blocking=True)
        dynamic = dynamic.to(DEVICE, non_blocking=True)
        j = j.to(DEVICE, non_blocking=True)
        r = r.to(DEVICE, non_blocking=True)
        u = u.to(DEVICE, non_blocking=True)
        f = f.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(static, dynamic)
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
    for static, dynamic, j, r, u, f in loader:
        static = static.to(DEVICE, non_blocking=True)
        dynamic = dynamic.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(static, dynamic)
        preds["jump"].extend(outputs["jump"].argmax(1).cpu().numpy())
        preds["rot"].extend(outputs["rot"].argmax(1).cpu().numpy())
        preds["under"].extend(outputs["under"].argmax(1).cpu().numpy())
        preds["fall"].extend(outputs["fall"].argmax(1).cpu().numpy())
        trues["jump"].extend(j.numpy())
        trues["rot"].extend(r.numpy())
        trues["under"].extend(u.numpy())
        trues["fall"].extend(f.numpy())
    return {k: f1_score(trues[k], preds[k], average="macro", zero_division=0) for k in preds}


def save_plots(history, epoch):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epoch + 1), history["loss"], marker="o")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss (dual-stream)")
    ax.grid(True); fig.tight_layout(); fig.savefig(CHECKPOINT_DIR / "loss.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epoch + 1), history["jump_f1"], marker="o", label="jump_type")
    ax.plot(range(1, epoch + 1), history["rot_f1"], marker="s", label="rotations")
    ax.plot(range(1, epoch + 1), history["under_f1"], marker="^", label="underrotation")
    ax.plot(range(1, epoch + 1), history["fall_f1"], marker="D", label="fall")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (macro)"); ax.set_title("Validation F1 (dual-stream)")
    ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(CHECKPOINT_DIR / "f1.png"); plt.close(fig)


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
    print(f"Architecture: dual-stream (static={STATIC_DIM} + dynamic={DYNAMIC_DIM}), hidden={TEMPORAL_HIDDEN_DIM}")
    print(f"Routing: jump←static, rot/under/fall←concat(static,dynamic)")
    print(f"Smooth σ={SMOOTH_SIGMA}, hflip={USE_HFLIP}, temporal_roll=±{MAX_TEMPORAL_ROLL}")

    # Кэши уже готовы из v1/v2
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

    torch.manual_seed(42)
    np.random.seed(42)
    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    train_dataset = DualStreamDataset(
        df_clips, kpts_dir, jump_labels, rotation_labels, underrotation_labels, fall_labels,
        smooth_sigma=SMOOTH_SIGMA, augment_hflip=USE_HFLIP, augment_temporal_roll=True,
    )
    val_dataset = DualStreamDataset(
        df_clips, kpts_dir, jump_labels, rotation_labels, underrotation_labels, fall_labels,
        smooth_sigma=SMOOTH_SIGMA, augment_hflip=False, augment_temporal_roll=False,
    )

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

    model = DualStreamPoseModel(
        static_dim=STATIC_DIM,
        dynamic_dim=DYNAMIC_DIM,
        hidden_dim=TEMPORAL_HIDDEN_DIM,
        num_classes=num_classes,
        num_frames=NUM_FRAMES,
    ).to(DEVICE)

    temporal_params = (
        list(model.static_proj.parameters())
        + list(model.static_temporal.parameters())
        + list(model.dynamic_proj.parameters())
        + list(model.dynamic_temporal.parameters())
    )
    head_params = (
        list(model.jump_head.parameters())
        + list(model.rotation_head.parameters())
        + list(model.underrotation_head.parameters())
        + list(model.fall_head.parameters())
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
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

    history = {"loss": [], "jump_f1": [], "rot_f1": [], "under_f1": [], "fall_f1": []}
    best_score = -1.0
    best_f1s = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
        f1s = eval_epoch(model, val_loader)
        score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]

        history["loss"].append(loss)
        history["jump_f1"].append(f1s["jump"])
        history["rot_f1"].append(f1s["rot"])
        history["under_f1"].append(f1s["under"])
        history["fall_f1"].append(f1s["fall"])

        print(
            f"[{epoch}/{EPOCHS}] loss={loss:.4f}  "
            f"jump={f1s['jump']:.3f}  rot={f1s['rot']:.3f}  "
            f"under={f1s['under']:.3f}  fall={f1s['fall']:.3f}  score={score:.3f}"
        )

        save_plots(history, epoch)

        if score > best_score:
            best_score = score
            best_f1s = dict(f1s)
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "rotation_map": rotation_map,
                "underrotation_map": underrotation_map,
                "label_map": LABEL_MAP,
                "config": {
                    "static_dim": STATIC_DIM,
                    "dynamic_dim": DYNAMIC_DIM,
                    "hidden_dim": TEMPORAL_HIDDEN_DIM,
                    "smooth_sigma": SMOOTH_SIGMA,
                    "use_hflip": USE_HFLIP,
                    "num_frames": NUM_FRAMES,
                },
                "f1s": f1s,
                "score": score,
                "epoch": epoch,
            }, CHECKPOINT_DIR / "best.pt")
            print(f"         -> best saved (score={score:.3f})")

    # === финальное сравнение с F4 ===
    print(f"\n{'=' * 70}")
    print(f"DUAL-STREAM RESULT  (best @ epoch {best_epoch})")
    print('=' * 70)
    print(f"  jump_f1   = {best_f1s['jump']:.3f}")
    print(f"  rot_f1    = {best_f1s['rot']:.3f}")
    print(f"  under_f1  = {best_f1s['under']:.3f}")
    print(f"  fall_f1   = {best_f1s['fall']:.3f}")
    print(f"  score     = {best_score:.3f}")
    print(f"\nFor reference (from ablation v2):")
    print(f"  F4 (best score) : score=0.793  jump=0.788  rot=0.743  under=0.866  fall=0.973")
    print(f"  F6 (best rot)   : score=0.778  jump=0.715  rot=0.781  under=0.892  fall=0.971")
    print(f"\nIdeal (task-wise): jump≈0.79, rot≈0.78, under≈0.89, fall≈0.97")


if __name__ == "__main__":
    main()
