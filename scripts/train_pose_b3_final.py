"""B3 preprocessing + F4 architecture + aux + 5 seeds + plots. Финальная сборка.

Конфиг (winners из всех ablation):
  - yolov8-pose (без RTMW/MediaPipe — проверенный, GPU стабилен)
  - target_fps = 16   (B1/B3: wider window даёт +jump)
  - bbox_padding = 0.30 + bbox smooth σ = 2.0   (B2/B3: чистые crops)
  - F4 features: 61 base + 34 velocity + 1 cum_rotation = 96
  - rot_regression aux loss (MSE на число оборотов)
  - phase_aux head (per-frame approach/takeoff/air/landing/exit)
  - 5 seeds → ensemble of averaged logits

Использование:
    python scripts/train_pose_b3_final.py

Output (checkpoints_pose_b3_final/):
  seed_{N}/best.pt, loss.png, f1.png
  ensemble.png  — финальная F1 диаграмма
  summary.json
  per_seed.csv
"""

from __future__ import annotations

import json
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    normalize_underrotation_value,
)
from scripts.train_pose_ablation import _smooth_keypoints
from scripts.train_pose_temporal import (
    KP,
    _joint_cos,
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

CHECKPOINT_DIR = Path("checkpoints_pose_b3_final")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# B3 preprocessing winners
NUM_FRAMES = 64
TARGET_FPS = 16.0           # wider window (≈4 сек контекста)
BBOX_PADDING = 0.30          # больше контекста ноги/конька
BBOX_SMOOTH_SIGMA = 2.0      # сглаживание bbox по времени
SMOOTH_SIGMA = 1.0           # сглаживание keypoints по времени
IMAGE_SIZE = 224

# F4 features
FEATURE_DIM = 96             # 61 base + 34 velocity + 1 cum_rotation

# Training
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

# Aux losses
LAMBDA_ROT_REG = 0.3
LAMBDA_PHASE = 0.1
NUM_PHASES = 5
PHASE_HALF_WIDTH = 0.15

# Multi-seed
SEEDS = [42, 43, 44, 45, 46]

# Аксель — главный источник ошибок (двойной аксель = 2.5 оборота, визуально
# между 2 и 3). Дополнительно поднимаем его вес в сэмплере поверх обратной
# частоты типов. 1.0 = без доп. веса; тюнится.
AXEL_SAMPLE_WEIGHT = 2.0

NUM_JUMP_CLASSES = len(LABEL_MAP)

# Cache paths (зависят от preprocessing)
BBOX_CACHE_PATH = _REPO_ROOT / "data" / (
    f"skater_bboxes_n{NUM_FRAMES}_fps{int(TARGET_FPS)}"
    f"_pad{int(BBOX_PADDING * 100):03d}_bsm{int(BBOX_SMOOTH_SIGMA * 10):02d}.json"
)
POSE_CACHE_DIR = _REPO_ROOT / "data" / (
    f"pose_kpts_yolov8_n{NUM_FRAMES}_fps{int(TARGET_FPS)}"
    f"_pad{int(BBOX_PADDING * 100):03d}_bsm{int(BBOX_SMOOTH_SIGMA * 10):02d}"
)


# ============================================================
# Bbox detection + smoothing
# ============================================================

def detect_and_smooth_bboxes(df_clips) -> dict:
    """YOLO bbox с padding=0.30 + сглаживание по времени σ=2.0."""
    bboxes = detect_skater_bboxes(
        df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        device=DEVICE, cache_path=BBOX_CACHE_PATH, padding=BBOX_PADDING,
    )
    if BBOX_SMOOTH_SIGMA > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed = {}
        for k, v in bboxes.items():
            arr = np.array(v, dtype=np.float32)
            arr_s = gaussian_filter1d(arr, sigma=BBOX_SMOOTH_SIGMA, axis=0, mode="nearest")
            smoothed[k] = arr_s.tolist()
        return smoothed
    return bboxes


# ============================================================
# Phase pseudo-labels (parametric)
# ============================================================

def compute_frame_timestamps(start_sec, end_sec, num_frames=NUM_FRAMES, target_fps=TARGET_FPS):
    seg = max(end_sec - start_sec, 1e-6)
    target = num_frames / target_fps
    if seg >= target:
        ws, we = start_sec, end_sec
    else:
        c = (start_sec + end_sec) / 2.0
        h = target / 2.0
        ws = min(c - h, start_sec)
        we = max(c + h, end_sec)
    return np.linspace(ws, we, num=num_frames, endpoint=True)


def compute_phase_labels(start_sec_in_clip, end_sec_in_clip, num_frames=NUM_FRAMES):
    timestamps = compute_frame_timestamps(start_sec_in_clip, end_sec_in_clip, num_frames, TARGET_FPS)
    takeoff = start_sec_in_clip
    landing = max(end_sec_in_clip - 1.0, start_sec_in_clip + 0.1)
    hw = PHASE_HALF_WIDTH
    labels = np.zeros(num_frames, dtype=np.int64)
    for i, t in enumerate(timestamps):
        if t < takeoff - hw: labels[i] = 0
        elif t < takeoff + hw: labels[i] = 1
        elif t < landing - hw: labels[i] = 2
        elif t < landing + hw: labels[i] = 3
        else: labels[i] = 4
    return labels


# ============================================================
# F4 features (96-dim)
# ============================================================

def extract_features_f4(kpts: torch.Tensor, smooth_sigma: float = 0.0) -> torch.Tensor:
    if smooth_sigma > 0:
        kpts = _smooth_keypoints(kpts, smooth_sigma)
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

    xy_velocity = torch.zeros_like(xy_norm)
    xy_velocity[1:] = xy_norm[1:] - xy_norm[:-1]

    sh_dx = xy[:, KP["R_shoulder"], 0] - xy[:, KP["L_shoulder"], 0]
    sh_dy = xy[:, KP["R_shoulder"], 1] - xy[:, KP["L_shoulder"], 1]
    sh_angle = torch.atan2(sh_dy, sh_dx)
    sh_unwrapped = torch.from_numpy(np.unwrap(sh_angle.numpy())).float()
    cum_rot = (sh_unwrapped - sh_unwrapped[0:1]).abs() / (2 * math.pi)

    return torch.cat([
        xy_norm.flatten(1), conf, shoulder_sc, hip_sc, spine_sc,
        L_knee, R_knee, L_elbow, R_elbow,
        xy_velocity.flatten(1), cum_rot.unsqueeze(-1),
    ], dim=-1)  # 96


# ============================================================
# Dataset
# ============================================================

class B3Dataset(Dataset):
    def __init__(
        self, df, kpts_dir,
        jump_labels, rotation_labels, rotation_floats,
        underrotation_labels, fall_labels,
        augment_temporal_roll=False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.kpts_dir = kpts_dir
        self.augment_temporal_roll = augment_temporal_roll
        self.jump = torch.tensor(jump_labels, dtype=torch.long)
        self.rotation = torch.tensor(rotation_labels, dtype=torch.long)
        self.rotation_float = torch.tensor(rotation_floats, dtype=torch.float)
        self.underrotation = torch.tensor(underrotation_labels, dtype=torch.long)
        self.fall = torch.tensor(fall_labels, dtype=torch.long)

        # phase pseudo-labels (один раз)
        phase_list = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            phase_list.append(compute_phase_labels(
                float(row["start_sec_in_clip"]), float(row["end_sec_in_clip"]),
            ))
        self.phase_labels = torch.tensor(np.stack(phase_list), dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kpts = torch.load(_kpt_cache_path(row, self.kpts_dir), weights_only=True)
        phase = self.phase_labels[idx].clone()

        if self.augment_temporal_roll and MAX_TEMPORAL_ROLL > 0:
            shift = int(torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (1,)).item())
            if shift != 0:
                kpts = kpts.roll(shifts=shift, dims=0)
                phase = phase.roll(shifts=shift, dims=0)

        feats = extract_features_f4(kpts, smooth_sigma=SMOOTH_SIGMA)
        return (
            feats, self.jump[idx], self.rotation[idx], self.rotation_float[idx],
            self.underrotation[idx], self.fall[idx], phase,
        )


# ============================================================
# Model + Loss
# ============================================================

class B3PoseModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, num_phases=NUM_PHASES, num_frames=NUM_FRAMES):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.temporal = TemporalHybrid(dim=hidden_dim, num_frames=num_frames)
        self.jump_head = TaskAttentionHead(hidden_dim, num_classes["jump"], HEAD_DROPOUT)
        self.rotation_head = TaskAttentionHead(hidden_dim, num_classes["rot"], HEAD_DROPOUT)
        self.underrotation_head = TaskAttentionHead(hidden_dim, num_classes["under"], HEAD_DROPOUT)
        self.fall_head = TaskAttentionHead(hidden_dim, num_classes["fall"], HEAD_DROPOUT)
        self.rot_regression_head = TaskAttentionHead(hidden_dim, 1, HEAD_DROPOUT)
        self.phase_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(HEAD_DROPOUT), nn.Linear(hidden_dim, num_phases),
        )

    def forward(self, x):
        x = self.input_proj(x)
        seq = self.temporal(x)
        j_out, _ = self.jump_head(seq)
        r_out, _ = self.rotation_head(seq)
        u_out, _ = self.underrotation_head(seq)
        f_out, _ = self.fall_head(seq)
        rot_reg, _ = self.rot_regression_head(seq)
        rot_reg = rot_reg.squeeze(-1)
        phase_logits = self.phase_head(seq)
        return {
            "jump": j_out, "rot": r_out, "under": u_out, "fall": f_out,
            "rot_reg": rot_reg, "phase": phase_logits,
        }


def full_loss(outputs, labels, criterion, num_classes):
    j, r_class, r_float, u, f, phase_lbl = labels
    main = (
        criterion(outputs["jump"], j) / math.log(num_classes["jump"])
        + criterion(outputs["rot"], r_class) / math.log(num_classes["rot"])
        + criterion(outputs["under"], u) / math.log(num_classes["under"])
        + 0.5 * criterion(outputs["fall"], f) / math.log(num_classes["fall"])
    )
    rot_reg_loss = F.mse_loss(outputs["rot_reg"], r_float)
    phase_loss = F.cross_entropy(
        outputs["phase"].reshape(-1, outputs["phase"].size(-1)),
        phase_lbl.reshape(-1),
    )
    return main + LAMBDA_ROT_REG * rot_reg_loss + LAMBDA_PHASE * phase_loss


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, num_classes):
    model.train()
    total_loss, total = 0.0, 0
    for feats, j, r_class, r_float, u, f, phase in loader:
        feats = feats.to(DEVICE, non_blocking=True)
        j = j.to(DEVICE, non_blocking=True)
        r_class = r_class.to(DEVICE, non_blocking=True)
        r_float = r_float.to(DEVICE, non_blocking=True)
        u = u.to(DEVICE, non_blocking=True)
        f = f.to(DEVICE, non_blocking=True)
        phase = phase.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(feats)
            loss = full_loss(outputs, (j, r_class, r_float, u, f, phase), criterion, num_classes)
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
def eval_epoch(model, loader, return_logits=False):
    model.eval()
    preds = {k: [] for k in ("jump", "rot", "under", "fall")}
    trues = {k: [] for k in ("jump", "rot", "under", "fall")}
    logits_buf = {k: [] for k in ("jump", "rot", "under", "fall")}
    for feats, j, r, _rf, u, f, _phase in loader:
        feats = feats.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(feats)
        for k in preds:
            logits = outputs[k].float()
            preds[k].extend(logits.argmax(1).cpu().numpy())
            if return_logits:
                logits_buf[k].append(logits.cpu())
        trues["jump"].extend(j.numpy())
        trues["rot"].extend(r.numpy())
        trues["under"].extend(u.numpy())
        trues["fall"].extend(f.numpy())
    f1s = {k: f1_score(trues[k], preds[k], average="macro", zero_division=0) for k in preds}
    if return_logits:
        return f1s, {k: torch.cat(logits_buf[k]) for k in preds}, trues
    return f1s


# ============================================================
# Plots
# ============================================================

def save_plots_per_seed(history, ckpt_dir, seed):
    """Loss и F1 по эпохам для одного seed."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["loss"], marker="o", color="tab:blue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss")
    ax.set_title(f"Training Loss — seed {seed}")
    ax.grid(True); fig.tight_layout()
    fig.savefig(ckpt_dir / "loss.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["jump_f1"], marker="o", label="jump")
    ax.plot(epochs, history["rot_f1"], marker="s", label="rot")
    ax.plot(epochs, history["under_f1"], marker="^", label="under")
    ax.plot(epochs, history["fall_f1"], marker="D", label="fall")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (macro)")
    ax.set_title(f"Validation F1 — seed {seed}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(ckpt_dir / "f1.png"); plt.close(fig)


def save_ensemble_plot(per_seed_results, ens_f1s, ens_score, mean_score, std_score):
    """Итоговая диаграмма: F1 по каждой задаче для каждого seed + ensemble."""
    tasks = ["jump_f1", "rot_f1", "under_f1", "fall_f1"]
    n_seeds = len(per_seed_results)
    x = np.arange(len(tasks))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(per_seed_results):
        ax.bar(x + (i - n_seeds / 2) * width, [r[t] for t in tasks], width, label=f"seed {r['seed']}", alpha=0.7)
    ax.bar(x + (n_seeds / 2) * width, [ens_f1s[t.replace("_f1", "")] for t in tasks],
           width, label="ENSEMBLE", color="black", alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels([t.replace("_f1", "") for t in tasks])
    ax.set_ylabel("F1 (macro)"); ax.set_ylim([0, 1])
    ax.set_title(f"Per-seed + ensemble F1\nscore: mean={mean_score:.3f}±{std_score:.3f},  ensemble={ens_score:.3f}")
    ax.legend(loc="lower right"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "ensemble.png"); plt.close(fig)


# ============================================================
# Single seed
# ============================================================

def run_one_seed(seed, df_clips, kpts_dir, jump_labels, rotation_labels, rotation_floats,
                 underrotation_labels, fall_labels, num_classes):
    print(f"\n{'=' * 70}\nSeed: {seed}\n{'=' * 70}")
    seed_dir = CHECKPOINT_DIR / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = B3Dataset(
        df_clips, kpts_dir,
        jump_labels, rotation_labels, rotation_floats, underrotation_labels, fall_labels,
        augment_temporal_roll=True,
    )
    val_dataset = B3Dataset(
        df_clips, kpts_dir,
        jump_labels, rotation_labels, rotation_floats, underrotation_labels, fall_labels,
        augment_temporal_roll=False,
    )

    train_jl = jump_labels[train_idx]
    counts = np.maximum(np.bincount(train_jl, minlength=NUM_JUMP_CLASSES), 1)
    weights = (1.0 / counts)[train_jl].astype(np.float64)
    # доп. вес акселю поверх обратной частоты типов
    weights[train_jl == LABEL_MAP["A"]] *= AXEL_SAMPLE_WEIGHT
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    model = B3PoseModel(
        feature_dim=FEATURE_DIM, hidden_dim=TEMPORAL_HIDDEN_DIM, num_classes=num_classes,
    ).to(DEVICE)
    temporal_params = list(model.input_proj.parameters()) + list(model.temporal.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not (n.startswith("input_proj") or n.startswith("temporal."))]

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
    best = {"score": -1.0, "epoch": 0, "f1s": None}

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
        f1s = eval_epoch(model, val_loader)
        score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]

        history["loss"].append(loss)
        history["jump_f1"].append(f1s["jump"])
        history["rot_f1"].append(f1s["rot"])
        history["under_f1"].append(f1s["under"])
        history["fall_f1"].append(f1s["fall"])

        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "f1s": dict(f1s)}
            torch.save({
                "model_state_dict": model.state_dict(),
                "score": score, "f1s": f1s, "epoch": epoch, "seed": seed,
            }, seed_dir / "best.pt")

        # графики обновляем каждые 5 эпох + на финале
        if epoch % 5 == 0 or epoch == EPOCHS or epoch == 1:
            save_plots_per_seed(history, seed_dir, seed)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{epoch}/{EPOCHS}] loss={loss:.4f}  "
                  f"jump={f1s['jump']:.3f}  rot={f1s['rot']:.3f}  "
                  f"under={f1s['under']:.3f}  fall={f1s['fall']:.3f}  score={score:.3f}")

    save_plots_per_seed(history, seed_dir, seed)

    # подгрузить best и собрать val логиты для ensemble
    ckpt = torch.load(seed_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    f1s_best, val_logits, val_trues = eval_epoch(model, val_loader, return_logits=True)

    print(f"  ► best @ epoch {best['epoch']}:  "
          f"jump={best['f1s']['jump']:.3f}  rot={best['f1s']['rot']:.3f}  "
          f"under={best['f1s']['under']:.3f}  fall={best['f1s']['fall']:.3f}  "
          f"score={best['score']:.3f}")

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "score": round(best["score"], 4),
        "jump_f1": round(best["f1s"]["jump"], 4),
        "rot_f1": round(best["f1s"]["rot"], 4),
        "under_f1": round(best["f1s"]["under"], 4),
        "fall_f1": round(best["f1s"]["fall"], 4),
    }, val_logits, val_trues


# ============================================================
# Main
# ============================================================

def main():
    print(f"=== B3 Final: 5 seeds × {EPOCHS} epochs ===")
    print(f"Device: {DEVICE}")
    print(f"Preprocessing: target_fps={TARGET_FPS}, bbox_padding={BBOX_PADDING}, bbox_smooth_σ={BBOX_SMOOTH_SIGMA}")
    print(f"Architecture: F4 features (96) + TCN+Transformer + 4 task heads + rot_reg + phase_aux")
    print()

    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=25.0, image_size=IMAGE_SIZE, return_meta=False)
    data_config = DataLoaderConfig(batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])
    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}
    inv_rotation_map = {i: v for v, i in rotation_map.items()}
    under_values = df_clips["underrotation"].apply(normalize_underrotation_value).values
    underrotation_map = make_underrotation_map(under_values)

    print(f"Dataset: {len(df_clips)} jumps")

    # === bbox + pose precompute (один раз) ===
    bboxes = detect_and_smooth_bboxes(df_clips)
    print(f"Skater bboxes: {len(bboxes)}/{len(df_clips)}")

    frame_dataset = CropClipDataset(
        df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
    )
    precompute_pose_keypoints(frame_dataset, POSE_CACHE_DIR, DEVICE)

    # === labels ===
    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    rotation_floats = np.array([float(inv_rotation_map[c]) for c in rotation_labels], dtype=np.float32)
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    num_classes = {
        "jump": NUM_JUMP_CLASSES, "rot": len(rotation_map),
        "under": len(underrotation_map), "fall": 2,
    }

    # === train all seeds ===
    seed_results = []
    seed_logits = {}
    val_trues = None

    for seed in SEEDS:
        result, val_logits, val_trues = run_one_seed(
            seed, df_clips, POSE_CACHE_DIR,
            jump_labels, rotation_labels, rotation_floats, underrotation_labels, fall_labels,
            num_classes,
        )
        seed_results.append(result)
        seed_logits[seed] = val_logits
        pd.DataFrame(seed_results).to_csv(CHECKPOINT_DIR / "per_seed.csv", index=False)

    # === summary ===
    print(f"\n\n{'=' * 80}\nPER-SEED RESULTS\n{'=' * 80}")
    df_results = pd.DataFrame(seed_results)
    print(df_results.to_string(index=False))
    means = df_results[["score", "jump_f1", "rot_f1", "under_f1", "fall_f1"]].mean()
    stds = df_results[["score", "jump_f1", "rot_f1", "under_f1", "fall_f1"]].std()
    print(f"\nMean ± Std across {len(SEEDS)} seeds:")
    for k in ["score", "jump_f1", "rot_f1", "under_f1", "fall_f1"]:
        print(f"  {k:10s}  {means[k]:.3f} ± {stds[k]:.3f}")

    # === ensemble ===
    tasks = ("jump", "rot", "under", "fall")
    avg_logits = {k: torch.stack([seed_logits[s][k] for s in SEEDS]).mean(dim=0) for k in tasks}
    ens_f1s = {k: f1_score(val_trues[k], avg_logits[k].argmax(1).numpy(), average="macro", zero_division=0) for k in tasks}
    ens_score = 0.45 * ens_f1s["jump"] + 0.35 * ens_f1s["rot"] + 0.15 * ens_f1s["under"] + 0.05 * ens_f1s["fall"]

    print(f"\n{'=' * 80}\nENSEMBLE (averaged logits across 5 seeds)\n{'=' * 80}")
    for k in tasks:
        print(f"  {k}_f1  = {ens_f1s[k]:.3f}")
    print(f"  score   = {ens_score:.3f}")

    print(f"\nReference (ablation v3 B3, 3 seeds):")
    print(f"  score=0.847  jump=0.911  rot=0.725  under=0.896  fall=0.973")

    # === финальная диаграмма ===
    save_ensemble_plot(seed_results, ens_f1s, ens_score, means["score"], stds["score"])

    summary = {
        "per_seed": seed_results,
        "mean": {k: float(means[k]) for k in means.index},
        "std": {k: float(stds[k]) for k in stds.index},
        "ensemble": {**{k: float(v) for k, v in ens_f1s.items()}, "score": float(ens_score)},
    }
    (CHECKPOINT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary: {CHECKPOINT_DIR / 'summary.json'}")
    print(f"Ensemble plot: {CHECKPOINT_DIR / 'ensemble.png'}")


if __name__ == "__main__":
    main()
