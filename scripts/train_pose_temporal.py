"""Pose-based multi-task training. Зеркалит train_dinov_2_temporal.py, но вместо
DINOv2 фичей кадра использует keypoints от YOLOv8-pose.

Pipeline:
  1. CropClipDataset (YOLO-bbox crop фигуриста + resize → 224×224)
  2. YOLOv8-pose → (T, 17, 3) сырые keypoints (x, y, confidence) — кэшируется на диск
  3. PoseFeatureExtractor: нормализация по midhip + torso, sin/cos углов, joint angles → (T, 61)
  4. Linear projection → temporal Transformer/TCN → 4 task-attention heads

Использование:
    pip install ultralytics
    python scripts/train_pose_temporal.py

Структуру temporal-модели и multi-task loss переиспользуем из train_dinov_2_temporal.py.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from sklearn.model_selection import train_test_split

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
    save_plots,
)
from scripts.train_videomae_phase1 import CropClipDataset, detect_skater_bboxes
from src.config import DataLoaderConfig, VideoConfig


# ============================================================
# Config
# ============================================================

DEVICE_ID = int(os.environ.get("POSE_CUDA_DEVICE", "1"))
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints_pose_temporal")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Архитектура
NUM_FRAMES = 64
IMAGE_SIZE = 224
TARGET_FPS = 25.0

# YOLOv8-pose. yolov8x-pose даёт лучшее качество кейпойнтов на small/distant людях,
# 70MB, прогоняется один раз.
POSE_MODEL = "yolov8x-pose.pt"
POSE_IMGSZ = 224  # CropClipDataset уже отдаёт 224×224 кропы, увеличение не нужно
POSE_CONF = 0.05  # низкий порог: лучше шумные keypoints чем None (фильтруется через confidence-feature)

# Pose feature extractor outputs F=96 per frame:
#   34 normalized (x,y) + 17 conf + 6 sin/cos углов линий + 4 joint cos
#   + 34 velocity Δ(x,y) + 1 cumulative rotation
POSE_FEATURE_DIM = 96

# Сглаживание keypoints по времени (Gaussian σ кадров) — снижает дрожание YOLOv8-pose.
# 0 = без сглаживания. ~1.0 кадр на 64fps клипе — мягко.
SMOOTH_SIGMA = 1.0

# Horizontal flip keypoints с правильным L↔R swap. Эффективно удваивает датасет.
USE_HFLIP_KPTS = True

# Включает crop вокруг фигуриста перед YOLO-pose. На больших ареных без crop pose
# модель часто промахивается — фигурист 50px в кадре. Crop увеличивает до ~150px.
USE_SKATER_CROP = True

# Мы кэшируем СЫРЫЕ keypoints, фичи извлекаются на лету. Менять feature engineering
# можно без перезапуска YOLO-pose (1 раз ~5 минут).
USE_KEYPOINT_CACHE = True

# Аугментации
USE_HORIZONTAL_FLIP = False  # требует swap L↔R keypoints, пока не поддержано
USE_TEMPORAL_ROLL = True
MAX_TEMPORAL_ROLL = 8

# Optimizer / training
EPOCHS = 80
BATCH_SIZE = 32  # фичи 61-dim, не 768 как у DINO — большой батч свободно влезает
LR_TEMPORAL = 5e-4
LR_HEADS = 1e-3
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.05
HEAD_DROPOUT = 0.35
USE_AMP = True

# Temporal backbone params (те же что у DINO)
TEMPORAL_BACKBONE: Literal["transformer", "tcn", "hybrid"] = "transformer"
TEMPORAL_HIDDEN_DIM = 256  # меньше чем у DINO (768), т.к. вход 61 а не 768

NUM_JUMP_CLASSES = len(LABEL_MAP)


# ============================================================
# Pose feature extraction
# ============================================================

# COCO 17 keypoints layout
KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]
KP = {n: i for i, n in enumerate(KP_NAMES)}

# Пары L↔R для зеркального flip
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
    """Гауссово сглаживание (x, y) по временной оси. Confidence не трогаем."""
    if sigma <= 0:
        return kpts
    from scipy.ndimage import gaussian_filter1d
    kpts = kpts.clone()
    xy_smoothed = gaussian_filter1d(kpts[..., :2].numpy(), sigma=sigma, axis=0, mode="nearest")
    kpts[..., :2] = torch.from_numpy(xy_smoothed)
    return kpts


def _hflip_keypoints(kpts: torch.Tensor) -> torch.Tensor:
    """Зеркало по x + swap L↔R. Координатный origin не важен — extract_pose_features
    нормализует относительно midhip."""
    kpts = kpts.clone()
    kpts[..., 0] = -kpts[..., 0]
    for L, R in KEYPOINT_LR_PAIRS:
        tmp = kpts[:, L].clone()
        kpts[:, L] = kpts[:, R]
        kpts[:, R] = tmp
    return kpts


def _line_sincos(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Угол линии p1→p2 как (sin θ, cos θ). Без wrap-around на ±π."""
    d = p2 - p1
    theta = torch.atan2(d[..., 1], d[..., 0])
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def _joint_cos(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Косинус угла в суставе b между b→a и b→c. Возвращает (..., 1)."""
    v1 = a - b
    v2 = c - b
    cos = (v1 * v2).sum(-1) / (
        torch.linalg.norm(v1, dim=-1).clamp(min=1e-3)
        * torch.linalg.norm(v2, dim=-1).clamp(min=1e-3)
    )
    return cos.unsqueeze(-1)


def extract_pose_features(kpts: torch.Tensor, smooth_sigma: float = 0.0) -> torch.Tensor:
    """
    (T, 17, 3) сырые keypoints (x, y, confidence) → (T, 96) фичи.

    Состав 96-dim вектора:
      - 34: нормализованные (x, y) для 17 keypoints (центр=midhip, scale=torso_len)
      - 17: confidence от YOLOv8-pose
      - 6:  sin/cos линий плеч, бёдер, позвоночника
      - 4:  cos углов в L/R коленях и L/R локтях
      - 34: velocity Δ(x,y) нормированных координат (т.е. движение per-frame)
      - 1:  abs cumulative rotation плечевой линии в "оборотах" с кадра 0
    """
    if smooth_sigma > 0:
        kpts = _smooth_keypoints(kpts, smooth_sigma)

    xy = kpts[..., :2]                          # (T, 17, 2)
    conf = kpts[..., 2]                          # (T, 17)

    midhip = (xy[:, KP["L_hip"]] + xy[:, KP["R_hip"]]) / 2          # (T, 2)
    midshoulder = (xy[:, KP["L_shoulder"]] + xy[:, KP["R_shoulder"]]) / 2  # (T, 2)
    torso_len = torch.linalg.norm(midshoulder - midhip, dim=-1, keepdim=True).clamp(min=1e-3)  # (T, 1)

    xy_norm = (xy - midhip.unsqueeze(1)) / torso_len.unsqueeze(1)  # (T, 17, 2)

    shoulder_sc = _line_sincos(xy[:, KP["L_shoulder"]], xy[:, KP["R_shoulder"]])  # (T, 2)
    hip_sc = _line_sincos(xy[:, KP["L_hip"]], xy[:, KP["R_hip"]])                  # (T, 2)
    spine_sc = _line_sincos(midhip, midshoulder)                                    # (T, 2)

    L_knee = _joint_cos(xy[:, KP["L_hip"]], xy[:, KP["L_knee"]], xy[:, KP["L_ankle"]])
    R_knee = _joint_cos(xy[:, KP["R_hip"]], xy[:, KP["R_knee"]], xy[:, KP["R_ankle"]])
    L_elbow = _joint_cos(xy[:, KP["L_shoulder"]], xy[:, KP["L_elbow"]], xy[:, KP["L_wrist"]])
    R_elbow = _joint_cos(xy[:, KP["R_shoulder"]], xy[:, KP["R_elbow"]], xy[:, KP["R_wrist"]])

    # Velocity Δ(x,y) кадр-в-кадр на нормализованных координатах
    xy_velocity = torch.zeros_like(xy_norm)
    xy_velocity[1:] = xy_norm[1:] - xy_norm[:-1]                                    # (T, 17, 2)

    # Cumulative rotation плечевой линии: unwrap чтобы не было прыжков на ±π,
    # затем |Δ θ от начала| в "оборотах". abs() — чтобы инвариантно к hflip.
    sh_dx = xy[:, KP["R_shoulder"], 0] - xy[:, KP["L_shoulder"], 0]
    sh_dy = xy[:, KP["R_shoulder"], 1] - xy[:, KP["L_shoulder"], 1]
    sh_angle = torch.atan2(sh_dy, sh_dx)                                            # (T,)
    sh_unwrapped = torch.from_numpy(np.unwrap(sh_angle.numpy())).float()
    cum_rotation_abs = (sh_unwrapped - sh_unwrapped[0:1]).abs() / (2 * math.pi)    # (T,) в оборотах

    feats = torch.cat([
        xy_norm.flatten(1),                # (T, 34)
        conf,                              # (T, 17)
        shoulder_sc,                       # (T, 2)
        hip_sc,                            # (T, 2)
        spine_sc,                          # (T, 2)
        L_knee, R_knee,                    # (T, 2)
        L_elbow, R_elbow,                  # (T, 2)
        xy_velocity.flatten(1),            # (T, 34) — NEW
        cum_rotation_abs.unsqueeze(-1),    # (T, 1)  — NEW
    ], dim=-1)
    return feats  # (T, 96)


# ============================================================
# Cache: YOLOv8-pose keypoints
# ============================================================

def _keypoints_dir_for_config() -> Path:
    crop_tag = "crop" if USE_SKATER_CROP else "nocrop"
    return _REPO_ROOT / "data" / f"pose_keypoints_n{NUM_FRAMES}_{crop_tag}"


def _kpt_cache_path(row, kpts_dir: Path) -> Path:
    p = Path(str(row["clip_path"]))
    return kpts_dir / f"{p.parent.name}_{p.stem}.pt"


@torch.no_grad()
def precompute_pose_keypoints(base_dataset, kpts_dir: Path, device: torch.device):
    """One-time YOLOv8-pose inference. Сохраняет (T, 17, 3) raw keypoints per clip."""
    kpts_dir.mkdir(parents=True, exist_ok=True)
    df = base_dataset.df

    missing = [i for i in range(len(df)) if not _kpt_cache_path(df.iloc[i], kpts_dir).is_file()]
    if not missing:
        print(f"Pose keypoints: all {len(df)} clips cached in {kpts_dir}")
        return

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Установи `pip install ultralytics`.") from e

    print(f"Precomputing pose keypoints: {len(missing)} clips → {kpts_dir}")
    model = YOLO(POSE_MODEL)
    model.to(device)

    for idx in tqdm(missing, desc="YOLOv8-pose"):
        row = df.iloc[idx]
        frames, _ = base_dataset[idx]                   # (T, C, H, W) [0,1]
        # YOLO ожидает uint8 (H, W, 3) RGB
        frames_np = (frames * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

        results = model(list(frames_np), verbose=False, imgsz=POSE_IMGSZ, conf=POSE_CONF, device=device)

        T = len(results)
        kpts = torch.zeros(T, 17, 3, dtype=torch.float32)
        for t, res in enumerate(results):
            if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
                continue
            data = res.keypoints.data  # (N_persons, 17, 3)
            # выбираем человека с самым большим bbox (главный фигурист)
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                idx_best = int(areas.argmax().item())
            else:
                idx_best = 0
            kpts[t] = data[idx_best].cpu().float()

        torch.save(kpts, _kpt_cache_path(row, kpts_dir))

    del model
    torch.cuda.empty_cache()
    print(f"Pose keypoints: cached to {kpts_dir}")


class KeypointsBaseDataset(Dataset):
    """Возвращает (T, 96) фичи позы. Сырые keypoints читаются с диска,
    feature extraction (smoothing → optional hflip → нормализация → углы → velocity → cum_rot)
    делается на лету в __getitem__."""

    def __init__(self, df, kpts_dir: Path, smooth_sigma: float = 0.0, augment_hflip: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.kpts_dir = kpts_dir
        self.smooth_sigma = smooth_sigma
        self.augment_hflip = augment_hflip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kpts = torch.load(_kpt_cache_path(row, self.kpts_dir), weights_only=True)  # (T, 17, 3)

        # H-flip применяется НА СЫРЫХ keypoints до feature extraction —
        # потому что extract_pose_features делает нелинейности (sin/cos углов, joint cos).
        # Просто отзеркалить готовые фичи нельзя.
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            kpts = _hflip_keypoints(kpts)

        feats = extract_pose_features(kpts, smooth_sigma=self.smooth_sigma)         # (T, 96)
        return feats, 0


# ============================================================
# Augmentation
# ============================================================

def augment_train_features(x: torch.Tensor) -> torch.Tensor:
    """temporal_roll по времени, h-flip пока не поддержан."""
    if USE_TEMPORAL_ROLL and MAX_TEMPORAL_ROLL > 0:
        shifts = torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (x.shape[0],))
        if (shifts != 0).any():
            x = x.clone()
            for i, shift in enumerate(shifts.tolist()):
                if shift != 0:
                    x[i] = x[i].roll(shifts=shift, dims=0)
    return x


# ============================================================
# Model
# ============================================================

class MultiTaskPoseTemporal(nn.Module):
    """61-dim pose-фичи → projection → temporal model → 4 task heads."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_jump_classes: int,
        num_rotation_classes: int,
        num_underrotation_classes: int,
        num_fall_classes: int,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if TEMPORAL_BACKBONE == "transformer":
            self.temporal = TemporalTransformer(dim=hidden_dim, num_frames=NUM_FRAMES)
        elif TEMPORAL_BACKBONE == "tcn":
            self.temporal = TemporalTCN(dim=hidden_dim, num_frames=NUM_FRAMES)
        elif TEMPORAL_BACKBONE == "hybrid":
            self.temporal = TemporalHybrid(dim=hidden_dim, num_frames=NUM_FRAMES)
        else:
            raise ValueError(f"Unknown TEMPORAL_BACKBONE: {TEMPORAL_BACKBONE}")

        self.jump_head = TaskAttentionHead(hidden_dim, num_jump_classes, HEAD_DROPOUT)
        self.rotation_head = TaskAttentionHead(hidden_dim, num_rotation_classes, HEAD_DROPOUT)
        self.underrotation_head = TaskAttentionHead(hidden_dim, num_underrotation_classes, HEAD_DROPOUT)
        self.fall_head = TaskAttentionHead(hidden_dim, num_fall_classes, HEAD_DROPOUT)

    def forward(self, x: torch.Tensor):
        # x: (B, T, feature_dim)
        x = self.input_proj(x)              # (B, T, hidden_dim)
        seq = self.temporal(x)              # (B, T, hidden_dim)

        j_out, j_w = self.jump_head(seq)
        r_out, r_w = self.rotation_head(seq)
        u_out, u_w = self.underrotation_head(seq)
        f_out, f_w = self.fall_head(seq)

        return {
            "jump": j_out, "rot": r_out, "under": u_out, "fall": f_out,
            "attn": {"jump": j_w, "rot": r_w, "under": u_w, "fall": f_w},
        }


# ============================================================
# Train / eval
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, num_classes: dict):
    model.train()
    total_loss, total = 0.0, 0

    for features, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        features = augment_train_features(features).to(DEVICE, non_blocking=True)
        j_lbl = j_lbl.to(DEVICE, non_blocking=True)
        r_lbl = r_lbl.to(DEVICE, non_blocking=True)
        u_lbl = u_lbl.to(DEVICE, non_blocking=True)
        f_lbl = f_lbl.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(features)
            loss = multitask_loss(outputs, (j_lbl, r_lbl, u_lbl, f_lbl), criterion, num_classes)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(j_lbl)
        total += len(j_lbl)

    return total_loss / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    preds = {k: [] for k in ("jump", "rot", "under", "fall")}
    trues = {k: [] for k in ("jump", "rot", "under", "fall")}
    attn_sum = {k: torch.zeros(NUM_FRAMES) for k in ("jump", "rot", "under", "fall")}
    attn_count = 0

    for features, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        features = features.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(features)

        preds["jump"].extend(outputs["jump"].argmax(1).cpu().numpy())
        preds["rot"].extend(outputs["rot"].argmax(1).cpu().numpy())
        preds["under"].extend(outputs["under"].argmax(1).cpu().numpy())
        preds["fall"].extend(outputs["fall"].argmax(1).cpu().numpy())

        trues["jump"].extend(j_lbl.numpy())
        trues["rot"].extend(r_lbl.numpy())
        trues["under"].extend(u_lbl.numpy())
        trues["fall"].extend(f_lbl.numpy())

        for k in attn_sum:
            attn_sum[k] += outputs["attn"][k].detach().cpu().sum(dim=0)
        attn_count += features.shape[0]

    f1s = {k: f1_score(trues[k], preds[k], average="macro", zero_division=0) for k in preds}
    avg_attn = {k: (attn_sum[k] / max(attn_count, 1)).numpy() for k in attn_sum}
    return f1s, avg_attn


# ============================================================
# Loaders (отдельные base-датасеты: train с hflip, val без)
# ============================================================

def build_loaders_pose(df_clips, kpts_dir, jump_labels, rotation_labels, underrotation_labels, fall_labels):
    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    train_base = KeypointsBaseDataset(
        df_clips, kpts_dir, smooth_sigma=SMOOTH_SIGMA, augment_hflip=USE_HFLIP_KPTS,
    )
    val_base = KeypointsBaseDataset(
        df_clips, kpts_dir, smooth_sigma=SMOOTH_SIGMA, augment_hflip=False,
    )

    train_dataset = MultiTaskDataset(train_base, jump_labels, rotation_labels, underrotation_labels, fall_labels)
    val_dataset = MultiTaskDataset(val_base, jump_labels, rotation_labels, underrotation_labels, fall_labels)

    train_jump_labels = jump_labels[train_idx]
    class_counts = np.maximum(np.bincount(train_jump_labels, minlength=NUM_JUMP_CLASSES), 1)
    sample_weights = torch.tensor((1.0 / class_counts)[train_jump_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    return train_loader, val_loader, train_idx, val_idx


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
    print(f"Pose model: {POSE_MODEL}")
    print(f"Temporal backbone: {TEMPORAL_BACKBONE}, hidden={TEMPORAL_HIDDEN_DIM}")
    print(f"Frames: {NUM_FRAMES}, batch_size={BATCH_SIZE}")
    print(f"Rotation classes: {rotation_map}")
    print(f"Underrotation classes: {underrotation_map}")
    print(f"Device: {DEVICE}")

    # CropClipDataset нужен только для precompute (отдаёт кадры для YOLOv8-pose)
    if USE_SKATER_CROP:
        bbox_cache_path = _REPO_ROOT / "data" / f"skater_bboxes_n{NUM_FRAMES}.json"
        bboxes = detect_skater_bboxes(
            df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
            device=DEVICE, cache_path=bbox_cache_path,
        )
        print(f"Skater bboxes: {len(bboxes)}/{len(df_clips)} clips")
        frame_dataset = CropClipDataset(
            df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
            image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
        )
    else:
        from scripts.clip_dataset import ClipDataset
        frame_dataset = ClipDataset(
            df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
            image_size=IMAGE_SIZE, return_meta=False,
        )

    # YOLOv8-pose precompute (один раз)
    kpts_dir = _keypoints_dir_for_config()
    precompute_pose_keypoints(frame_dataset, kpts_dir, DEVICE)

    print(f"Training on cached pose keypoints ({kpts_dir})")
    print(f"Smooth σ={SMOOTH_SIGMA}, hflip={USE_HFLIP_KPTS}, feature_dim={POSE_FEATURE_DIM}")

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    train_loader, val_loader, train_idx, val_idx = build_loaders_pose(
        df_clips, kpts_dir, jump_labels, rotation_labels, underrotation_labels, fall_labels,
    )

    model = MultiTaskPoseTemporal(
        feature_dim=POSE_FEATURE_DIM,
        hidden_dim=TEMPORAL_HIDDEN_DIM,
        num_jump_classes=NUM_JUMP_CLASSES,
        num_rotation_classes=len(rotation_map),
        num_underrotation_classes=len(underrotation_map),
        num_fall_classes=2,
    ).to(DEVICE)

    temporal_params = list(model.input_proj.parameters()) + list(model.temporal.parameters())
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

    num_classes = {
        "jump": NUM_JUMP_CLASSES, "rot": len(rotation_map),
        "under": len(underrotation_map), "fall": 2,
    }

    history = {"train_loss": [], "jump_f1": [], "rot_f1": [], "under_f1": [], "fall_f1": []}
    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
        f1s, avg_attn = eval_epoch(model, val_loader)

        history["train_loss"].append(train_loss)
        history["jump_f1"].append(f1s["jump"])
        history["rot_f1"].append(f1s["rot"])
        history["under_f1"].append(f1s["under"])
        history["fall_f1"].append(f1s["fall"])

        score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]

        print(
            f"[{epoch}/{EPOCHS}] loss={train_loss:.4f} "
            f"jump_f1={f1s['jump']:.3f} rot_f1={f1s['rot']:.3f} "
            f"under_f1={f1s['under']:.3f} fall_f1={f1s['fall']:.3f} "
            f"score={score:.3f}"
        )

        # save_plots импортирован из dino-скрипта, но он пишет в его CHECKPOINT_DIR.
        # перезаписываем его таргет на наш через monkey-patch модуля
        save_plots.__globals__["CHECKPOINT_DIR"] = CHECKPOINT_DIR
        save_plots(history, epoch, avg_attn)

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "rotation_map": rotation_map,
                    "underrotation_map": underrotation_map,
                    "label_map": LABEL_MAP,
                    "config": {
                        "pose_model": POSE_MODEL,
                        "feature_dim": POSE_FEATURE_DIM,
                        "hidden_dim": TEMPORAL_HIDDEN_DIM,
                        "num_frames": NUM_FRAMES,
                        "temporal_backbone": TEMPORAL_BACKBONE,
                    },
                    "f1s": f1s,
                    "score": score,
                },
                CHECKPOINT_DIR / "best_pose_temporal.pt",
            )
            print(f"         -> best saved (score={score:.3f})")

    print(f"Training finished. Best score={best_score:.3f}")
    print(f"Artifacts saved to: {CHECKPOINT_DIR.resolve()}")


if __name__ == "__main__":
    main()
