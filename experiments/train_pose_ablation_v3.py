"""Ablation v3: preprocessing/data quality improvements над F4 + aux + multi-seed.

Все конфиги используют ту же модель что в train_pose_final.py (F4 features +
rot_regression + phase_aux + multi-seed ensemble). Меняется только preprocessing:

  B0_baseline    — текущий best (TARGET_FPS=25, padding=0.15, no bbox smooth, yolov8-pose)
  B1_wider       — TARGET_FPS=16 (≈4 сек контекста вместо 2.56)
  B2_bbox        — bbox padding=0.30 + bbox smoothing σ=2 кадра
  B3_combined    — B1 + B2
  B4_rtmw        — RTMW whole-body pose (133 keypoints, foot keypoints включены)

Для каждого конфига: 3 seeds → ensemble → одна строка в результаты.
Кэши bbox/pose в отдельных папках per-config — переиспользуются между запусками.

Установка для B4_rtmw:
    pip install rtmlib onnxruntime-gpu
(B4 пропускается gracefully если rtmlib не установлен.)

Использование:
    python scripts/train_pose_ablation_v3.py
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
from tqdm import tqdm
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

CHECKPOINT_DIR = Path("checkpoints_pose_ablation_v3")
CHECKPOINT_DIR.mkdir(exist_ok=True)

NUM_FRAMES = 64
IMAGE_SIZE = 224
EPOCHS = 50  # чуть меньше для скорости (5 конфигов × 3 seeds × 50 эпох ≈ 1ч)
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

SMOOTH_SIGMA = 1.0  # keypoints smoothing (не bbox)
LAMBDA_ROT_REG = 0.3
LAMBDA_PHASE = 0.1
NUM_PHASES = 5
PHASE_HALF_WIDTH = 0.15

SEEDS = [42, 43, 44]  # 3 seeds для скорости
NUM_JUMP_CLASSES = len(LABEL_MAP)

# RTMW использует 23 keypoints (17 body + 6 foot из COCO-WholeBody)
KP_RTMW = {
    **{name: i for i, name in enumerate([
        "nose", "L_eye", "R_eye", "L_ear", "R_ear",
        "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
        "L_wrist", "R_wrist", "L_hip", "R_hip",
        "L_knee", "R_knee", "L_ankle", "R_ankle",
    ])},
    "L_big_toe": 17, "L_small_toe": 18, "L_heel": 19,
    "R_big_toe": 20, "R_small_toe": 21, "R_heel": 22,
}


CONFIGS = [
    {
        "name": "B0_baseline",
        "target_fps": 25.0, "bbox_padding": 0.15, "bbox_smooth_sigma": 0.0,
        "pose_model": "yolov8",
    },
    {
        "name": "B1_wider_window",
        "target_fps": 16.0, "bbox_padding": 0.15, "bbox_smooth_sigma": 0.0,
        "pose_model": "yolov8",
    },
    {
        "name": "B2_bbox_smooth_pad",
        "target_fps": 25.0, "bbox_padding": 0.30, "bbox_smooth_sigma": 2.0,
        "pose_model": "yolov8",
    },
    {
        "name": "B3_combined",
        "target_fps": 16.0, "bbox_padding": 0.30, "bbox_smooth_sigma": 2.0,
        "pose_model": "yolov8",
    },
    {
        "name": "B4_rtmw",
        "target_fps": 25.0, "bbox_padding": 0.30, "bbox_smooth_sigma": 0.0,
        "pose_model": "rtmw",
    },
]


# ============================================================
# Cache paths
# ============================================================

def _bbox_cache_path(cfg) -> Path:
    return _REPO_ROOT / "data" / (
        f"skater_bboxes_n{NUM_FRAMES}_fps{int(cfg['target_fps'])}"
        f"_pad{int(cfg['bbox_padding'] * 100):03d}"
        f"_bsm{int(cfg['bbox_smooth_sigma'] * 10):02d}.json"
    )


def _pose_cache_dir(cfg) -> Path:
    return _REPO_ROOT / "data" / (
        f"pose_kpts_{cfg['pose_model']}_n{NUM_FRAMES}_fps{int(cfg['target_fps'])}"
        f"_pad{int(cfg['bbox_padding'] * 100):03d}"
        f"_bsm{int(cfg['bbox_smooth_sigma'] * 10):02d}"
    )


# ============================================================
# Bbox detection с smoothing
# ============================================================

def detect_and_smooth_bboxes(df_clips, cfg) -> dict:
    """Wrapper над detect_skater_bboxes с post-processing temporal smoothing."""
    cache_path = _bbox_cache_path(cfg)
    bboxes = detect_skater_bboxes(
        df_clips,
        num_frames=NUM_FRAMES,
        target_fps=cfg["target_fps"],
        device=DEVICE,
        cache_path=cache_path,
        padding=cfg["bbox_padding"],
    )

    if cfg["bbox_smooth_sigma"] > 0:
        # Сглаживание по времени, in-place в kept dict (не пишем в кэш — он содержит несглаженный)
        from scipy.ndimage import gaussian_filter1d
        smoothed = {}
        for k, v in bboxes.items():
            arr = np.array(v, dtype=np.float32)  # (T, 4)
            arr_s = gaussian_filter1d(arr, sigma=cfg["bbox_smooth_sigma"], axis=0, mode="nearest")
            smoothed[k] = arr_s.tolist()
        return smoothed
    return bboxes


# ============================================================
# RTMW pose precompute (rtmlib)
# ============================================================

def precompute_rtmw_keypoints(base_dataset, kpts_dir: Path, device: torch.device):
    """Использует rtmlib.Wholebody. Сохраняет (T, 23, 3) — body + 6 foot keypoints."""
    try:
        from rtmlib import Wholebody
    except ImportError as e:
        raise RuntimeError(
            "RTMW требует rtmlib. Установка:\n"
            "  pip install rtmlib onnxruntime-gpu\n"
            "Или закомментируй B4_rtmw в CONFIGS."
        ) from e

    kpts_dir.mkdir(parents=True, exist_ok=True)
    df = base_dataset.df

    missing = [i for i in range(len(df)) if not _kpt_cache_path(df.iloc[i], kpts_dir).is_file()]
    if not missing:
        print(f"RTMW keypoints: all {len(df)} clips cached in {kpts_dir}")
        return

    print(f"Precomputing RTMW keypoints: {len(missing)} clips → {kpts_dir}")
    device_str = "cuda" if device.type == "cuda" else "cpu"
    model = Wholebody(to_openpose=False, mode="balanced", backend="onnxruntime", device=device_str)

    for idx in tqdm(missing, desc="RTMW"):
        row = df.iloc[idx]
        frames, _ = base_dataset[idx]                       # (T, C, H, W) [0,1]
        frames_np = (frames * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

        T = len(frames_np)
        kpts = torch.zeros(T, 23, 3, dtype=torch.float32)   # 17 body + 6 foot
        for t, frame in enumerate(frames_np):
            keypoints, scores = model(frame)  # (n_persons, 133, 2), (n_persons, 133)
            if keypoints is None or len(keypoints) == 0:
                continue
            # take person with largest "spread" (assume primary skater)
            xs = keypoints[..., 0]
            ys = keypoints[..., 1]
            spreads = (xs.max(1) - xs.min(1)) * (ys.max(1) - ys.min(1))
            best = int(np.argmax(spreads))
            kpts[t, :, :2] = torch.from_numpy(keypoints[best, :23].astype(np.float32))
            kpts[t, :, 2] = torch.from_numpy(scores[best, :23].astype(np.float32))

        torch.save(kpts, _kpt_cache_path(row, kpts_dir))

    del model


# ============================================================
# Phase pseudo-labels (parametric on target_fps)
# ============================================================

def compute_frame_timestamps(start_sec, end_sec, num_frames, target_fps):
    segment = max(end_sec - start_sec, 1e-6)
    target = num_frames / target_fps
    if segment >= target:
        ws, we = start_sec, end_sec
    else:
        c = (start_sec + end_sec) / 2.0
        h = target / 2.0
        ws = min(c - h, start_sec)
        we = max(c + h, end_sec)
    return np.linspace(ws, we, num=num_frames, endpoint=True)


def compute_phase_labels_for(start_sec_in_clip, end_sec_in_clip, num_frames, target_fps):
    timestamps = compute_frame_timestamps(start_sec_in_clip, end_sec_in_clip, num_frames, target_fps)
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
# Feature extraction
# ============================================================

def extract_features_yolov8(kpts: torch.Tensor, smooth_sigma: float = 0.0) -> torch.Tensor:
    """F4 features: 96-dim (61 + 34 vel + 1 cum_rot)."""
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


def extract_features_rtmw(kpts: torch.Tensor, smooth_sigma: float = 0.0) -> torch.Tensor:
    """RTMW features: 132-dim (23 keypoints включая foot).
    Поверх F4 добавлены: foot_angle (sin/cos heel→big_toe для L/R) и foot_length/torso."""
    if smooth_sigma > 0:
        kpts = _smooth_keypoints(kpts, smooth_sigma)
    KR = KP_RTMW
    xy = kpts[..., :2]   # (T, 23, 2)
    conf = kpts[..., 2]  # (T, 23)

    midhip = (xy[:, KR["L_hip"]] + xy[:, KR["R_hip"]]) / 2
    midshoulder = (xy[:, KR["L_shoulder"]] + xy[:, KR["R_shoulder"]]) / 2
    torso_len = torch.linalg.norm(midshoulder - midhip, dim=-1, keepdim=True).clamp(min=1e-3)
    xy_norm = (xy - midhip.unsqueeze(1)) / torso_len.unsqueeze(1)

    shoulder_sc = _line_sincos(xy[:, KR["L_shoulder"]], xy[:, KR["R_shoulder"]])
    hip_sc = _line_sincos(xy[:, KR["L_hip"]], xy[:, KR["R_hip"]])
    spine_sc = _line_sincos(midhip, midshoulder)
    L_knee = _joint_cos(xy[:, KR["L_hip"]], xy[:, KR["L_knee"]], xy[:, KR["L_ankle"]])
    R_knee = _joint_cos(xy[:, KR["R_hip"]], xy[:, KR["R_knee"]], xy[:, KR["R_ankle"]])
    L_elbow = _joint_cos(xy[:, KR["L_shoulder"]], xy[:, KR["L_elbow"]], xy[:, KR["L_wrist"]])
    R_elbow = _joint_cos(xy[:, KR["R_shoulder"]], xy[:, KR["R_elbow"]], xy[:, KR["R_wrist"]])

    # Foot orientation: heel → big_toe (направление носка)
    L_foot_sc = _line_sincos(xy[:, KR["L_heel"]], xy[:, KR["L_big_toe"]])
    R_foot_sc = _line_sincos(xy[:, KR["R_heel"]], xy[:, KR["R_big_toe"]])
    # Foot length normalized by torso
    L_foot_len = torch.linalg.norm(xy[:, KR["L_big_toe"]] - xy[:, KR["L_heel"]], dim=-1, keepdim=True) / torso_len
    R_foot_len = torch.linalg.norm(xy[:, KR["R_big_toe"]] - xy[:, KR["R_heel"]], dim=-1, keepdim=True) / torso_len

    xy_velocity = torch.zeros_like(xy_norm)
    xy_velocity[1:] = xy_norm[1:] - xy_norm[:-1]

    sh_dx = xy[:, KR["R_shoulder"], 0] - xy[:, KR["L_shoulder"], 0]
    sh_dy = xy[:, KR["R_shoulder"], 1] - xy[:, KR["L_shoulder"], 1]
    sh_angle = torch.atan2(sh_dy, sh_dx)
    sh_unwrapped = torch.from_numpy(np.unwrap(sh_angle.numpy())).float()
    cum_rot = (sh_unwrapped - sh_unwrapped[0:1]).abs() / (2 * math.pi)

    return torch.cat([
        xy_norm.flatten(1),                                # 46
        conf,                                              # 23
        shoulder_sc, hip_sc, spine_sc,                     # 6
        L_knee, R_knee, L_elbow, R_elbow,                  # 4
        L_foot_sc, R_foot_sc,                              # 4
        L_foot_len, R_foot_len,                            # 2
        xy_velocity.flatten(1),                            # 46
        cum_rot.unsqueeze(-1),                             # 1
    ], dim=-1)  # 132


def feature_dim_for(pose_model: str) -> int:
    return 96 if pose_model == "yolov8" else 132


# ============================================================
# Dataset (parametric on extractor)
# ============================================================

class AblationDataset(Dataset):
    def __init__(
        self, df, kpts_dir, jump_labels, rotation_labels, rotation_floats,
        underrotation_labels, fall_labels,
        feature_extractor, target_fps, smooth_sigma=0.0, augment_temporal_roll=False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.kpts_dir = kpts_dir
        self.feature_extractor = feature_extractor
        self.target_fps = target_fps
        self.smooth_sigma = smooth_sigma
        self.augment_temporal_roll = augment_temporal_roll

        self.jump = torch.tensor(jump_labels, dtype=torch.long)
        self.rotation = torch.tensor(rotation_labels, dtype=torch.long)
        self.rotation_float = torch.tensor(rotation_floats, dtype=torch.float)
        self.underrotation = torch.tensor(underrotation_labels, dtype=torch.long)
        self.fall = torch.tensor(fall_labels, dtype=torch.long)

        phase_list = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            phase_list.append(compute_phase_labels_for(
                float(row["start_sec_in_clip"]), float(row["end_sec_in_clip"]),
                NUM_FRAMES, target_fps,
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

        feats = self.feature_extractor(kpts, smooth_sigma=self.smooth_sigma)
        return (
            feats, self.jump[idx], self.rotation[idx], self.rotation_float[idx],
            self.underrotation[idx], self.fall[idx], phase,
        )


# ============================================================
# Model
# ============================================================

class FullPoseModel(nn.Module):
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
# Run one config
# ============================================================

def run_config(cfg, df_clips, jump_labels, rotation_labels, rotation_floats, underrotation_labels, fall_labels, num_classes):
    name = cfg["name"]
    print(f"\n{'#' * 70}")
    print(f"Config: {name}")
    print(f"  target_fps={cfg['target_fps']}  bbox_padding={cfg['bbox_padding']}  bbox_smooth_σ={cfg['bbox_smooth_sigma']}  pose={cfg['pose_model']}")
    print('#' * 70)

    # === Шаг 1: bbox cache ===
    try:
        bboxes = detect_and_smooth_bboxes(df_clips, cfg)
    except Exception as e:
        print(f"  ✗ bbox detection failed: {e}")
        return None
    print(f"  Skater bboxes: {len(bboxes)}/{len(df_clips)}")

    # === Шаг 2: pose extraction ===
    frame_dataset = CropClipDataset(
        df=df_clips, num_frames=NUM_FRAMES, target_fps=cfg["target_fps"],
        image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
    )
    kpts_dir = _pose_cache_dir(cfg)
    try:
        if cfg["pose_model"] == "yolov8":
            precompute_pose_keypoints(frame_dataset, kpts_dir, DEVICE)
            feature_extractor = extract_features_yolov8
        elif cfg["pose_model"] == "rtmw":
            precompute_rtmw_keypoints(frame_dataset, kpts_dir, DEVICE)
            feature_extractor = extract_features_rtmw
        else:
            raise ValueError(cfg["pose_model"])
    except RuntimeError as e:
        print(f"  ✗ pose extraction failed: {e}")
        return None

    feature_dim = feature_dim_for(cfg["pose_model"])
    print(f"  feature_dim={feature_dim}")

    # === Шаг 3: split + multi-seed train ===
    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    seed_results = []
    seed_logits = {}
    val_trues_capture = None
    cfg_dir = CHECKPOINT_DIR / name
    cfg_dir.mkdir(exist_ok=True)

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = AblationDataset(
            df_clips, kpts_dir, jump_labels, rotation_labels, rotation_floats,
            underrotation_labels, fall_labels,
            feature_extractor, cfg["target_fps"], SMOOTH_SIGMA, augment_temporal_roll=True,
        )
        val_dataset = AblationDataset(
            df_clips, kpts_dir, jump_labels, rotation_labels, rotation_floats,
            underrotation_labels, fall_labels,
            feature_extractor, cfg["target_fps"], SMOOTH_SIGMA, augment_temporal_roll=False,
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

        model = FullPoseModel(
            feature_dim=feature_dim, hidden_dim=TEMPORAL_HIDDEN_DIM, num_classes=num_classes,
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

        best = {"score": -1.0, "epoch": 0, "f1s": None}
        seed_dir = cfg_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
            f1s = eval_epoch(model, val_loader)
            score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "f1s": dict(f1s)}
                torch.save({"model_state_dict": model.state_dict(), "f1s": f1s, "score": score}, seed_dir / "best.pt")

        # подгружаем best и собираем логиты
        ckpt = torch.load(seed_dir / "best.pt", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        f1s_best, val_logits, val_trues = eval_epoch(model, val_loader, return_logits=True)
        seed_logits[seed] = val_logits
        val_trues_capture = val_trues

        seed_results.append({
            "seed": seed, "best_epoch": best["epoch"], "score": best["score"],
            **{f"{k}_f1": best["f1s"][k] for k in ("jump", "rot", "under", "fall")},
        })
        print(f"  seed {seed}: jump={best['f1s']['jump']:.3f}  rot={best['f1s']['rot']:.3f}  under={best['f1s']['under']:.3f}  fall={best['f1s']['fall']:.3f}  score={best['score']:.3f}")

        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()

    # === Шаг 4: ensemble ===
    tasks = ("jump", "rot", "under", "fall")
    avg_logits = {k: torch.stack([seed_logits[s][k] for s in SEEDS]).mean(dim=0) for k in tasks}
    ens_f1s = {k: f1_score(val_trues_capture[k], avg_logits[k].argmax(1).numpy(), average="macro", zero_division=0) for k in tasks}
    ens_score = 0.45 * ens_f1s["jump"] + 0.35 * ens_f1s["rot"] + 0.15 * ens_f1s["under"] + 0.05 * ens_f1s["fall"]

    df_seeds = pd.DataFrame(seed_results)
    df_seeds.to_csv(cfg_dir / "per_seed.csv", index=False)
    print(f"  ► ENSEMBLE: jump={ens_f1s['jump']:.3f}  rot={ens_f1s['rot']:.3f}  under={ens_f1s['under']:.3f}  fall={ens_f1s['fall']:.3f}  score={ens_score:.3f}")

    return {
        "config": name,
        "feature_dim": feature_dim,
        "pose_model": cfg["pose_model"],
        "target_fps": cfg["target_fps"],
        "bbox_padding": cfg["bbox_padding"],
        "bbox_smooth_sigma": cfg["bbox_smooth_sigma"],
        "mean_score": round(df_seeds["score"].mean(), 4),
        "std_score": round(df_seeds["score"].std(), 4),
        "ens_score": round(ens_score, 4),
        "ens_jump_f1": round(ens_f1s["jump"], 4),
        "ens_rot_f1": round(ens_f1s["rot"], 4),
        "ens_under_f1": round(ens_f1s["under"], 4),
        "ens_fall_f1": round(ens_f1s["fall"], 4),
    }


# ============================================================
# Main
# ============================================================

def main():
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

    print(f"Dataset: {len(df_clips)} jumps, Device: {DEVICE}")
    print(f"Configs: {[c['name'] for c in CONFIGS]}, Seeds: {SEEDS}")

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    rotation_floats = np.array([float(inv_rotation_map[c]) for c in rotation_labels], dtype=np.float32)
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    num_classes = {
        "jump": NUM_JUMP_CLASSES, "rot": len(rotation_map),
        "under": len(underrotation_map), "fall": 2,
    }

    results = []
    for cfg in CONFIGS:
        result = run_config(
            cfg, df_clips, jump_labels, rotation_labels, rotation_floats,
            underrotation_labels, fall_labels, num_classes,
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(CHECKPOINT_DIR / "ablation_v3_results.csv", index=False)

    # === Финальная сводка ===
    print(f"\n\n{'=' * 95}")
    print("ABLATION V3 RESULTS")
    print('=' * 95)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print(f"\nCSV: {CHECKPOINT_DIR / 'ablation_v3_results.csv'}")


if __name__ == "__main__":
    main()
