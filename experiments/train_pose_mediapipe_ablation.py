"""MediaPipe Pose ablation.

Заменяет двухэтапный YOLO pipeline (bbox + keypoints) одной моделью MediaPipe Pose.
33 ландмарки на кадр, включая 4 foot keypoints (heel + foot_index per side) —
больше чем 17 COCO, меньше чем 133 RTMW. Главное — простота и предсказуемость.

Configs (зеркалят ablation v3 для прямого сравнения):
  M0_baseline      — cropped, target_fps=25, padding=0.15, no bbox smooth
  M1_wider         — cropped, target_fps=16
  M2_bbox          — cropped, target_fps=25, padding=0.30, bbox smooth σ=2
  M3_combined      — cropped, target_fps=16, padding=0.30, bbox smooth σ=2 (matches B3)
  M4_no_crop       — full frame, target_fps=16 (MediaPipe alone, no YOLO crop)

Использование:
    pip install mediapipe
    python scripts/train_pose_mediapipe_ablation.py
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
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

from scripts.clip_dataset import ClipDataset, LABEL_MAP, prepare_clip_dataset
from scripts.train_dinov_2_temporal import (
    make_underrotation_map,
    map_fall,
    normalize_underrotation_value,
)
from scripts.train_pose_ablation import _smooth_keypoints
from scripts.train_pose_temporal import _joint_cos, _kpt_cache_path, _line_sincos
# Импортируем v3 для модели и тренировочной инфраструктуры
import experiments.train_pose_ablation_v3 as v3
from experiments.train_pose_ablation_v3 import (
    AblationDataset,
    NUM_FRAMES,
    NUM_JUMP_CLASSES,
    detect_and_smooth_bboxes,
    run_config as _v3_run_config,
)
from scripts.train_videomae_phase1 import CropClipDataset
from src.config import DataLoaderConfig, VideoConfig


# === override v3 module-level constants для этого скрипта ===
v3.CHECKPOINT_DIR = Path("checkpoints_pose_mediapipe")
v3.CHECKPOINT_DIR.mkdir(exist_ok=True)
v3.SEEDS = [42, 43, 44]


# ============================================================
# MediaPipe keypoints (33 landmarks)
# ============================================================

# Подмножество 21 ключевой точки полезное для фигурного катания
MP_INDICES = [
    0,       # nose
    2, 5,    # L_eye, R_eye
    7, 8,    # L_ear, R_ear
    11, 12,  # L_shoulder, R_shoulder
    13, 14,  # L_elbow, R_elbow
    15, 16,  # L_wrist, R_wrist
    23, 24,  # L_hip, R_hip
    25, 26,  # L_knee, R_knee
    27, 28,  # L_ankle, R_ankle
    29, 30,  # L_heel, R_heel
    31, 32,  # L_foot_index (big toe), R_foot_index
]

KP_MP = {name: i for i, name in enumerate([
    "nose",
    "L_eye", "R_eye",
    "L_ear", "R_ear",
    "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow",
    "L_wrist", "R_wrist",
    "L_hip", "R_hip",
    "L_knee", "R_knee",
    "L_ankle", "R_ankle",
    "L_heel", "R_heel",
    "L_big_toe", "R_big_toe",
])}


def precompute_mediapipe_keypoints(frame_dataset, kpts_dir: Path):
    """Прогоняет MediaPipe Pose по кадрам. Сохраняет (T, 21, 3) keypoints
    (выбранные body+foot landmarks из MP 33)."""
    try:
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError("Установи: pip install mediapipe") from e

    kpts_dir.mkdir(parents=True, exist_ok=True)
    df = frame_dataset.df
    missing = [i for i in range(len(df)) if not _kpt_cache_path(df.iloc[i], kpts_dir).is_file()]
    if not missing:
        print(f"MediaPipe keypoints: all {len(df)} clips cached in {kpts_dir}")
        return

    print(f"Precomputing MediaPipe keypoints: {len(missing)} clips → {kpts_dir}")
    pose_estimator = mp.solutions.pose.Pose(
        static_image_mode=False,         # tracking режим — быстрее
        model_complexity=1,               # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    for idx in tqdm(missing, desc="MediaPipe"):
        row = df.iloc[idx]
        frames, _ = frame_dataset[idx]                          # (T, C, H, W) [0,1]
        frames_np = (frames * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()  # (T, H, W, 3) RGB

        T = len(frames_np)
        kpts = torch.zeros(T, len(MP_INDICES), 3, dtype=torch.float32)
        h, w = frames_np.shape[1], frames_np.shape[2]
        for t, frame in enumerate(frames_np):
            results = pose_estimator.process(frame)
            if results.pose_landmarks is None:
                continue
            landmarks = results.pose_landmarks.landmark
            for out_i, mp_i in enumerate(MP_INDICES):
                lm = landmarks[mp_i]
                kpts[t, out_i, 0] = lm.x * w   # пиксельные координаты
                kpts[t, out_i, 1] = lm.y * h
                kpts[t, out_i, 2] = lm.visibility

        torch.save(kpts, _kpt_cache_path(row, kpts_dir))

    pose_estimator.close()


# ============================================================
# MediaPipe feature extraction (21 keypoints включая foot)
# ============================================================

def extract_features_mediapipe(kpts: torch.Tensor, smooth_sigma: float = 0.0) -> torch.Tensor:
    """MediaPipe 21 keypoints → 122-dim features (с foot info)."""
    if smooth_sigma > 0:
        kpts = _smooth_keypoints(kpts, smooth_sigma)

    xy = kpts[..., :2]     # (T, 21, 2)
    conf = kpts[..., 2]    # (T, 21)

    midhip = (xy[:, KP_MP["L_hip"]] + xy[:, KP_MP["R_hip"]]) / 2
    midshoulder = (xy[:, KP_MP["L_shoulder"]] + xy[:, KP_MP["R_shoulder"]]) / 2
    torso_len = torch.linalg.norm(midshoulder - midhip, dim=-1, keepdim=True).clamp(min=1e-3)
    xy_norm = (xy - midhip.unsqueeze(1)) / torso_len.unsqueeze(1)

    shoulder_sc = _line_sincos(xy[:, KP_MP["L_shoulder"]], xy[:, KP_MP["R_shoulder"]])
    hip_sc = _line_sincos(xy[:, KP_MP["L_hip"]], xy[:, KP_MP["R_hip"]])
    spine_sc = _line_sincos(midhip, midshoulder)
    L_knee = _joint_cos(xy[:, KP_MP["L_hip"]], xy[:, KP_MP["L_knee"]], xy[:, KP_MP["L_ankle"]])
    R_knee = _joint_cos(xy[:, KP_MP["R_hip"]], xy[:, KP_MP["R_knee"]], xy[:, KP_MP["R_ankle"]])
    L_elbow = _joint_cos(xy[:, KP_MP["L_shoulder"]], xy[:, KP_MP["L_elbow"]], xy[:, KP_MP["L_wrist"]])
    R_elbow = _joint_cos(xy[:, KP_MP["R_shoulder"]], xy[:, KP_MP["R_elbow"]], xy[:, KP_MP["R_wrist"]])

    # Foot orientation: heel → big_toe (направление носка)
    L_foot_sc = _line_sincos(xy[:, KP_MP["L_heel"]], xy[:, KP_MP["L_big_toe"]])
    R_foot_sc = _line_sincos(xy[:, KP_MP["R_heel"]], xy[:, KP_MP["R_big_toe"]])
    L_foot_len = torch.linalg.norm(xy[:, KP_MP["L_big_toe"]] - xy[:, KP_MP["L_heel"]], dim=-1, keepdim=True) / torso_len
    R_foot_len = torch.linalg.norm(xy[:, KP_MP["R_big_toe"]] - xy[:, KP_MP["R_heel"]], dim=-1, keepdim=True) / torso_len

    xy_velocity = torch.zeros_like(xy_norm)
    xy_velocity[1:] = xy_norm[1:] - xy_norm[:-1]

    sh_dx = xy[:, KP_MP["R_shoulder"], 0] - xy[:, KP_MP["L_shoulder"], 0]
    sh_dy = xy[:, KP_MP["R_shoulder"], 1] - xy[:, KP_MP["L_shoulder"], 1]
    sh_angle = torch.atan2(sh_dy, sh_dx)
    sh_unwrapped = torch.from_numpy(np.unwrap(sh_angle.numpy())).float()
    cum_rot = (sh_unwrapped - sh_unwrapped[0:1]).abs() / (2 * math.pi)

    return torch.cat([
        xy_norm.flatten(1),                # 21*2 = 42
        conf,                              # 21
        shoulder_sc, hip_sc, spine_sc,     # 6
        L_knee, R_knee, L_elbow, R_elbow,  # 4
        L_foot_sc, R_foot_sc,              # 4
        L_foot_len, R_foot_len,            # 2
        xy_velocity.flatten(1),            # 42
        cum_rot.unsqueeze(-1),             # 1
    ], dim=-1)  # 122


# ============================================================
# Configs
# ============================================================

CONFIGS = [
    {
        "name": "M0_baseline",
        "target_fps": 25.0, "bbox_padding": 0.15, "bbox_smooth_sigma": 0.0,
        "use_crop": True,
    },
    {
        "name": "M1_wider",
        "target_fps": 16.0, "bbox_padding": 0.15, "bbox_smooth_sigma": 0.0,
        "use_crop": True,
    },
    {
        "name": "M2_bbox",
        "target_fps": 25.0, "bbox_padding": 0.30, "bbox_smooth_sigma": 2.0,
        "use_crop": True,
    },
    {
        "name": "M3_combined",
        "target_fps": 16.0, "bbox_padding": 0.30, "bbox_smooth_sigma": 2.0,
        "use_crop": True,
    },
    {
        "name": "M4_no_crop",
        "target_fps": 16.0, "bbox_padding": 0.0, "bbox_smooth_sigma": 0.0,
        "use_crop": False,
    },
]


def _mp_pose_cache_dir(cfg) -> Path:
    crop = "crop" if cfg["use_crop"] else "fullframe"
    return _REPO_ROOT / "data" / (
        f"pose_kpts_mediapipe_n{NUM_FRAMES}_fps{int(cfg['target_fps'])}"
        f"_pad{int(cfg['bbox_padding'] * 100):03d}"
        f"_bsm{int(cfg['bbox_smooth_sigma'] * 10):02d}_{crop}"
    )


def run_mp_config(cfg, df_clips, jump_labels, rotation_labels, rotation_floats,
                  underrotation_labels, fall_labels, num_classes):
    name = cfg["name"]
    print(f"\n{'#' * 70}")
    print(f"Config: {name}")
    print(f"  target_fps={cfg['target_fps']}  use_crop={cfg['use_crop']}  "
          f"bbox_pad={cfg['bbox_padding']}  bbox_smooth_σ={cfg['bbox_smooth_sigma']}")
    print('#' * 70)

    # === Шаг 1: frame_dataset (с crop или без) ===
    if cfg["use_crop"]:
        try:
            bboxes = detect_and_smooth_bboxes(df_clips, cfg)
        except Exception as e:
            print(f"  ✗ bbox detection failed: {e}")
            return None
        print(f"  Skater bboxes: {len(bboxes)}/{len(df_clips)}")
        frame_dataset = CropClipDataset(
            df=df_clips, num_frames=NUM_FRAMES, target_fps=cfg["target_fps"],
            image_size=v3.IMAGE_SIZE, return_meta=False, bboxes=bboxes,
        )
    else:
        frame_dataset = ClipDataset(
            df=df_clips, num_frames=NUM_FRAMES, target_fps=cfg["target_fps"],
            image_size=v3.IMAGE_SIZE, return_meta=False,
        )

    # === Шаг 2: MediaPipe precompute ===
    kpts_dir = _mp_pose_cache_dir(cfg)
    try:
        precompute_mediapipe_keypoints(frame_dataset, kpts_dir)
    except RuntimeError as e:
        print(f"  ✗ MediaPipe extraction failed: {e}")
        return None

    # === Шаг 3: тренировка (используем общую инфраструктуру v3) ===
    # Подменяем feature_dim_for и feature extractor через monkey-patch
    feature_dim = 122
    print(f"  feature_dim={feature_dim}")

    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    seed_results = []
    seed_logits = {}
    val_trues_capture = None
    cfg_dir = v3.CHECKPOINT_DIR / name
    cfg_dir.mkdir(exist_ok=True)

    from sklearn.metrics import f1_score
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup

    for seed in v3.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = AblationDataset(
            df_clips, kpts_dir, jump_labels, rotation_labels, rotation_floats,
            underrotation_labels, fall_labels,
            extract_features_mediapipe, cfg["target_fps"], v3.SMOOTH_SIGMA,
            augment_temporal_roll=True,
        )
        val_dataset = AblationDataset(
            df_clips, kpts_dir, jump_labels, rotation_labels, rotation_floats,
            underrotation_labels, fall_labels,
            extract_features_mediapipe, cfg["target_fps"], v3.SMOOTH_SIGMA,
            augment_temporal_roll=False,
        )

        train_jl = jump_labels[train_idx]
        counts = np.maximum(np.bincount(train_jl, minlength=NUM_JUMP_CLASSES), 1)
        weights = torch.tensor((1.0 / counts)[train_jl], dtype=torch.float)
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(
            Subset(train_dataset, train_idx), batch_size=v3.BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        )
        val_loader = DataLoader(
            Subset(val_dataset, val_idx), batch_size=v3.BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        )

        model = v3.FullPoseModel(
            feature_dim=feature_dim, hidden_dim=v3.TEMPORAL_HIDDEN_DIM, num_classes=num_classes,
        ).to(v3.DEVICE)

        temporal_params = list(model.input_proj.parameters()) + list(model.temporal.parameters())
        head_params = [p for n, p in model.named_parameters()
                       if not (n.startswith("input_proj") or n.startswith("temporal."))]

        optimizer = torch.optim.AdamW([
            {"params": temporal_params, "lr": v3.LR_TEMPORAL, "weight_decay": v3.WEIGHT_DECAY},
            {"params": head_params, "lr": v3.LR_HEADS, "weight_decay": v3.WEIGHT_DECAY},
        ])
        total_steps = v3.EPOCHS * len(train_loader)
        warmup_steps = v3.WARMUP_EPOCHS * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=v3.LABEL_SMOOTHING)
        scaler = torch.cuda.amp.GradScaler(enabled=v3.USE_AMP and v3.DEVICE.type == "cuda")

        best = {"score": -1.0, "epoch": 0, "f1s": None}
        seed_dir = cfg_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        for epoch in range(1, v3.EPOCHS + 1):
            loss = v3.train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
            f1s = v3.eval_epoch(model, val_loader)
            score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "f1s": dict(f1s)}
                torch.save({"model_state_dict": model.state_dict(), "f1s": f1s, "score": score}, seed_dir / "best.pt")

        ckpt = torch.load(seed_dir / "best.pt", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        f1s_best, val_logits, val_trues = v3.eval_epoch(model, val_loader, return_logits=True)
        seed_logits[seed] = val_logits
        val_trues_capture = val_trues

        seed_results.append({
            "seed": seed, "best_epoch": best["epoch"], "score": best["score"],
            **{f"{k}_f1": best["f1s"][k] for k in ("jump", "rot", "under", "fall")},
        })
        print(f"  seed {seed}: jump={best['f1s']['jump']:.3f}  rot={best['f1s']['rot']:.3f}  "
              f"under={best['f1s']['under']:.3f}  fall={best['f1s']['fall']:.3f}  score={best['score']:.3f}")

        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()

    tasks = ("jump", "rot", "under", "fall")
    avg_logits = {k: torch.stack([seed_logits[s][k] for s in v3.SEEDS]).mean(dim=0) for k in tasks}
    ens_f1s = {k: f1_score(val_trues_capture[k], avg_logits[k].argmax(1).numpy(),
                            average="macro", zero_division=0) for k in tasks}
    ens_score = 0.45 * ens_f1s["jump"] + 0.35 * ens_f1s["rot"] + 0.15 * ens_f1s["under"] + 0.05 * ens_f1s["fall"]

    df_seeds = pd.DataFrame(seed_results)
    df_seeds.to_csv(cfg_dir / "per_seed.csv", index=False)
    print(f"  ► ENSEMBLE: jump={ens_f1s['jump']:.3f}  rot={ens_f1s['rot']:.3f}  "
          f"under={ens_f1s['under']:.3f}  fall={ens_f1s['fall']:.3f}  score={ens_score:.3f}")

    return {
        "config": name,
        "feature_dim": feature_dim,
        "use_crop": cfg["use_crop"],
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

    print(f"Dataset: {len(df_clips)} jumps, Device: {v3.DEVICE}")
    print(f"Configs: {[c['name'] for c in CONFIGS]}, Seeds: {v3.SEEDS}")

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
        result = run_mp_config(
            cfg, df_clips, jump_labels, rotation_labels, rotation_floats,
            underrotation_labels, fall_labels, num_classes,
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(v3.CHECKPOINT_DIR / "mediapipe_ablation_results.csv", index=False)

    print(f"\n\n{'=' * 95}")
    print("MEDIAPIPE ABLATION RESULTS")
    print('=' * 95)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print(f"\nReference (ablation v3, yolov8):")
    print(f"  B0:  score=0.758  jump=0.780  rot=0.674  under=0.843  fall=0.898")
    print(f"  B3:  score=0.847  jump=0.911  rot=0.725  under=0.896  fall=0.973")
    print(f"\nCSV: {v3.CHECKPOINT_DIR / 'mediapipe_ablation_results.csv'}")


if __name__ == "__main__":
    main()
