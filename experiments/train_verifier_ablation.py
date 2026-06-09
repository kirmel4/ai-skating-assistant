"""Ablation для verifier'а — что выжимает максимум из pose-only.

Работает на ГОТОВОМ pose-кэше (data/verifier_pose/) — YOLO заново не гоняется.
Каждый конфиг: 3 seeds → ensemble логитов. В конце — сводная таблица.

Configs:
  A_hybrid_full        — baseline: TCN+Transformer, полный F4 (96)
  B_tcn_full           — только TCN
  C_transformer_full   — только Transformer
  D_hybrid_no_velocity — hybrid, фичи без velocity (62)
  E_hybrid_no_cumrot   — hybrid, фичи без cum_rotation (95)

Покажет: какой backbone лучше, нужны ли velocity / cum_rotation.

Требует заранее запущенного train_verifier.py (создаёт verifier_windows_v2.json
и pose-кэш).

Использование:
    python scripts/train_verifier_ablation.py
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from scripts.train_dinov_2_temporal import (
    TaskAttentionHead,
    TemporalHybrid,
    TemporalTCN,
    TemporalTransformer,
)
from scripts.train_pose_ablation import (
    _hflip_keypoints,
    extract_pose_features,
    feature_dim_for_cfg,
)
from scripts.train_verifier import _window_cache_path, build_windows


# ============================================================
# Config
# ============================================================

DEVICE_ID = int(os.environ.get("POSE_CUDA_DEVICE", "1"))


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if count == 1:
        return torch.device("cuda:0")
    idx = DEVICE_ID if 0 <= DEVICE_ID < count else 0
    return torch.device(f"cuda:{idx}")


DEVICE = _resolve_device()
CHECKPOINT_DIR = _REPO_ROOT / "checkpoints_verifier_ablation"
CHECKPOINT_DIR.mkdir(exist_ok=True)

NUM_FRAMES = 64
SMOOTH_SIGMA = 1.0
TEMPORAL_HIDDEN_DIM = 128
EPOCHS = 50
BATCH_SIZE = 32
LR = 7e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.15
HEAD_DROPOUT = 0.5
INPUT_DROPOUT = 0.2
MAX_TEMPORAL_ROLL = 8
USE_HFLIP = True
SEEDS = [42, 43, 44]


CONFIGS = [
    {"name": "A_hybrid_full",        "backbone": "hybrid",      "use_velocity": True,  "use_cum_rotation": True},
    {"name": "B_tcn_full",           "backbone": "tcn",         "use_velocity": True,  "use_cum_rotation": True},
    {"name": "C_transformer_full",   "backbone": "transformer", "use_velocity": True,  "use_cum_rotation": True},
    {"name": "D_hybrid_no_velocity", "backbone": "hybrid",      "use_velocity": False, "use_cum_rotation": True},
    {"name": "E_hybrid_no_cumrot",   "backbone": "hybrid",      "use_velocity": True,  "use_cum_rotation": False},
]


def _feature_cfg(cfg: dict) -> dict:
    """cfg для extract_pose_features из train_pose_ablation."""
    return {
        "smooth_sigma": SMOOTH_SIGMA,
        "use_velocity": cfg["use_velocity"],
        "use_acceleration": False,
        "use_cum_rotation": cfg["use_cum_rotation"],
    }


# ============================================================
# Dataset / Model
# ============================================================

class AblationVerifierDataset(Dataset):
    def __init__(self, windows, feature_cfg, augment=False):
        self.windows = windows
        self.feature_cfg = feature_cfg
        self.augment = augment
        self.labels = torch.tensor([w["label"] for w in windows], dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        kpts = torch.load(_window_cache_path(self.windows[idx]), weights_only=True)
        if self.augment:
            if USE_HFLIP and torch.rand(1).item() < 0.5:
                kpts = _hflip_keypoints(kpts)
            if MAX_TEMPORAL_ROLL > 0:
                shift = int(torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (1,)).item())
                if shift != 0:
                    kpts = kpts.roll(shifts=shift, dims=0)
        feats = extract_pose_features(kpts, self.feature_cfg)
        return feats, self.labels[idx]


class AblVerifierModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, backbone, num_frames=NUM_FRAMES):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(INPUT_DROPOUT),
        )
        if backbone == "hybrid":
            self.temporal = TemporalHybrid(dim=hidden_dim, num_frames=num_frames)
        elif backbone == "tcn":
            self.temporal = TemporalTCN(dim=hidden_dim, num_frames=num_frames)
        elif backbone == "transformer":
            self.temporal = TemporalTransformer(dim=hidden_dim, num_frames=num_frames)
        else:
            raise ValueError(backbone)
        self.head = TaskAttentionHead(hidden_dim, 2, HEAD_DROPOUT)

    def forward(self, x):
        x = self.input_proj(x)
        seq = self.temporal(x)
        logits, _ = self.head(seq)
        return logits


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    for feats, labels in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(feats), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def eval_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for feats, lbl in loader:
        logits = model(feats.to(DEVICE))
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        labels.extend(lbl.numpy())
    return np.array(probs), np.array(labels)


def metrics_at(probs, labels, thr=0.5):
    preds = (probs >= thr).astype(int)
    return (
        f1_score(labels, preds, zero_division=0),
        precision_score(labels, preds, zero_division=0),
        recall_score(labels, preds, zero_division=0),
    )


# ============================================================
# Run one config (3 seeds → ensemble)
# ============================================================

def run_config(cfg, windows, train_idx, val_idx):
    name = cfg["name"]
    feature_cfg = _feature_cfg(cfg)
    feature_dim = feature_dim_for_cfg(feature_cfg)
    print(f"\n{'#' * 64}\n{name}  (backbone={cfg['backbone']}, feature_dim={feature_dim})\n{'#' * 64}")

    labels_all = np.array([w["label"] for w in windows])
    train_labels = labels_all[train_idx]
    class_counts = np.bincount(train_labels, minlength=2)
    cls_w = torch.tensor([1.0, float(class_counts[0]) / max(class_counts[1], 1)], dtype=torch.float).to(DEVICE)

    seed_probs = []
    val_labels = None

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_ds = AblationVerifierDataset(windows, feature_cfg, augment=True)
        val_ds = AblationVerifierDataset(windows, feature_cfg, augment=False)

        weights = torch.tensor((1.0 / np.maximum(class_counts, 1))[train_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = AblVerifierModel(feature_dim, TEMPORAL_HIDDEN_DIM, cfg["backbone"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = EPOCHS * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS * len(train_loader), total_steps)
        criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=LABEL_SMOOTHING)

        best_f1, best_probs = -1.0, None
        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, train_loader, optimizer, scheduler, criterion)
            probs, lbls = eval_probs(model, val_loader)
            f1, _, _ = metrics_at(probs, lbls)
            if f1 > best_f1:
                best_f1, best_probs = f1, probs
        seed_probs.append(best_probs)
        val_labels = lbls
        f1, p, r = metrics_at(best_probs, val_labels)
        print(f"  seed {seed}: f1={f1:.3f}  P={p:.3f}  R={r:.3f}")

        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # ensemble — усреднение вероятностей
    ens_probs = np.mean(seed_probs, axis=0)
    print(f"  {'-' * 50}")
    print(f"  {'threshold':>10} {'precision':>10} {'recall':>9} {'f1':>8}")
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        f1, p, r = metrics_at(ens_probs, val_labels, thr)
        print(f"  {thr:>10.2f} {p:>10.3f} {r:>9.3f} {f1:>8.3f}")
        if best is None or f1 > best["f1"]:
            best = {"thr": thr, "f1": f1, "precision": p, "recall": r}

    return {
        "config": name,
        "backbone": cfg["backbone"],
        "feature_dim": feature_dim,
        "best_thr": best["thr"],
        "ens_f1": round(best["f1"], 4),
        "ens_precision": round(best["precision"], 4),
        "ens_recall": round(best["recall"], 4),
    }


# ============================================================
# Main
# ============================================================

def main():
    print(f"=== Verifier ablation ===  Device: {DEVICE}")

    windows = build_windows()  # загрузит verifier_windows_v2.json
    labels = np.array([w["label"] for w in windows])
    train_idx, val_idx = train_test_split(
        np.arange(len(windows)), test_size=0.2, stratify=labels, random_state=42,
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, configs: {[c['name'] for c in CONFIGS]}")

    results = []
    for cfg in CONFIGS:
        results.append(run_config(cfg, windows, train_idx, val_idx))
        pd.DataFrame(results).to_csv(CHECKPOINT_DIR / "ablation_results.csv", index=False)

    print(f"\n\n{'=' * 80}")
    print("VERIFIER ABLATION RESULTS  (3-seed ensemble, лучший порог)")
    print('=' * 80)
    df = pd.DataFrame(results).sort_values("ens_f1", ascending=False)
    print(df.to_string(index=False))
    print(f"\nReference (train_verifier.py v2, single seed): f1=0.889  P=0.835  R=0.950")
    print(f"CSV: {CHECKPOINT_DIR / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
