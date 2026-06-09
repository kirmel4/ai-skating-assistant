"""Single-task VideoMAE training. Один скрипт, одна задача.

Использование:
    python scripts/train_single_task.py --task jump
    python scripts/train_single_task.py --task rot
    python scripts/train_single_task.py --task under
    python scripts/train_single_task.py --task fall

Параллельный запуск 4 задач — см. scripts/launch_4tasks.sh.

Чекпоинты пишутся в checkpoints_{task}/. Backbone, dataset, YOLO crop, focal loss —
переиспользуются из train_videomae_phase1.py (один источник истины).
"""

from __future__ import annotations

import argparse
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
from transformers import VideoMAEModel, get_cosine_schedule_with_warmup

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from scripts.train_videomae_phase1 import (
    DEVICE,
    FOCAL_GAMMA,
    GRAD_CLIP,
    IMAGE_SIZE,
    LABEL_SMOOTHING,
    LR_BACKBONE,
    LR_HEADS,
    MODEL_NAME,
    NUM_FRAMES,
    N_UNFREEZE_BLOCKS,
    TARGET_FPS,
    UNDERROTATION_MAP,
    WARMUP_EPOCHS,
    CropClipDataset,
    FocalLoss,
    _SSV2_LOCAL_ONLY,
    augment_train,
    detect_skater_bboxes,
    make_mlp_head,
    map_fall,
    normalize,
)
from src.config import DataLoaderConfig, VideoConfig

EPOCHS = 30
BATCH_SIZE = 4

TASK_CONFIG = {
    "jump":  {"field": "jump_type",     "num_classes": 6, "use_focal": True,  "use_sampler": True,  "mapping": "label_map"},
    "rot":   {"field": "rotations",     "num_classes": 4, "use_focal": False, "use_sampler": False, "mapping": "auto_int"},
    "under": {"field": "underrotation", "num_classes": 2, "use_focal": False, "use_sampler": False, "mapping": "underrotation_map"},
    "fall":  {"field": "fall",          "num_classes": 2, "use_focal": False, "use_sampler": False, "mapping": "fall"},
}


class SingleTaskDataset(Dataset):
    def __init__(self, base, labels):
        self.base = base
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, _ = self.base[idx]
        return frames, self.labels[idx]


class SingleTaskVideoMAE(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(MODEL_NAME, local_files_only=_SSV2_LOCAL_ONLY)
        # экономит ~50% активационной памяти ценой ~30% времени — критично при параллельном запуске
        self.backbone.gradient_checkpointing_enable()
        d = self.backbone.config.hidden_size
        self.head = make_mlp_head(d, num_classes)

    def forward(self, pixel_values: torch.Tensor):
        pooled = self.backbone(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        return self.head(pooled)


def get_labels(df_clips, cfg):
    field = cfg["field"]
    m = cfg["mapping"]
    if m == "label_map":
        return df_clips[field].map(LABEL_MAP).values
    if m == "underrotation_map":
        return df_clips[field].map(UNDERROTATION_MAP).values
    if m == "fall":
        return df_clips[field].apply(map_fall).values
    if m == "auto_int":
        values = sorted(df_clips[field].astype(int).unique().tolist())
        rotation_map = {v: i for i, v in enumerate(values)}
        return df_clips[field].astype(int).map(rotation_map).values
    raise ValueError(f"unknown mapping: {m}")


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, total = 0.0, 0
    for frames, lbl in loader:
        frames = normalize(augment_train(frames).to(DEVICE))
        lbl = lbl.to(DEVICE)
        out = model(pixel_values=frames)
        loss = criterion(out, lbl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(lbl)
        total += len(lbl)
    return total_loss / total


@torch.no_grad()
def eval_epoch(model, loader):
    """TTA: original + h-flip."""
    model.eval()
    preds, trues = [], []
    for frames, lbl in loader:
        frames = normalize(frames.to(DEVICE))
        out1 = model(pixel_values=frames)
        out2 = model(pixel_values=frames.flip(-1))
        out = (out1 + out2) / 2
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(lbl.numpy())
    return f1_score(trues, preds, average="macro", zero_division=0)


def save_plots(history, ckpt_dir, task: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history["loss"]) + 1), history["loss"], marker="o")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"Training Loss — {task}")
    ax.grid(True); fig.tight_layout(); fig.savefig(ckpt_dir / "loss.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history["f1"]) + 1), history["f1"], marker="o")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (macro)"); ax.set_title(f"Validation F1 (TTA) — {task}")
    ax.grid(True); fig.tight_layout(); fig.savefig(ckpt_dir / "f1.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]
    ckpt_dir = Path(f"checkpoints_{args.task}")
    ckpt_dir.mkdir(exist_ok=True)

    print(f"=== Task: {args.task} | Device: {DEVICE} ===")

    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=TARGET_FPS, image_size=IMAGE_SIZE, return_meta=False)
    data_config = DataLoaderConfig(batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2)
    df_clips, _, _ = prepare_clip_dataset(video_config, data_config)

    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    bboxes = detect_skater_bboxes(df_clips)
    print(f"Skater bboxes: {len(bboxes)}/{len(df_clips)}")

    base_dataset = CropClipDataset(
        df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
    )

    labels = get_labels(df_clips, cfg)
    dataset = SingleTaskDataset(base_dataset, labels)

    # стратификация всегда по jump_type — общий val split для всех 4 задач, метрики сравнимы
    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    if cfg["use_sampler"]:
        train_lbls = labels[train_idx]
        counts = np.bincount(train_lbls, minlength=cfg["num_classes"])
        weights = torch.tensor((1.0 / counts)[train_lbls], dtype=torch.float)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SingleTaskVideoMAE(num_classes=cfg["num_classes"]).to(DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in model.backbone.encoder.layer[-N_UNFREEZE_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    trainable = sum(p.numel() for p in backbone_params + head_params)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 0.05},
        {"params": head_params, "lr": LR_HEADS, "weight_decay": 0.01},
    ])
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if cfg["use_focal"]:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Trainable: {trainable:,}")
    print(f"Loss: {'Focal' if cfg['use_focal'] else 'CE'}, Sampler: {'weighted' if cfg['use_sampler'] else 'shuffle'}")

    history = {"loss": [], "f1": []}
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        f1 = eval_epoch(model, val_loader)

        history["loss"].append(loss)
        history["f1"].append(f1)

        print(f"[{args.task} {epoch}/{EPOCHS}]  loss={loss:.4f}  f1={f1:.3f}")
        save_plots(history, ckpt_dir, args.task)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"         -> best saved (f1={f1:.3f})")


if __name__ == "__main__":
    main()
