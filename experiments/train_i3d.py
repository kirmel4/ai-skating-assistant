from __future__ import annotations

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

from scripts.clip_dataset import ClipDataset, LABEL_MAP, prepare_clip_dataset
from src.config import DataLoaderConfig, VideoConfig


def _resolve_training_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if count == 1:
        return torch.device("cuda:0")
    idx = int(os.environ.get("I3D_CUDA_DEVICE", "1"))
    if idx < 0 or idx >= count:
        raise RuntimeError(f"I3D_CUDA_DEVICE={idx}, доступно только {count} GPU (индексы 0..{count - 1})")
    return torch.device(f"cuda:{idx}")


DEVICE = _resolve_training_device()
NUM_JUMP_CLASSES = len(LABEL_MAP)
CHECKPOINT_DIR = Path("checkpoints_i3d")

EPOCHS = 20
BATCH_SIZE = 4
LR_HEADS = 1e-4
LR_BACKBONE = 1e-5
N_UNFREEZE_STAGES = 2
NUM_FRAMES = 16
IMAGE_SIZE = 224
I3D_FEATURE_DIM = 2048

KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45])
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225])

UNDERROTATION_MAP = {"clean": 0, "ur": 1}


def map_fall(val) -> int:
    s = str(val).strip().lower().replace("?", "").replace("(0)", "")
    return int(float(s)) if s else 0


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        base: ClipDataset,
        jump_labels: np.ndarray,
        rotation_labels: np.ndarray,
        underrotation_labels: np.ndarray,
        fall_labels: np.ndarray,
    ):
        self.base = base
        self.jump = torch.tensor(jump_labels, dtype=torch.long)
        self.rotation = torch.tensor(rotation_labels, dtype=torch.long)
        self.underrotation = torch.tensor(underrotation_labels, dtype=torch.long)
        self.fall = torch.tensor(fall_labels, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, _ = self.base[idx]
        return frames, self.jump[idx], self.rotation[idx], self.underrotation[idx], self.fall[idx]


class _I3DBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        full = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
        # full.blocks: stem + 4 res stages + head — убираем голову (последний блок)
        self.blocks = nn.Sequential(*list(full.blocks[:-1]))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.blocks(x)
        x = self.pool(x)
        return x.flatten(1)  # (B, 2048)


class MultiTaskI3D(nn.Module):
    def __init__(
        self,
        num_jump_classes: int,
        num_rotation_classes: int,
        num_underrotation_classes: int,
        num_fall_classes: int,
    ):
        super().__init__()
        self.backbone = _I3DBackbone()
        self.jump_head = nn.Linear(I3D_FEATURE_DIM, num_jump_classes)
        self.rotation_head = nn.Linear(I3D_FEATURE_DIM, num_rotation_classes)
        self.underrotation_head = nn.Linear(I3D_FEATURE_DIM, num_underrotation_classes)
        self.fall_head = nn.Linear(I3D_FEATURE_DIM, num_fall_classes)

    def forward(self, frames: torch.Tensor):
        # frames от DataLoader: (B, T, C, H, W) -> I3D ожидает (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)
        features = self.backbone(x)
        return (
            self.jump_head(features),
            self.rotation_head(features),
            self.underrotation_head(features),
            self.fall_head(features),
        )


def normalize(frames: torch.Tensor) -> torch.Tensor:
    # frames: (B, T, C, H, W)
    mean = KINETICS_MEAN.to(frames.device)[None, None, :, None, None]
    std = KINETICS_STD.to(frames.device)[None, None, :, None, None]
    return (frames - mean) / std


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total = 0.0, 0
    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = normalize(frames.to(DEVICE))
        j_lbl, r_lbl, u_lbl, f_lbl = j_lbl.to(DEVICE), r_lbl.to(DEVICE), u_lbl.to(DEVICE), f_lbl.to(DEVICE)

        j_out, r_out, u_out, f_out = model(frames)
        loss = criterion(j_out, j_lbl) + criterion(r_out, r_lbl) + criterion(u_out, u_lbl) + criterion(f_out, f_lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(j_lbl)
        total += len(j_lbl)

    return total_loss / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    preds = {k: [] for k in ("jump", "rot", "under", "fall")}
    trues = {k: [] for k in ("jump", "rot", "under", "fall")}

    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = normalize(frames.to(DEVICE))
        j_out, r_out, u_out, f_out = model(frames)

        preds["jump"].extend(j_out.argmax(1).cpu().numpy())
        preds["rot"].extend(r_out.argmax(1).cpu().numpy())
        preds["under"].extend(u_out.argmax(1).cpu().numpy())
        preds["fall"].extend(f_out.argmax(1).cpu().numpy())

        trues["jump"].extend(j_lbl.numpy())
        trues["rot"].extend(r_lbl.numpy())
        trues["under"].extend(u_lbl.numpy())
        trues["fall"].extend(f_lbl.numpy())

    return {
        k: f1_score(trues[k], preds[k], average="macro", zero_division=0)
        for k in preds
    }


def save_plots(history: dict, epoch: int):
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epoch + 1), history["train_loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epoch + 1), history["jump_f1"], marker="o", label="jump_type_f1")
    ax.plot(range(1, epoch + 1), history["rot_f1"], marker="s", label="rotations_f1")
    ax.plot(range(1, epoch + 1), history["under_f1"], marker="^", label="underrotation_f1")
    ax.plot(range(1, epoch + 1), history["fall_f1"], marker="D", label="fall_f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Validation F1")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "f1.png")
    plt.close(fig)


def main():
    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=25.0, image_size=IMAGE_SIZE, return_meta=False)
    data_config = DataLoaderConfig(batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2)

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config)

    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}
    print(f"Rotation classes: {rotation_map}")

    base_dataset = ClipDataset(df=df_clips, num_frames=NUM_FRAMES, target_fps=25.0, image_size=IMAGE_SIZE, return_meta=False)

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    underrotation_labels = df_clips["underrotation"].map(UNDERROTATION_MAP).values
    fall_labels = df_clips["fall"].apply(map_fall).values

    dataset = MultiTaskDataset(base_dataset, jump_labels, rotation_labels, underrotation_labels, fall_labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)),
        test_size=0.2,
        stratify=jump_labels,
        random_state=42,
    )

    train_jump_labels = jump_labels[train_idx]
    class_counts = np.bincount(train_jump_labels, minlength=NUM_JUMP_CLASSES)
    sample_weights = torch.tensor((1.0 / class_counts)[train_jump_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskI3D(
        num_jump_classes=NUM_JUMP_CLASSES,
        num_rotation_classes=len(rotation_map),
        num_underrotation_classes=len(UNDERROTATION_MAP),
        num_fall_classes=2,
    ).to(DEVICE)

    for param in model.backbone.parameters():
        param.requires_grad = False
    stages = list(model.backbone.blocks.children())
    for stage in stages[-N_UNFREEZE_STAGES:]:
        for param in stage.parameters():
            param.requires_grad = True

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.jump_head.parameters())
        + list(model.rotation_head.parameters())
        + list(model.underrotation_head.parameters())
        + list(model.fall_head.parameters())
    )
    trainable = sum(p.numel() for p in backbone_params + head_params)
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"Trainable params: {trainable:,}  (I3D last {N_UNFREEZE_STAGES} res-stages + heads)")

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEADS},
    ])
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = {"train_loss": [], "jump_f1": [], "rot_f1": [], "under_f1": [], "fall_f1": []}
    best_jump_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        f1s = eval_epoch(model, val_loader)

        history["train_loss"].append(train_loss)
        history["jump_f1"].append(f1s["jump"])
        history["rot_f1"].append(f1s["rot"])
        history["under_f1"].append(f1s["under"])
        history["fall_f1"].append(f1s["fall"])

        print(
            f"[{epoch}/{EPOCHS}]  loss={train_loss:.4f}  "
            f"jump_f1={f1s['jump']:.3f}  rot_f1={f1s['rot']:.3f}  "
            f"under_f1={f1s['under']:.3f}  fall_f1={f1s['fall']:.3f}"
        )

        save_plots(history, epoch)

        if f1s["jump"] > best_jump_f1:
            best_jump_f1 = f1s["jump"]
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
            print(f"         -> best saved (jump_f1={f1s['jump']:.3f})")


if __name__ == "__main__":
    main()
