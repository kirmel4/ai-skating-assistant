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
from transformers import AutoModel, get_cosine_schedule_with_warmup

from scripts.clip_dataset import ClipDataset, LABEL_MAP, prepare_clip_dataset
from src.config import DataLoaderConfig, VideoConfig


def _resolve_model_dir() -> str:
    override = os.environ.get("VIDEOMAE_MODEL_DIR", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir() or not (p / "config.json").is_file():
            raise FileNotFoundError(f"VIDEOMAE_MODEL_DIR: нужна директория с config.json, получено: {p}")
        return str(p)
    local = _REPO_ROOT / "data" / "VideoMAEv2-Base"
    if local.is_dir() and (local / "config.json").is_file():
        return str(local.resolve())
    raise FileNotFoundError("Не найдена data/VideoMAEv2-Base. Укажи VIDEOMAE_MODEL_DIR.")


def _resolve_training_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if count == 1:
        return torch.device("cuda:0")
    idx = int(os.environ.get("VIDEOMAE_CUDA_DEVICE", "1"))
    if idx < 0 or idx >= count:
        raise RuntimeError(f"VIDEOMAE_CUDA_DEVICE={idx}, доступно только {count} GPU (индексы 0..{count - 1})")
    return torch.device(f"cuda:{idx}")


DEVICE = _resolve_training_device()
NUM_JUMP_CLASSES = len(LABEL_MAP)
MODEL_DIR = _resolve_model_dir()
CHECKPOINT_DIR = Path("checkpoints")

EPOCHS = 30
BATCH_SIZE = 4
LR_HEADS = 3e-4
LR_BACKBONE = 3e-6
N_UNFREEZE_BLOCKS = 6
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
HEAD_DROPOUT = 0.3
NUM_FRAMES = 16
IMAGE_SIZE = 224
HIDDEN_SIZE = 768

# VideoMAEv2 pretraining normalization (не ImageNet)
PIXEL_MEAN = torch.tensor([0.5, 0.5, 0.5])
PIXEL_STD = torch.tensor([0.5, 0.5, 0.5])

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


class MultiTaskVideoMAEv2(nn.Module):
    def __init__(
        self,
        num_jump_classes: int,
        num_rotation_classes: int,
        num_underrotation_classes: int,
        num_fall_classes: int,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.jump_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(HIDDEN_SIZE, num_jump_classes))
        self.rotation_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(HIDDEN_SIZE, num_rotation_classes))
        self.underrotation_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(HIDDEN_SIZE, num_underrotation_classes))
        self.fall_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(HIDDEN_SIZE, num_fall_classes))

    def forward(self, frames: torch.Tensor):
        # DataLoader: (B, T, C, H, W) → VideoMAEv2 ожидает (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        features = self.backbone.extract_features(x)  # (B, 768)
        return (
            self.jump_head(features),
            self.rotation_head(features),
            self.underrotation_head(features),
            self.fall_head(features),
        )


def normalize(frames: torch.Tensor) -> torch.Tensor:
    mean = PIXEL_MEAN.to(frames.device)[None, None, :, None, None]
    std = PIXEL_STD.to(frames.device)[None, None, :, None, None]
    return (frames - mean) / std


def augment_train(frames: torch.Tensor) -> torch.Tensor:
    # per-sample random horizontal flip на CPU
    flip = torch.rand(frames.shape[0]) > 0.5
    if flip.any():
        frames = frames.clone()
        frames[flip] = frames[flip].flip(-1)
    return frames


def train_epoch(model, loader, optimizer, scheduler, criterion, num_classes: dict):
    model.train()
    total_loss, total = 0.0, 0
    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = normalize(augment_train(frames).to(DEVICE))
        j_lbl, r_lbl, u_lbl, f_lbl = j_lbl.to(DEVICE), r_lbl.to(DEVICE), u_lbl.to(DEVICE), f_lbl.to(DEVICE)

        j_out, r_out, u_out, f_out = model(frames)

        # нормализация по log(K): без этого задачи с малым K (fall=2) доминируют,
        # мешая обучению jump (K=6) — именно это вызывало снижение jump_f1
        loss = (
            criterion(j_out, j_lbl) / math.log(num_classes["jump"])
            + criterion(r_out, r_lbl) / math.log(num_classes["rot"])
            + criterion(u_out, u_lbl) / math.log(num_classes["under"])
            + criterion(f_out, f_lbl) / math.log(num_classes["fall"])
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

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

    model = MultiTaskVideoMAEv2(
        num_jump_classes=NUM_JUMP_CLASSES,
        num_rotation_classes=len(rotation_map),
        num_underrotation_classes=len(UNDERROTATION_MAP),
        num_fall_classes=2,
    ).to(DEVICE)

    for param in model.backbone.parameters():
        param.requires_grad = False
    for block in model.backbone.model.blocks[-N_UNFREEZE_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.jump_head.parameters())
        + list(model.rotation_head.parameters())
        + list(model.underrotation_head.parameters())
        + list(model.fall_head.parameters())
    )
    trainable = sum(p.numel() for p in backbone_params + head_params)
    print(f"VideoMAEv2-Base: {MODEL_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"Trainable params: {trainable:,}  (backbone last {N_UNFREEZE_BLOCKS} blocks + heads)")

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 0.05},
        {"params": head_params, "lr": LR_HEADS, "weight_decay": 0.01},
    ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    num_classes = {
        "jump": NUM_JUMP_CLASSES,
        "rot": len(rotation_map),
        "under": len(UNDERROTATION_MAP),
        "fall": 2,
    }

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = {"train_loss": [], "jump_f1": [], "rot_f1": [], "under_f1": [], "fall_f1": []}
    best_jump_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, num_classes)
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
