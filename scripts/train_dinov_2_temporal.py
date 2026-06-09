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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup

from scripts.clip_dataset import ClipDataset, LABEL_MAP, prepare_clip_dataset
from scripts.train_videomae_phase1 import CropClipDataset, detect_skater_bboxes
from src.config import DataLoaderConfig, VideoConfig


# ============================================================
# Config
# ============================================================

DEVICE_ID = int(os.environ.get("DINO_CUDA_DEVICE", "1"))
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints_dinov2_temporal")
CHECKPOINT_DIR.mkdir(exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 2
NUM_FRAMES = 64
IMAGE_SIZE = 224
TARGET_FPS = 25.0

# Можно указать локальный путь:
# export DINO_MODEL_DIR=/path/to/dinov2-base
# Если env/local model не найдены, будет использован huggingface id.
DINO_MODEL_ID = "facebook/dinov2-base"

# transformer | tcn | hybrid
TEMPORAL_BACKBONE: Literal["transformer", "tcn", "hybrid"] = "transformer"

# DINOv2 frozen, поэтому обучаются только temporal model + heads.
FREEZE_DINO = True
FRAME_CHUNK_SIZE = 32  # сколько кадров прогонять через DINO за один micro-batch

# Для честного сравнения с вашим VideoMAE-скриптом по умолчанию оставлен random split.
# Для финальной оценки лучше включить True, если в df есть video_id/source_video/path.
USE_GROUP_SPLIT = False
GROUP_COL_CANDIDATES = ["video_id", "source_video", "video", "file", "path", "video_path"]

# Аугментации. Горизонтальный flip для фигурного катания спорный: может портить jump_type.
USE_HORIZONTAL_FLIP = False
USE_TEMPORAL_ROLL = True
# 64 кадра охватывают ~2.5 сек — без проблем сдвинуть на 8 кадров (~0.3 сек)
MAX_TEMPORAL_ROLL = 8

# YOLO crop вокруг фигуриста (использует кэш data/skater_bboxes.json)
USE_SKATER_CROP = True

# Pre-cache фичей DINO. DINO заморожен → один раз прогнали по всем клипам, сохранили (T, D)
# тензоры. Обучение читает их с диска вместо кадров — нет forward через DINO каждую эпоху.
# Скорость эпохи: с минут до секунд. Память: освобождает ~2GB DINO во время train.
# Trade-off: h-flip аугментация невозможна (нужен fresh DINO inference), но она и так выключена.
USE_FEATURE_CACHE = True

# Optimizer
LR_TEMPORAL = 3e-4
LR_HEADS = 5e-4
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.05
HEAD_DROPOUT = 0.35
USE_AMP = True

# Transformer params
TRANSFORMER_DEPTH = 3
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_RATIO = 4
TRANSFORMER_DROPOUT = 0.15

# TCN params
TCN_DEPTH = 4
TCN_KERNEL_SIZE = 5
TCN_DROPOUT = 0.15

NUM_JUMP_CLASSES = len(LABEL_MAP)

# DINOv2 / ImageNet normalization.
PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])


# ============================================================
# Helpers
# ============================================================

def _resolve_dino_model_dir_or_id() -> str:
    override = os.environ.get("DINO_MODEL_DIR", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir() or not (p / "config.json").is_file():
            raise FileNotFoundError(f"DINO_MODEL_DIR: нужна директория с config.json, получено: {p}")
        return str(p)

    for local_name in ["dinov2-base", "DINOv2-Base", "facebook_dinov2-base"]:
        local = _REPO_ROOT / "data" / local_name
        if local.is_dir() and (local / "config.json").is_file():
            return str(local.resolve())

    return DINO_MODEL_ID


def map_fall(val) -> int:
    s = str(val).strip().lower().replace("?", "").replace("(0)", "")
    return int(float(s)) if s else 0


def normalize_underrotation_value(val) -> str:
    s = str(val).strip().lower()
    s = s.replace(" ", "")
    if s in {"0", "false", "no", "none", "clean", "ok", "недокрутанет", "безнедокрута"}:
        return "clean"
    if s in {"1", "true", "yes", "ur", "under", "underrotated", "недокрут"}:
        return "ur"
    if s in {"q", "quarter"}:
        return "q"
    if s in {"<", "under<"}:
        return "<"
    if s in {"<<", "downgrade", "dg"}:
        return "<<"
    return s


def make_underrotation_map(values) -> dict[str, int]:
    normalized = [normalize_underrotation_value(v) for v in values]
    preferred = ["clean", "q", "ur", "<", "<<"]
    result = {}
    for key in preferred:
        if key in normalized and key not in result:
            result[key] = len(result)
    for key in sorted(set(normalized)):
        if key not in result:
            result[key] = len(result)
    return result


def normalize_frames(x: torch.Tensor) -> torch.Tensor:
    # Если на вход уже фичи (B, T, D) — нормализация не нужна, DINO уже выдал нормированный output.
    if x.dim() != 5:
        return x
    mean = PIXEL_MEAN.to(x.device)[None, None, :, None, None]
    std = PIXEL_STD.to(x.device)[None, None, :, None, None]
    return (x - mean) / std


def augment_train(x: torch.Tensor) -> torch.Tensor:
    """Работает и с кадрами (B, T, C, H, W), и с фичами (B, T, D).
    H-flip только для кадров (на фичах потерял бы смысл — DINO не linear)."""
    if USE_HORIZONTAL_FLIP and x.dim() == 5:
        flip = torch.rand(x.shape[0]) > 0.5
        if flip.any():
            x = x.clone()
            x[flip] = x[flip].flip(-1)

    if USE_TEMPORAL_ROLL and MAX_TEMPORAL_ROLL > 0:
        shifts = torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (x.shape[0],))
        if (shifts != 0).any():
            x = x.clone()
            for i, shift in enumerate(shifts.tolist()):
                if shift != 0:
                    x[i] = x[i].roll(shifts=shift, dims=0)
    return x


# ============================================================
# DINO feature cache
# ============================================================

def _features_dir_for_config() -> Path:
    """Отдельный кэш на каждую комбинацию параметров — кросс-конфиги не путаются."""
    crop_tag = "crop" if USE_SKATER_CROP else "nocrop"
    return _REPO_ROOT / "data" / f"dino_features_n{NUM_FRAMES}_{crop_tag}"


def _feature_cache_path(row, features_dir: Path) -> Path:
    # data/clips/1/0_2102.mp4 → 1_0_2102.pt (уникальное имя per-jump)
    p = Path(str(row["clip_path"]))
    return features_dir / f"{p.parent.name}_{p.stem}.pt"


@torch.no_grad()
def precompute_dino_features(base_dataset, dino_model_dir_or_id: str, features_dir: Path, device: torch.device):
    """One-time DINO inference per clip. Saves (T, D) features per jump as .pt files."""
    features_dir.mkdir(parents=True, exist_ok=True)
    df = base_dataset.df

    missing = [i for i in range(len(df)) if not _feature_cache_path(df.iloc[i], features_dir).is_file()]
    if not missing:
        print(f"DINO features: all {len(df)} clips cached in {features_dir}")
        return

    print(f"Precomputing DINO features: {len(missing)} clips → {features_dir}")
    encoder = FrozenDINOv2FrameEncoder(
        model_dir_or_id=dino_model_dir_or_id,
        chunk_size=FRAME_CHUNK_SIZE,
        freeze=True,
    ).to(device)

    use_amp = USE_AMP and device.type == "cuda"
    for idx in tqdm(missing, desc="DINO inference"):
        row = df.iloc[idx]
        frames, _ = base_dataset[idx]                       # (T, C, H, W) [0,1]
        frames = frames.unsqueeze(0).to(device)             # (1, T, C, H, W)
        frames = normalize_frames(frames)
        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = encoder(frames)                         # (1, T, D)
        torch.save(feats.squeeze(0).cpu().float(), _feature_cache_path(row, features_dir))

    del encoder
    torch.cuda.empty_cache()
    print(f"DINO features: cached to {features_dir}")


class FeatureBaseDataset(Dataset):
    """Заменяет ClipDataset когда USE_FEATURE_CACHE=True. Возвращает (T, D) фичи с диска
    вместо кадров. Требует чтобы precompute_dino_features() был запущен заранее."""

    def __init__(self, df, features_dir: Path):
        self.df = df.reset_index(drop=True).copy()
        self.features_dir = features_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = torch.load(_feature_cache_path(row, self.features_dir), weights_only=True)
        # второй элемент для совместимости с MultiTaskDataset — он всё равно отбрасывается
        return feats, 0


# ============================================================
# Dataset
# ============================================================

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
        frames, _ = self.base[idx]  # expected: (T, C, H, W), values in [0, 1]
        return frames, self.jump[idx], self.rotation[idx], self.underrotation[idx], self.fall[idx]


# ============================================================
# Model: DINOv2 frozen frame encoder + temporal model + task heads
# ============================================================

class FrozenDINOv2FrameEncoder(nn.Module):
    def __init__(self, model_dir_or_id: str, chunk_size: int = 32, freeze: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir_or_id)
        self.chunk_size = chunk_size
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.freeze = freeze

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True):
        # Важно: внешний model.train() не должен переводить frozen DINO в train mode.
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def _encode_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (B*T, C, H, W)
        outputs = []
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            for start in range(0, x_flat.shape[0], self.chunk_size):
                chunk = x_flat[start:start + self.chunk_size]
                out = self.encoder(pixel_values=chunk)
                # DINOv2 in transformers returns last_hidden_state: (N, tokens, D).
                cls = out.last_hidden_state[:, 0]
                outputs.append(cls)
        return torch.cat(outputs, dim=0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W), normalized for DINOv2
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self._encode_flat(flat)
        return feats.view(b, t, self.hidden_size)


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int, num_frames: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=dim * TRANSFORMER_MLP_RATIO,
            dropout=TRANSFORMER_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_DEPTH)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.encoder(x)
        return self.norm(x)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(1, 2),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            Transpose(1, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class TemporalTCN(nn.Module):
    def __init__(self, dim: int, num_frames: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        blocks = []
        for i in range(TCN_DEPTH):
            blocks.append(
                TemporalConvBlock(
                    dim=dim,
                    kernel_size=TCN_KERNEL_SIZE,
                    dilation=2 ** i,
                    dropout=TCN_DROPOUT,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.blocks(x)
        return self.norm(x)


class TemporalHybrid(nn.Module):
    def __init__(self, dim: int, num_frames: int):
        super().__init__()
        self.tcn = TemporalTCN(dim, num_frames)
        self.transformer = TemporalTransformer(dim, num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.tcn(x))


class TaskAttentionHead(nn.Module):
    """
    Separate attention pooling per task.

    Это важно для фигурного катания:
    - jump_type может смотреть на approach/takeoff;
    - rotations — на air phase;
    - underrotation — на landing moment;
    - fall — на after-landing/exit.
    """
    def __init__(self, dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(seq).squeeze(-1)       # (B, T)
        weights = torch.softmax(scores, dim=1)    # (B, T)
        pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return logits, weights


class MultiTaskDINOv2Temporal(nn.Module):
    def __init__(
        self,
        dino_model_dir_or_id: str,
        num_jump_classes: int,
        num_rotation_classes: int,
        num_underrotation_classes: int,
        num_fall_classes: int,
        use_cached_features: bool = False,
    ):
        super().__init__()
        if use_cached_features:
            # фичи уже посчитаны заранее — DINO в памяти не держим
            self.frame_encoder = None
            dim = int(AutoConfig.from_pretrained(dino_model_dir_or_id).hidden_size)
        else:
            self.frame_encoder = FrozenDINOv2FrameEncoder(
                model_dir_or_id=dino_model_dir_or_id,
                chunk_size=FRAME_CHUNK_SIZE,
                freeze=FREEZE_DINO,
            )
            dim = self.frame_encoder.hidden_size

        if TEMPORAL_BACKBONE == "transformer":
            self.temporal = TemporalTransformer(dim=dim, num_frames=NUM_FRAMES)
        elif TEMPORAL_BACKBONE == "tcn":
            self.temporal = TemporalTCN(dim=dim, num_frames=NUM_FRAMES)
        elif TEMPORAL_BACKBONE == "hybrid":
            self.temporal = TemporalHybrid(dim=dim, num_frames=NUM_FRAMES)
        else:
            raise ValueError(f"Unknown TEMPORAL_BACKBONE: {TEMPORAL_BACKBONE}")

        self.jump_head = TaskAttentionHead(dim, num_jump_classes, HEAD_DROPOUT)
        self.rotation_head = TaskAttentionHead(dim, num_rotation_classes, HEAD_DROPOUT)
        self.underrotation_head = TaskAttentionHead(dim, num_underrotation_classes, HEAD_DROPOUT)
        self.fall_head = TaskAttentionHead(dim, num_fall_classes, HEAD_DROPOUT)

    def forward(self, x: torch.Tensor):
        # x: либо (B, T, C, H, W) кадры, либо (B, T, D) кэшированные DINO фичи
        if x.dim() == 5:
            assert self.frame_encoder is not None, "Got frames but model в feature-cache mode"
            frame_features = self.frame_encoder(x)   # (B, T, D)
        else:
            frame_features = x                        # уже фичи
        seq = self.temporal(frame_features)          # (B, T, D)

        j_out, j_w = self.jump_head(seq)
        r_out, r_w = self.rotation_head(seq)
        u_out, u_w = self.underrotation_head(seq)
        f_out, f_w = self.fall_head(seq)

        return {
            "jump": j_out,
            "rot": r_out,
            "under": u_out,
            "fall": f_out,
            "attn": {
                "jump": j_w,
                "rot": r_w,
                "under": u_w,
                "fall": f_w,
            },
        }


# ============================================================
# Train / eval
# ============================================================

def multitask_loss(outputs, labels, criterion, num_classes: dict) -> torch.Tensor:
    j_lbl, r_lbl, u_lbl, f_lbl = labels

    # Fall обычно проще и может быстро доминировать. Дадим ему меньший вес.
    # Если цель — именно fall detection, вес можно вернуть к 1.0.
    return (
        criterion(outputs["jump"], j_lbl) / math.log(num_classes["jump"])
        + criterion(outputs["rot"], r_lbl) / math.log(num_classes["rot"])
        + criterion(outputs["under"], u_lbl) / math.log(num_classes["under"])
        + 0.5 * criterion(outputs["fall"], f_lbl) / math.log(num_classes["fall"])
    )


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, num_classes: dict):
    model.train()
    total_loss, total = 0.0, 0

    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = augment_train(frames).to(DEVICE, non_blocking=True)
        frames = normalize_frames(frames)
        j_lbl = j_lbl.to(DEVICE, non_blocking=True)
        r_lbl = r_lbl.to(DEVICE, non_blocking=True)
        u_lbl = u_lbl.to(DEVICE, non_blocking=True)
        f_lbl = f_lbl.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(frames)
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

    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = normalize_frames(frames.to(DEVICE, non_blocking=True))

        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(frames)

        preds["jump"].extend(outputs["jump"].argmax(1).cpu().numpy())
        preds["rot"].extend(outputs["rot"].argmax(1).cpu().numpy())
        preds["under"].extend(outputs["under"].argmax(1).cpu().numpy())
        preds["fall"].extend(outputs["fall"].argmax(1).cpu().numpy())

        trues["jump"].extend(j_lbl.numpy())
        trues["rot"].extend(r_lbl.numpy())
        trues["under"].extend(u_lbl.numpy())
        trues["fall"].extend(f_lbl.numpy())

        batch_size = frames.shape[0]
        for k in attn_sum:
            attn_sum[k] += outputs["attn"][k].detach().cpu().sum(dim=0)
        attn_count += batch_size

    f1s = {
        k: f1_score(trues[k], preds[k], average="macro", zero_division=0)
        for k in preds
    }
    reports = {
        k: classification_report(trues[k], preds[k], zero_division=0)
        for k in preds
    }
    cms = {
        k: confusion_matrix(trues[k], preds[k])
        for k in preds
    }
    avg_attn = {
        k: (attn_sum[k] / max(attn_count, 1)).numpy()
        for k in attn_sum
    }
    return f1s, reports, cms, avg_attn


def save_plots(history: dict, epoch: int, avg_attn: dict[str, np.ndarray]):
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
    ax.set_ylabel("F1 macro")
    ax.set_title("Validation F1")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "f1.png")
    plt.close(fig)

    for task, weights in avg_attn.items():
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(np.arange(len(weights)), weights)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Average attention")
        ax.set_title(f"Average task attention: {task}")
        fig.tight_layout()
        fig.savefig(CHECKPOINT_DIR / f"attention_{task}.png")
        plt.close(fig)


def _find_group_values(df):
    for col in GROUP_COL_CANDIDATES:
        if col in df.columns:
            return df[col].astype(str).values, col
    return None, None


def make_split(df_clips, jump_labels):
    if USE_GROUP_SPLIT:
        groups, group_col = _find_group_values(df_clips)
        if groups is not None and len(np.unique(groups)) > 1:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(splitter.split(np.arange(len(df_clips)), jump_labels, groups=groups))
            print(f"Split: GroupShuffleSplit by {group_col}; groups={len(np.unique(groups))}")
            return train_idx, val_idx
        print("Split: group column not found, fallback to stratified random split")

    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)),
        test_size=0.2,
        stratify=jump_labels,
        random_state=42,
    )
    print("Split: stratified random by jump_type")
    return train_idx, val_idx


def build_loaders(dataset, df_clips, jump_labels):
    train_idx, val_idx = make_split(df_clips, jump_labels)

    train_jump_labels = jump_labels[train_idx]
    class_counts = np.bincount(train_jump_labels, minlength=NUM_JUMP_CLASSES)
    class_counts = np.maximum(class_counts, 1)
    sample_weights = torch.tensor((1.0 / class_counts)[train_jump_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_loader, val_loader, train_idx, val_idx


# ============================================================
# Main
# ============================================================

def main():
    dino_model_dir_or_id = _resolve_dino_model_dir_or_id()

    video_config = VideoConfig(
        num_frames=NUM_FRAMES,
        target_fps=TARGET_FPS,
        image_size=IMAGE_SIZE,
        return_meta=False,
    )
    data_config = DataLoaderConfig(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])
    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}

    under_values = df_clips["underrotation"].apply(normalize_underrotation_value).values
    underrotation_map = make_underrotation_map(under_values)

    print(f"Dataset: {len(df_clips)} jumps")
    print(f"DINOv2: {dino_model_dir_or_id}")
    print(f"Temporal backbone: {TEMPORAL_BACKBONE}")
    print(f"Frames: {NUM_FRAMES}, image_size={IMAGE_SIZE}, batch_size={BATCH_SIZE}")
    print(f"Rotation classes: {rotation_map}")
    print(f"Underrotation classes: {underrotation_map}")
    print(f"Device: {DEVICE}")

    if USE_SKATER_CROP:
        # отдельный кэш для 64-кадровой раскладки (phase1 использует 16-кадровый)
        bbox_cache_path = _REPO_ROOT / "data" / f"skater_bboxes_n{NUM_FRAMES}.json"
        bboxes = detect_skater_bboxes(
            df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
            device=DEVICE, cache_path=bbox_cache_path,
        )
        print(f"Skater bboxes: {len(bboxes)}/{len(df_clips)} clips with detection")
        frame_dataset = CropClipDataset(
            df=df_clips,
            num_frames=NUM_FRAMES,
            target_fps=TARGET_FPS,
            image_size=IMAGE_SIZE,
            return_meta=False,
            bboxes=bboxes,
        )
    else:
        frame_dataset = ClipDataset(
            df=df_clips,
            num_frames=NUM_FRAMES,
            target_fps=TARGET_FPS,
            image_size=IMAGE_SIZE,
            return_meta=False,
        )

    if USE_FEATURE_CACHE:
        features_dir = _features_dir_for_config()
        precompute_dino_features(frame_dataset, dino_model_dir_or_id, features_dir, DEVICE)
        base_dataset = FeatureBaseDataset(df_clips, features_dir)
        print(f"Training on cached DINO features ({features_dir})")
    else:
        base_dataset = frame_dataset
        print("Training on raw frames (DINO inference each step)")

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    dataset = MultiTaskDataset(base_dataset, jump_labels, rotation_labels, underrotation_labels, fall_labels)
    train_loader, val_loader, train_idx, val_idx = build_loaders(dataset, df_clips, jump_labels)

    model = MultiTaskDINOv2Temporal(
        dino_model_dir_or_id=dino_model_dir_or_id,
        num_jump_classes=NUM_JUMP_CLASSES,
        num_rotation_classes=len(rotation_map),
        num_underrotation_classes=len(underrotation_map),
        num_fall_classes=2,
        use_cached_features=USE_FEATURE_CACHE,
    ).to(DEVICE)

    # Параметры DINO не обучаем. Оптимизируем только temporal + heads.
    temporal_params = [p for p in model.temporal.parameters() if p.requires_grad]
    head_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and not name.startswith("frame_encoder.") and not name.startswith("temporal."):
            head_params.append(p)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"Trainable params: {trainable:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": temporal_params, "lr": LR_TEMPORAL, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": LR_HEADS, "weight_decay": WEIGHT_DECAY},
        ]
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")

    num_classes = {
        "jump": NUM_JUMP_CLASSES,
        "rot": len(rotation_map),
        "under": len(underrotation_map),
        "fall": 2,
    }

    history = {"train_loss": [], "jump_f1": [], "rot_f1": [], "under_f1": [], "fall_f1": []}
    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, num_classes)
        f1s, reports, cms, avg_attn = eval_epoch(model, val_loader)

        history["train_loss"].append(train_loss)
        history["jump_f1"].append(f1s["jump"])
        history["rot_f1"].append(f1s["rot"])
        history["under_f1"].append(f1s["under"])
        history["fall_f1"].append(f1s["fall"])

        # Основная боль — jump_type и rotations. Fall обычно проще, поэтому почти не влияет на best.
        score = 0.45 * f1s["jump"] + 0.35 * f1s["rot"] + 0.15 * f1s["under"] + 0.05 * f1s["fall"]

        print(
            f"[{epoch}/{EPOCHS}] loss={train_loss:.4f} "
            f"jump_f1={f1s['jump']:.3f} rot_f1={f1s['rot']:.3f} "
            f"under_f1={f1s['under']:.3f} fall_f1={f1s['fall']:.3f} "
            f"score={score:.3f}"
        )

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
                        "dino": dino_model_dir_or_id,
                        "num_frames": NUM_FRAMES,
                        "image_size": IMAGE_SIZE,
                        "temporal_backbone": TEMPORAL_BACKBONE,
                        "frame_chunk_size": FRAME_CHUNK_SIZE,
                        "freeze_dino": FREEZE_DINO,
                    },
                    "f1s": f1s,
                    "score": score,
                },
                CHECKPOINT_DIR / "best_dinov2_temporal.pt",
            )
            print(f"         -> best saved (score={score:.3f})")
            print("         jump report:")
            print(reports["jump"])
            print("         rotation report:")
            print(reports["rot"])

    print(f"Training finished. Best score={best_score:.3f}")
    print(f"Artifacts saved to: {CHECKPOINT_DIR.resolve()}")


if __name__ == "__main__":
    main()
