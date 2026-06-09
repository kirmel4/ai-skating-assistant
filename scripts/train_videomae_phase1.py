"""
Фаза 1: SSv2-pretrain + crop-around-skater + 32 кадра + MLP-головы + focal loss + TTA.

Зависимости:
    pip install ultralytics            # для YOLOv8 (детекция фигуриста)

SSv2-веса:
    Локально:    data/videomae-base-ssv2/   (HF репо MCG-NJU/videomae-base-finetuned-ssv2)
    Или env:     VIDEOMAE_SSV2_DIR=/path/to/videomae-base-ssv2
    Иначе fallback на HF Hub.

YOLO модель yolov8n.pt подтянется автоматически в ~/.cache/Ultralytics при первом запуске.
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

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm
from transformers import VideoMAEModel, get_cosine_schedule_with_warmup

from scripts.clip_dataset import (
    BASE_DIR,
    ClipDataset,
    LABEL_MAP,
    _repo_relative_posix,
    _silence_libav_stderr,
    prepare_clip_dataset,
)
from src.config import DataLoaderConfig, VideoConfig


def _resolve_model_dir() -> tuple[str, bool]:
    override = os.environ.get("VIDEOMAE_SSV2_DIR", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir() or not (p / "config.json").is_file():
            raise FileNotFoundError(f"VIDEOMAE_SSV2_DIR: {p}")
        return str(p), True
    local = _REPO_ROOT / "data" / "videomae-base-ssv2"
    if local.is_dir() and (local / "config.json").is_file():
        return str(local.resolve()), True
    return "MCG-NJU/videomae-base-finetuned-ssv2", False


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
MODEL_NAME, _SSV2_LOCAL_ONLY = _resolve_model_dir()
CHECKPOINT_DIR = Path("checkpoints_phase1")
BBOX_CACHE_PATH = _REPO_ROOT / "data" / "skater_bboxes.json"

EPOCHS = 30
BATCH_SIZE = 4
LR_HEADS = 3e-4
LR_BACKBONE = 3e-6
# 4 блока + головы = ~28M обучаемых на 404 примера. 6 блоков (44M) переобучается.
N_UNFREEZE_BLOCKS = 4
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
HEAD_DROPOUT = 0.3
HEAD_HIDDEN = 512
FOCAL_GAMMA = 2.0
# SSv2 VideoMAE предобучен на 16 кадрах (positional embeds зашиты под 1568 токенов).
# Чтобы покрыть всю длительность прыжка (~1.3 сек), снижаем target_fps.
NUM_FRAMES = 16
TARGET_FPS = 12.5  # 16/12.5 = 1.28 сек контекста
IMAGE_SIZE = 224
BBOX_PADDING = 0.15
YOLO_MODEL = "yolov8m.pt"   # m лучше n на мелких объектах (фигурист далеко)
YOLO_CONF = 0.10
YOLO_IMGSZ = 1280

# SSv2 VideoMAE использует ImageNet нормализацию
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

UNDERROTATION_MAP = {"clean": 0, "ur": 1}


def map_fall(val) -> int:
    s = str(val).strip().lower().replace("?", "").replace("(0)", "")
    return int(float(s)) if s else 0


def _bbox_key(row) -> str:
    # уникальный ключ per-jump: путь клипа (jump_id это ТИП прыжка, не уникальный ID,
    # а start_sec_in_clip всегда BUFFER_SEC=2.0 у всех клипов)
    return Path(str(row["clip_path"])).as_posix()


def _build_indices_static(start_sec, end_sec, fps, total_frames, num_frames, target_fps):
    """Та же логика что в ClipDataset._build_indices, чтобы YOLO видел те же кадры что и тренер."""
    segment_duration = max(end_sec - start_sec, 1e-6)
    target_duration = num_frames / target_fps
    if segment_duration >= target_duration:
        ws, we = start_sec, end_sec
    else:
        c = (start_sec + end_sec) / 2.0
        h = target_duration / 2.0
        ws = min(c - h, start_sec)
        we = max(c + h, end_sec)
    ts = np.linspace(ws, we, num=num_frames, endpoint=True)
    idx = np.round(ts * fps).astype(np.int64)
    return np.clip(idx, 0, total_frames - 1)


def _interpolate_bboxes(bboxes: list) -> list:
    """Линейная интерполяция пропущенных bbox между известными."""
    out = list(bboxes)
    n = len(out)
    present = [i for i, b in enumerate(out) if b is not None]
    if not present:
        return out
    # экстраполяция концов: ближайшим известным
    for i in range(present[0]):
        out[i] = list(out[present[0]])
    for i in range(present[-1] + 1, n):
        out[i] = list(out[present[-1]])
    # линейная между известными
    for k in range(len(present) - 1):
        i, j = present[k], present[k + 1]
        if j - i > 1:
            for m in range(i + 1, j):
                t = (m - i) / (j - i)
                out[m] = [out[i][d] + t * (out[j][d] - out[i][d]) for d in range(4)]
    return out


def _expand_bbox(bbox, h, w, padding=BBOX_PADDING):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * padding, bh * padding
    return [
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(w, int(round(x2 + px))),
        min(h, int(round(y2 + py))),
    ]


def detect_skater_bboxes(df_clips, num_frames: int = NUM_FRAMES, target_fps: float = TARGET_FPS, device: torch.device | None = None, cache_path: Path | None = None, padding: float | None = None) -> dict[str, list]:
    """Per-frame YOLO детекция по тем же индексам, что использует тренировочный датасет.
    Возвращает {key: [bbox_per_frame] длины num_frames} с интерполяцией пропусков.

    device: GPU для YOLO. По умолчанию берёт модульный DEVICE.
    num_frames/target_fps: должны совпадать с теми что использует датасет на тренировке,
    иначе кэш bbox не сматчится с индексами кадров.
    cache_path: путь к JSON-кэшу. По умолчанию BBOX_CACHE_PATH. Для разных num_frames
    нужны разные кэши, т.к. длина списка bbox привязана к num_frames.
    padding: BBOX_PADDING override. Для разных padding нужны разные кэши.
    """
    if device is None:
        device = DEVICE
    if cache_path is None:
        cache_path = BBOX_CACHE_PATH
    if padding is None:
        padding = BBOX_PADDING
    cache: dict = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text())
        print(f"Loaded {len(cache)} cached bboxes from {cache_path}")

    keys_needed = [_bbox_key(r) for _, r in df_clips.iterrows()]
    missing = [k for k in keys_needed if k not in cache]

    if missing:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Установи `pip install ultralytics`.") from e
        print(f"Computing per-frame YOLO bboxes for {len(missing)} clips (one-time)...")
        yolo = YOLO(YOLO_MODEL)
        yolo.to(device)  # тот же GPU что и тренировка

        for _, row in tqdm(df_clips.iterrows(), total=len(df_clips), desc="YOLO"):
            key = _bbox_key(row)
            if key in cache:
                continue

            clip_path = str((BASE_DIR / _repo_relative_posix(row["clip_path"])).resolve())

            with _silence_libav_stderr():
                cap = cv2.VideoCapture(clip_path)
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            indices = _build_indices_static(
                float(row["start_sec_in_clip"]), float(row["end_sec_in_clip"]),
                fps, total, num_frames, target_fps,
            )

            # читаем все NUM_FRAMES кадров
            frames = []
            with _silence_libav_stderr():
                cap = cv2.VideoCapture(clip_path)
                for fi in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ret, fr = cap.read()
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ret else None)
                cap.release()

            valid_frames = [(i, f) for i, f in enumerate(frames) if f is not None]
            if not valid_frames:
                cache[key] = None
                continue

            # batched YOLO inference по всем валидным кадрам клипа
            valid_idx, valid_imgs = zip(*valid_frames)
            results = yolo(list(valid_imgs), classes=[0], verbose=False, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device)

            h, w = valid_imgs[0].shape[:2]
            per_frame = [None] * num_frames
            for vi, res in zip(valid_idx, results):
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                per_frame[vi] = _expand_bbox(xyxy[areas.argmax()].tolist(), h, w, padding=padding)

            if all(b is None for b in per_frame):
                cache[key] = None
                continue

            cache[key] = _interpolate_bboxes(per_frame)

        cache_path.write_text(json.dumps(cache))
        print(f"Saved bbox cache to {cache_path}")

    return {k: v for k, v in cache.items() if v is not None}


class CropClipDataset(ClipDataset):
    def __init__(self, df, num_frames, target_fps, image_size, return_meta, bboxes):
        super().__init__(df, num_frames, target_fps, image_size, return_meta)
        # bboxes[key] -> list длины num_frames из [x1,y1,x2,y2] (или None для всего клипа)
        self.bboxes = bboxes

    def _per_frame_crop_resize(self, frames_np: np.ndarray, per_frame_bboxes) -> np.ndarray:
        """Каждый кадр crop'ается своим bbox и ресайзится в IMAGE_SIZE с padding."""
        T = frames_np.shape[0]
        out = np.zeros((T, self.image_size, self.image_size, 3), dtype=np.uint8)
        for i in range(T):
            frame = frames_np[i]
            h, w = frame.shape[:2]

            if per_frame_bboxes is None or per_frame_bboxes[i] is None:
                crop = frame
            else:
                x1, y1, x2, y2 = (int(round(v)) for v in per_frame_bboxes[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame

            ch, cw = crop.shape[:2]
            scale = min(self.image_size / ch, self.image_size / cw)
            nh = max(1, int(round(ch * scale)))
            nw = max(1, int(round(cw * scale)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
            top = (self.image_size - nh) // 2
            left = (self.image_size - nw) // 2
            out[i, top:top + nh, left:left + nw] = resized
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = str((BASE_DIR / _repo_relative_posix(row["clip_path"])).resolve())

        with _silence_libav_stderr():
            cap = cv2.VideoCapture(clip_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        frame_indices = self._build_indices(
            start_sec=float(row["start_sec_in_clip"]),
            end_sec=float(row["end_sec_in_clip"]),
            fps=fps,
            total_frames=total_frames,
        )

        frames = self._read_frames(clip_path, frame_indices)  # (T, H, W, 3) RGB uint8

        key = Path(str(row["clip_path"])).as_posix()
        per_frame_bboxes = self.bboxes.get(key)

        cropped = self._per_frame_crop_resize(frames, per_frame_bboxes)
        frames_t = torch.from_numpy(cropped).permute(0, 3, 1, 2).contiguous().float() / 255.0

        label = LABEL_MAP.get(row["jump_type"], -1)
        return frames_t, label


class MultiTaskDataset(Dataset):
    def __init__(self, base, jump_labels, rotation_labels, underrotation_labels, fall_labels):
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


def make_mlp_head(in_dim: int, out_dim: int, hidden: int = HEAD_HIDDEN, dropout: float = HEAD_DROPOUT) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class MultiTaskVideoMAE(nn.Module):
    def __init__(self, num_jump, num_rot, num_under, num_fall):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(MODEL_NAME, local_files_only=_SSV2_LOCAL_ONLY)
        d = self.backbone.config.hidden_size
        self.jump_head = make_mlp_head(d, num_jump)
        self.rotation_head = make_mlp_head(d, num_rot)
        self.underrotation_head = make_mlp_head(d, num_under)
        self.fall_head = make_mlp_head(d, num_fall)

    def forward(self, pixel_values: torch.Tensor):
        # стандартный HF VideoMAE ожидает (B, T, C, H, W) — DataLoader даёт его напрямую
        pooled = self.backbone(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        return (
            self.jump_head(pooled),
            self.rotation_head(pooled),
            self.underrotation_head(pooled),
            self.fall_head(pooled),
        )


class FocalLoss(nn.Module):
    """Focal loss с label smoothing: focal_weight × CE_smoothed."""

    def __init__(self, gamma: float = FOCAL_GAMMA, label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pt = F.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        ce = F.cross_entropy(logits, target, label_smoothing=self.smoothing, reduction="none")
        return (focal_weight * ce).mean()


def normalize(frames: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(frames.device)[None, None, :, None, None]
    std = IMAGENET_STD.to(frames.device)[None, None, :, None, None]
    return (frames - mean) / std


def augment_train(frames: torch.Tensor) -> torch.Tensor:
    flip = torch.rand(frames.shape[0]) > 0.5
    if flip.any():
        frames = frames.clone()
        frames[flip] = frames[flip].flip(-1)
    return frames


def train_epoch(model, loader, optimizer, scheduler, focal, ce, num_classes):
    model.train()
    total_loss, total = 0.0, 0
    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = normalize(augment_train(frames).to(DEVICE))
        j_lbl, r_lbl, u_lbl, f_lbl = j_lbl.to(DEVICE), r_lbl.to(DEVICE), u_lbl.to(DEVICE), f_lbl.to(DEVICE)

        j_out, r_out, u_out, f_out = model(pixel_values=frames)

        # focal — только для jump (имбаланс A=122 vs T=45). Для остальных — CE.
        # нормализация по log(K): задачи с малым K (fall=2) не доминируют над jump (K=6).
        loss = (
            focal(j_out, j_lbl) / math.log(num_classes["jump"])
            + ce(r_out, r_lbl) / math.log(num_classes["rot"])
            + ce(u_out, u_lbl) / math.log(num_classes["under"])
            + ce(f_out, f_lbl) / math.log(num_classes["fall"])
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
    """TTA: original + horizontal flip, усреднение логитов."""
    model.eval()
    preds = {k: [] for k in ("jump", "rot", "under", "fall")}
    trues = {k: [] for k in ("jump", "rot", "under", "fall")}

    for frames, j_lbl, r_lbl, u_lbl, f_lbl in loader:
        frames = normalize(frames.to(DEVICE))
        j1, r1, u1, f1 = model(pixel_values=frames)
        j2, r2, u2, f2 = model(pixel_values=frames.flip(-1))
        j_out = (j1 + j2) / 2
        r_out = (r1 + r2) / 2
        u_out = (u1 + u2) / 2
        f_out = (f1 + f2) / 2

        preds["jump"].extend(j_out.argmax(1).cpu().numpy())
        preds["rot"].extend(r_out.argmax(1).cpu().numpy())
        preds["under"].extend(u_out.argmax(1).cpu().numpy())
        preds["fall"].extend(f_out.argmax(1).cpu().numpy())

        trues["jump"].extend(j_lbl.numpy())
        trues["rot"].extend(r_lbl.numpy())
        trues["under"].extend(u_lbl.numpy())
        trues["fall"].extend(f_lbl.numpy())

    return {k: f1_score(trues[k], preds[k], average="macro", zero_division=0) for k in preds}


def save_plots(history: dict, epoch: int):
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epoch + 1), history["train_loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Phase 1)")
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
    ax.set_title("Validation F1 (TTA)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "f1.png")
    plt.close(fig)


def main():
    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=TARGET_FPS, image_size=IMAGE_SIZE, return_meta=False)
    data_config = DataLoaderConfig(batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2)

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config)

    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}
    print(f"Rotation classes: {rotation_map}")

    bboxes = detect_skater_bboxes(df_clips)
    print(f"Skater bboxes: {len(bboxes)}/{len(df_clips)} clips")

    base_dataset = CropClipDataset(
        df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
    )

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    underrotation_labels = df_clips["underrotation"].map(UNDERROTATION_MAP).values
    fall_labels = df_clips["fall"].apply(map_fall).values

    dataset = MultiTaskDataset(base_dataset, jump_labels, rotation_labels, underrotation_labels, fall_labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(df_clips)), test_size=0.2, stratify=jump_labels, random_state=42,
    )

    train_jump_labels = jump_labels[train_idx]
    class_counts = np.bincount(train_jump_labels, minlength=NUM_JUMP_CLASSES)
    sample_weights = torch.tensor((1.0 / class_counts)[train_jump_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskVideoMAE(
        num_jump=NUM_JUMP_CLASSES, num_rot=len(rotation_map),
        num_under=len(UNDERROTATION_MAP), num_fall=2,
    ).to(DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in model.backbone.encoder.layer[-N_UNFREEZE_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.jump_head.parameters())
        + list(model.rotation_head.parameters())
        + list(model.underrotation_head.parameters())
        + list(model.fall_head.parameters())
    )
    trainable = sum(p.numel() for p in backbone_params + head_params)
    src = "локально" if _SSV2_LOCAL_ONLY else "Hugging Face Hub"
    print(f"Backbone (SSv2-pretrain): {MODEL_NAME} ({src})")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"NUM_FRAMES={NUM_FRAMES}, MLP heads (h={HEAD_HIDDEN}), focal_gamma={FOCAL_GAMMA}, TTA=h-flip")
    print(f"Trainable params: {trainable:,}  (backbone last {N_UNFREEZE_BLOCKS} blocks + heads)")

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 0.05},
        {"params": head_params, "lr": LR_HEADS, "weight_decay": 0.01},
    ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    focal = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    num_classes = {
        "jump": NUM_JUMP_CLASSES, "rot": len(rotation_map),
        "under": len(UNDERROTATION_MAP), "fall": 2,
    }

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = {"train_loss": [], "jump_f1": [], "rot_f1": [], "under_f1": [], "fall_f1": []}
    best_jump_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, focal, ce, num_classes)
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
