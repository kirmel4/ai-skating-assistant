"""Бинарный verifier: прыжок / не-прыжок. Версия v2 — больше данных + регуляризация.

Отсекает ложные кандидаты rotation-эвристики (detect_jumps.py).

Данные:
  Позитивы — ВСЕ нарезанные клипы прыжков (videos 1-11, ~505) из clip_offsets.json.
             Не требует скачивания полных видео — клипы уже на диске.
  Негативы — hard negatives: кандидаты detect_jumps (видео 9-11) не совпавшие с GT.

Единый pipeline для обоих классов:
  кадры окна (из клипа ИЛИ из полного видео) → YOLOv8-pose (главный фигурист)
  → (T,17,3) → F4 features (96) → input_proj → TCN+Transformer → binary head

pose-фичи нормализованы (центр=таз, масштаб=торс) → не несут инфо о видео/фоне,
поэтому позитивы из 1-11 и негативы из 9-11 не дают domain confound.

Требует заранее:
  - нарезанные клипы (data/clips/, clip_offsets.json)
  - detect_jumps.py на видео 9,10,11 → data/jump_detection/video{9,10,11}_candidates.json

Использование:
    python scripts/train_verifier.py
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
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from scripts.clip_dataset import prepare_clip_dataset
from scripts.train_dinov_2_temporal import TaskAttentionHead, TemporalHybrid
from scripts.train_pose_ablation import KEYPOINT_LR_PAIRS, _hflip_keypoints, _smooth_keypoints
from scripts.train_pose_temporal import KP, _joint_cos, _line_sincos
from src.config import DataLoaderConfig, VideoConfig
from src.utils.xlsx_parser import parse_excel


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

BASE_DIR = _REPO_ROOT
VIDEOS_DIR = BASE_DIR / "data" / "videos"
TIMECODES_PATH = BASE_DIR / "data" / "Разметка прыжков.xlsx"
JUMP_DETECTION_DIR = BASE_DIR / "data" / "jump_detection"
# доп. окна сверх кандидатов: подтверждённые прыжки (positives) и
# подтверждённые не-прыжки, на которых verifier ошибался (hard negatives)
VERIFIER_EXTRA_PATH = BASE_DIR / "data" / "verifier_extra_windows.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints_verifier"
CHECKPOINT_DIR.mkdir(exist_ok=True)
POSE_CACHE_DIR = BASE_DIR / "data" / "verifier_pose"
POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
WINDOWS_JSON = BASE_DIR / "data" / "verifier_windows_v2.json"

# полные видео — только для негативов (hard negatives)
VIDEO_FILES = {
    9: "Произвольная программа Девушки Омск.mp4",
    10: "Мужчины короткая программа Омск.mp4",
    11: "Произвольная программа Женщины Москва.mp4",
}

# Окно и сэмплинг
NUM_FRAMES = 64
WINDOW_SEC = 2.5           # тесное окно: прыжок занимает бóльшую долю кадров —
                           # сигнал не размывается, verifier увереннее (5с просаживал)
NEG_GT_MARGIN = 3.0        # запретный зазор вокруг размеченных прыжков: кандидат
                           # ближе этого в негативы НЕ берём. Второй прыжок каскада
                           # (часто не размечен) живёт ровно тут — иначе утёк бы
                           # в негативы и обучал verifier «прыжок = не прыжок».
FORCED_NEG_REPEAT = 8      # сколько раз дублировать каждый принудительный hard negative
                           # (один пример среди ~1500 негативов иначе почти не влияет)
SMOOTH_SIGMA = 1.0

# YOLO
POSE_MODEL = "yolov8m-pose.pt"
POSE_IMGSZ = 1280
POSE_CONF = 0.15
POSE_BATCH_FRAMES = 64      # цель размера батча YOLO в precompute_pose: кадры
                            # копятся с нескольких окон. Поднимай (128/256), если
                            # на GPU много свободной памяти; на занятом — OOM.

# Модель / тренировка — конфиг E из ablation (без cum_rotation)
FEATURE_DIM = 95            # 61 base + 34 velocity (cum_rotation убран — ablation показал что вредит)
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
SEEDS = [42, 43, 44]        # multi-seed ensemble (ablation: 3-seed ensemble F1=0.935)
DEFAULT_THRESHOLD = 0.6     # рабочий порог из ablation (R=1.0, P=0.88 на val)


def _time_to_sec(t) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second


# ============================================================
# Шаг 1: окна — позитивы из всех клипов + hard negatives из detect_jumps
# ============================================================

def build_windows() -> list[dict]:
    """{video_path, center_sec, label, uid}. label 1=jump, 0=hard-negative."""
    if WINDOWS_JSON.is_file():
        windows = json.loads(WINDOWS_JSON.read_text())
        pos = sum(w["label"] == 1 for w in windows)
        print(f"Загружено {len(windows)} окон ({pos} jump / {len(windows)-pos} non-jump)")
        return windows

    rng = np.random.default_rng(42)
    windows: list[dict] = []

    # --- позитивы: все нарезанные клипы прыжков ---
    video_config = VideoConfig(num_frames=NUM_FRAMES, target_fps=25.0, image_size=224, return_meta=False)
    data_config = DataLoaderConfig(batch_size=1, shuffle=False, num_workers=0,
                                   pin_memory=False, persistent_workers=False, prefetch_factor=2)
    # video_id=1 (data/clips/1) исключён из обучения — held-out
    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])

    for _, row in df_clips.iterrows():
        clip_rel = Path(str(row["clip_path"]).replace("\\", "/"))
        clip_abs = (BASE_DIR / clip_rel).resolve()
        if not clip_abs.is_file():
            continue
        center = (float(row["start_sec_in_clip"]) + float(row["end_sec_in_clip"])) / 2.0
        uid = f"p_{clip_rel.parent.name}_{clip_rel.stem}"
        windows.append({"video_path": str(clip_abs), "center_sec": center, "label": 1, "uid": uid})

    # --- доп. окна из verifier_extra_windows.json ---
    extra_pos_spans: dict[int, list] = {}   # для исключения из негативов
    forced_neg: list[dict] = []
    if VERIFIER_EXTRA_PATH.is_file():
        extra = json.loads(VERIFIER_EXTRA_PATH.read_text(encoding="utf-8"))
        for w in extra.get("positives", []):
            vid = int(w["video_id"])
            fname = VIDEO_FILES.get(vid)
            if fname is None:
                continue
            s, e = float(w["start"]), float(w["end"])
            c = (s + e) / 2.0
            windows.append({
                "video_path": str((VIDEOS_DIR / fname).resolve()),
                "center_sec": c, "label": 1, "uid": f"xp_{vid}_{int(round(c * 100))}",
            })
            extra_pos_spans.setdefault(vid, []).append((s, e))
        for w in extra.get("negatives", []):
            vid = int(w["video_id"])
            fname = VIDEO_FILES.get(vid)
            if fname is None:
                continue
            s, e = float(w["start"]), float(w["end"])
            c = (s + e) / 2.0
            neg = {
                "video_path": str((VIDEOS_DIR / fname).resolve()),
                "center_sec": c, "label": 0, "uid": f"xn_{vid}_{int(round(c * 100))}",
            }
            forced_neg.extend([neg] * FORCED_NEG_REPEAT)
        print(f"Доп. окна: +{len(extra.get('positives', []))} позитивов, "
              f"+{len(extra.get('negatives', []))} hard-negatives ×{FORCED_NEG_REPEAT}")

    n_pos = len(windows)

    # --- негативы: hard negatives из detect_jumps ---
    df_gt = parse_excel(TIMECODES_PATH)
    all_hard_neg: list[dict] = []
    for video_id, fname in VIDEO_FILES.items():
        sub = df_gt[df_gt["video_id"] == video_id]
        gt_spans = [(_time_to_sec(r["t_start_val"]), _time_to_sec(r["t_end_val"]))
                    for _, r in sub.iterrows()]
        # доп. подтверждённые прыжки тоже исключаем из негативов
        gt_spans += extra_pos_spans.get(video_id, [])
        cand_path = JUMP_DETECTION_DIR / f"video{video_id}_candidates.json"
        if not cand_path.is_file():
            raise FileNotFoundError(
                f"Нет {cand_path}. Запусти: python scripts/detect_jumps.py --video {video_id}"
            )
        cand_data = json.loads(cand_path.read_text())
        candidates = cand_data.get("candidates") or cand_data.get("candidates_raw") or []
        video_abs = str((VIDEOS_DIR / fname).resolve())
        n_skipped = 0
        for cs, ce in candidates:
            # запретный зазор NEG_GT_MARGIN: кандидат рядом с размеченным
            # прыжком в негативы не берём — там может быть каскадный партнёр
            near_jump = any(cs <= ge + NEG_GT_MARGIN and ce >= gs - NEG_GT_MARGIN
                            for gs, ge in gt_spans)
            if near_jump:
                n_skipped += 1
                continue
            c = (cs + ce) / 2.0
            all_hard_neg.append({
                "video_path": video_abs, "center_sec": c, "label": 0,
                "uid": f"n_{video_id}_{int(round(c * 100))}",
            })
        if n_skipped:
            print(f"  video{video_id}: {n_skipped} кандидатов в зоне ±{NEG_GT_MARGIN}с "
                  f"от прыжка — исключены из негативов")

    # весь пул hard negatives — все кандидаты detect_jumps вне GT по видео 9/10/11
    rng.shuffle(all_hard_neg)
    windows += all_hard_neg
    windows += forced_neg   # принудительные hard negatives — всегда в обучении

    print(f"Окна: {n_pos} позитивов + {len(all_hard_neg)} hard-negatives "
          f"+ {len(forced_neg)} forced-neg = {len(windows)}")
    WINDOWS_JSON.write_text(json.dumps(windows, indent=2, ensure_ascii=False))
    return windows


# ============================================================
# Шаг 2: YOLOv8-pose по окнам → кэш (T,17,3)
# ============================================================

def _window_cache_path(w: dict) -> Path:
    return POSE_CACHE_DIR / f"{w['uid']}.pt"


_POSE_META = POSE_CACHE_DIR / "_meta.json"


def _purge_stale_cache():
    """Кэш keypoints привязан к (WINDOW_SEC, NUM_FRAMES): при их смене старые
    .pt — тот же uid, но другая длина окна → невалидны. Сверяем с _meta.json
    и чистим автоматически, чтобы кэш не приходилось удалять руками."""
    POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cur = {"window_sec": WINDOW_SEC, "num_frames": NUM_FRAMES}
    if _POSE_META.is_file():
        if json.loads(_POSE_META.read_text()) == cur:
            return                                  # параметры окна не менялись — кэш валиден
        stale = list(POSE_CACHE_DIR.glob("*.pt"))
        for p in stale:
            p.unlink()
        print(f"Окно/кадры изменились — кэш keypoints устарел, удалено {len(stale)} файлов")
    _POSE_META.write_text(json.dumps(cur))


def _pick_person_kpts(res) -> np.ndarray | None:
    """YOLO-результат одного кадра → keypoints (17,3) главного фигуриста
    (самый большой bbox) или None, если никого не нашли."""
    if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
        return None
    kp = res.keypoints.data.cpu().numpy()
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        person = int(areas.argmax())
    else:
        person = 0
    return kp[person].astype(np.float32)


@torch.no_grad()
def precompute_pose(windows: list[dict]):
    """Для каждого окна: NUM_FRAMES кадров → YOLOv8-pose → главный фигурист.

    YOLO батчится сразу по нескольким окнам (~POSE_BATCH_FRAMES кадров за вызов)
    — GPU грузится полнее, чем при батче в одно окно (64 кадра)."""
    _purge_stale_cache()
    missing = [w for w in windows if not _window_cache_path(w).is_file()]
    if not missing:
        print(f"Pose: все {len(windows)} окон в кэше")
        return

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("pip install ultralytics") from e

    from tqdm import tqdm
    print(f"YOLOv8-pose: {len(missing)} окон → {POSE_CACHE_DIR}")
    yolo = YOLO(POSE_MODEL)
    yolo.to(DEVICE)

    by_path: dict[str, list[dict]] = {}
    for w in missing:
        by_path.setdefault(w["video_path"], []).append(w)

    for video_path, path_windows in tqdm(by_path.items(), desc="videos"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            for w in path_windows:
                torch.save(torch.zeros(NUM_FRAMES, 17, 3), _window_cache_path(w))
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        path_windows.sort(key=lambda w: w["center_sec"])

        # окна копятся в батч; YOLO вызывается одним прогоном на пачку окон
        batch: list[tuple[dict, list]] = []     # [(окно, NUM_FRAMES кадров)]

        def _flush_batch():
            if not batch:
                return
            all_frames = [fr for _, frs in batch for fr in frs]
            results = yolo(all_frames, verbose=False, conf=POSE_CONF,
                           imgsz=POSE_IMGSZ, device=DEVICE)
            off = 0
            for w, frs in batch:
                kpts = torch.zeros(NUM_FRAMES, 17, 3, dtype=torch.float32)
                for j in range(len(frs)):
                    person_kp = _pick_person_kpts(results[off + j])
                    if person_kp is not None:
                        kpts[j] = torch.from_numpy(person_kp)
                off += len(frs)
                torch.save(kpts, _window_cache_path(w))
            batch.clear()

        # внутренний бар по окнам — у полных видео их тысячи, иначе кажется, что висит
        bar = tqdm(path_windows, desc=Path(video_path).name[:30], leave=False)
        for w in bar:
            t0 = w["center_sec"] - WINDOW_SEC / 2
            start_frame = int(np.clip(t0 * src_fps, 0, max(total - 1, 0)))
            span_frames = int(WINDOW_SEC * src_fps) + 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            raw = []
            for _ in range(span_frames):
                ret, fr = cap.read()
                if not ret:
                    break
                raw.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

            if not raw:
                torch.save(torch.zeros(NUM_FRAMES, 17, 3), _window_cache_path(w))
                continue
            idx = np.linspace(0, len(raw) - 1, NUM_FRAMES).round().astype(int)
            batch.append((w, [raw[i] for i in idx]))
            if sum(len(frs) for _, frs in batch) >= POSE_BATCH_FRAMES:
                _flush_batch()
        _flush_batch()
        bar.close()
        cap.release()

    del yolo
    torch.cuda.empty_cache()


# ============================================================
# F4 features (96-dim)
# ============================================================

def extract_features_f4(kpts: torch.Tensor, smooth_sigma: float = 0.0) -> torch.Tensor:
    """Конфиг E: 95-dim (61 base + 34 velocity). cum_rotation убран — ablation показал
    что он вредит (высокий и у прыжка, и у твиззла → не разделяет классы)."""
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

    return torch.cat([
        xy_norm.flatten(1), conf, shoulder_sc, hip_sc, spine_sc,
        L_knee, R_knee, L_elbow, R_elbow,
        xy_velocity.flatten(1),
    ], dim=-1)  # 95


# ============================================================
# Dataset / Model
# ============================================================

class VerifierDataset(Dataset):
    def __init__(self, windows, augment=False):
        self.windows = windows
        self.augment = augment
        self.labels = torch.tensor([w["label"] for w in windows], dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        kpts = torch.load(_window_cache_path(self.windows[idx]), weights_only=True)  # (T,17,3)
        if self.augment:
            if USE_HFLIP and torch.rand(1).item() < 0.5:
                kpts = _hflip_keypoints(kpts)
            if MAX_TEMPORAL_ROLL > 0:
                shift = int(torch.randint(-MAX_TEMPORAL_ROLL, MAX_TEMPORAL_ROLL + 1, (1,)).item())
                if shift != 0:
                    kpts = kpts.roll(shifts=shift, dims=0)
        feats = extract_features_f4(kpts, smooth_sigma=SMOOTH_SIGMA)
        return feats, self.labels[idx]


class VerifierModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_frames=NUM_FRAMES):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(INPUT_DROPOUT),
        )
        self.temporal = TemporalHybrid(dim=hidden_dim, num_frames=num_frames)
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
    total_loss, total = 0.0, 0
    for feats, labels in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for feats, labels in loader:
        feats = feats.to(DEVICE)
        logits = model(feats)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "probs": probs, "labels": labels,
    }


def save_history_plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], marker="o", color="tab:blue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training Loss"); axes[0].grid(True)
    axes[1].plot(epochs, history["f1"], marker="o", label="F1")
    axes[1].plot(epochs, history["precision"], marker="s", label="Precision")
    axes[1].plot(epochs, history["recall"], marker="^", label="Recall")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation (jump class)"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "history.png", dpi=120)
    plt.close(fig)


def save_pr_curve(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="tab:blue", lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Verifier Precision-Recall (jump class)")
    ax.grid(True); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    for thr in (0.3, 0.5, 0.7, 0.9):
        if len(thresholds):
            idx = int(np.argmin(np.abs(thresholds - thr)))
            if idx < len(recall):
                ax.scatter(recall[idx], precision[idx], s=40, zorder=5, color="tab:red")
                ax.annotate(f" thr={thr}", (recall[idx], precision[idx]), fontsize=9)
    fig.tight_layout()
    fig.savefig(CHECKPOINT_DIR / "pr_curve.png", dpi=120)
    plt.close(fig)


def report_thresholds(probs, labels):
    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>8}")
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        preds = (probs >= thr).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f = f1_score(labels, preds, zero_division=0)
        print(f"{thr:>10.2f} {p:>10.3f} {r:>10.3f} {f:>8.3f}")


# ============================================================
# Тренировка одного seed
# ============================================================

def train_one_seed(seed, windows, train_idx, val_idx, class_counts, cls_w):
    """Обучает verifier с данным seed, сохраняет в seed_{seed}/best.pt.
    Возвращает (best_probs, val_labels) для ensemble."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_dir = CHECKPOINT_DIR / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    train_ds = VerifierDataset(windows, augment=True)
    val_ds = VerifierDataset(windows, augment=False)
    train_labels = np.array([w["label"] for w in windows])[train_idx]
    weights = torch.tensor((1.0 / np.maximum(class_counts, 1))[train_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = VerifierModel(FEATURE_DIM, TEMPORAL_HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS * len(train_loader), total_steps)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=LABEL_SMOOTHING)

    history = {"loss": [], "f1": [], "precision": [], "recall": []}
    best_f1, best_eval = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        ev = eval_epoch(model, val_loader)
        history["loss"].append(loss)
        history["f1"].append(ev["f1"])
        history["precision"].append(ev["precision"])
        history["recall"].append(ev["recall"])

        if ev["f1"] > best_f1:
            best_f1, best_eval = ev["f1"], ev
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "feature_dim": FEATURE_DIM, "hidden_dim": TEMPORAL_HIDDEN_DIM,
                    "num_frames": NUM_FRAMES, "window_sec": WINDOW_SEC,
                    "smooth_sigma": SMOOTH_SIGMA, "default_threshold": DEFAULT_THRESHOLD,
                },
                "f1": ev["f1"], "precision": ev["precision"], "recall": ev["recall"], "epoch": epoch,
            }, seed_dir / "best.pt")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ep = range(1, len(history["loss"]) + 1)
    axes[0].plot(ep, history["loss"], marker="o"); axes[0].set_title(f"Loss seed {seed}"); axes[0].grid(True)
    axes[1].plot(ep, history["f1"], marker="o", label="F1")
    axes[1].plot(ep, history["precision"], marker="s", label="P")
    axes[1].plot(ep, history["recall"], marker="^", label="R")
    axes[1].set_title(f"Val seed {seed}"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(seed_dir / "history.png", dpi=120); plt.close(fig)

    print(f"  seed {seed}: best f1={best_f1:.3f}  P={best_eval['precision']:.3f}  R={best_eval['recall']:.3f}")
    return best_eval["probs"], best_eval["labels"]


# ============================================================
# Main — multi-seed ensemble
# ============================================================

def main():
    print(f"=== Verifier final (config E, multi-seed) ===  Device: {DEVICE}")

    windows = build_windows()
    precompute_pose(windows)

    labels = np.array([w["label"] for w in windows])
    train_idx, val_idx = train_test_split(
        np.arange(len(windows)), test_size=0.2, stratify=labels, random_state=42,
    )
    class_counts = np.bincount(labels[train_idx], minlength=2)
    cls_w = torch.tensor([1.0, float(class_counts[0]) / max(class_counts[1], 1)], dtype=torch.float).to(DEVICE)
    print(f"Train: {len(train_idx)} ({class_counts[1]} jump / {class_counts[0]} non-jump), Val: {len(val_idx)}")

    seed_probs = []
    val_labels = None
    for seed in SEEDS:
        probs, val_labels = train_one_seed(seed, windows, train_idx, val_idx, class_counts, cls_w)
        seed_probs.append(probs)

    # ensemble — усреднение вероятностей
    ens_probs = np.mean(seed_probs, axis=0)
    save_pr_curve(ens_probs, val_labels)

    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE ({len(SEEDS)} seeds):")
    report_thresholds(ens_probs, val_labels)
    f1, p, r = (lambda pr: (
        f1_score(val_labels, pr, zero_division=0),
        precision_score(val_labels, pr, zero_division=0),
        recall_score(val_labels, pr, zero_division=0),
    ))((ens_probs >= DEFAULT_THRESHOLD).astype(int))
    print(f"{'=' * 60}")
    print(f"Рабочая точка thr={DEFAULT_THRESHOLD}:  f1={f1:.3f}  P={p:.3f}  R={r:.3f}")
    print(f"Чекпоинты: {CHECKPOINT_DIR}/seed_{{{','.join(map(str, SEEDS))}}}/best.pt")
    print(f"PR-кривая: {CHECKPOINT_DIR / 'pr_curve.png'}")


if __name__ == "__main__":
    main()
