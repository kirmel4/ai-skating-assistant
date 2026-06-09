"""Оценка B3-классификатора на тренировочных данных.

Прогоняет обученный B3 по val-сплиту (тот же, что при обучении:
test_size=0.2, stratify по типу прыжка, random_state=42) и печатает:

  - метрики (F1 macro + accuracy) по 4 задачам: тип / обороты / недокрут / падение;
  - сравнение одиночного чекпойнта seed 44 (его использует сервис) с
    ансамблем из нескольких сидов (усреднение вероятностей);
  - поведение регрессионной головы rot_reg — MAE против истинного числа
    оборотов и разбивка предсказаний по каждому целому значению оборотов
    (rot_reg обучалась MSE на ЦЕЛЫХ метках — дробные на выходе это шум,
     а не настоящие пол-оборота);
  - таблицу предсказаний по каждому прыжку → checkpoints_pose_b3_final/eval_b3.csv.

Использование:
    python scripts/eval_b3.py
    python scripts/eval_b3.py --ensemble-seeds 42 43 44
    python scripts/eval_b3.py --split all          # по всему датасету, не только val
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

import scripts.train_pose_b3_final as _b3
from scripts.train_pose_b3_final import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    FEATURE_DIM,
    IMAGE_SIZE,
    LABEL_MAP,
    NUM_FRAMES,
    NUM_JUMP_CLASSES,
    POSE_CACHE_DIR,
    TARGET_FPS,
    TEMPORAL_HIDDEN_DIM,
    B3Dataset,
    B3PoseModel,
    CropClipDataset,
    DataLoaderConfig,
    VideoConfig,
    detect_and_smooth_bboxes,
    make_underrotation_map,
    map_fall,
    normalize_underrotation_value,
    precompute_pose_keypoints,
    prepare_clip_dataset,
)

TASKS = ("jump", "rot", "under", "fall")
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}


def _resolve_device() -> torch.device:
    """Устройство с учётом CUDA_VISIBLE_DEVICES. POSE_CUDA_DEVICE=1 при
    CUDA_VISIBLE_DEVICES=1 указывает на cuda:1, которого нет (единственная
    видимая карта — cuda:0) → падаем на cuda:0."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    want = int(os.environ.get("POSE_CUDA_DEVICE", "1"))
    if want < 0 or want >= count:
        want = 0
    return torch.device(f"cuda:{want}")


DEVICE = _resolve_device()
_b3.DEVICE = DEVICE   # внутренние функции b3 (bbox/pose precompute) — на то же устройство


def build_data(split: str):
    """Готовит датасет и индексы для оценки. split: 'val' (20%) или 'all'.

    Шаги повторяют train_pose_b3_final.main() — иначе метки/сплит не совпадут.
    bbox и pose-кэш переиспользуются (B3 уже обучался) — YOLO заново не гоняется.
    """
    video_config = VideoConfig(
        num_frames=NUM_FRAMES, target_fps=25.0, image_size=IMAGE_SIZE, return_meta=False,
    )
    data_config = DataLoaderConfig(
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])
    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}
    inv_rotation_map = {i: v for v, i in rotation_map.items()}
    under_values = df_clips["underrotation"].apply(normalize_underrotation_value).values
    underrotation_map = make_underrotation_map(under_values)

    print(f"Dataset: {len(df_clips)} прыжков")

    bboxes = detect_and_smooth_bboxes(df_clips)
    frame_dataset = CropClipDataset(
        df=df_clips, num_frames=NUM_FRAMES, target_fps=TARGET_FPS,
        image_size=IMAGE_SIZE, return_meta=False, bboxes=bboxes,
    )
    precompute_pose_keypoints(frame_dataset, POSE_CACHE_DIR, DEVICE)

    jump_labels = df_clips["jump_type"].map(LABEL_MAP).values
    rotation_labels = df_clips["rotations"].astype(int).map(rotation_map).values
    rotation_floats = np.array(
        [float(inv_rotation_map[c]) for c in rotation_labels], dtype=np.float32,
    )
    underrotation_labels = np.array([underrotation_map[v] for v in under_values])
    fall_labels = df_clips["fall"].apply(map_fall).values

    num_classes = {
        "jump": NUM_JUMP_CLASSES, "rot": len(rotation_map),
        "under": len(underrotation_map), "fall": 2,
    }

    dataset = B3Dataset(
        df_clips, POSE_CACHE_DIR,
        jump_labels, rotation_labels, rotation_floats, underrotation_labels, fall_labels,
        augment_temporal_roll=False,
    )

    if split == "all":
        idx = np.arange(len(df_clips))
    else:
        _, idx = train_test_split(
            np.arange(len(df_clips)), test_size=0.2,
            stratify=jump_labels, random_state=42,
        )
    idx = np.sort(idx)

    labels = {
        "jump": jump_labels, "rot": rotation_labels,
        "under": underrotation_labels, "fall": fall_labels,
    }
    return df_clips, dataset, idx, labels, rotation_floats, num_classes, inv_rotation_map


def load_model(seed: int, num_classes: dict) -> B3PoseModel:
    ckpt_path = CHECKPOINT_DIR / f"seed_{seed}" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Чекпойнт не найден: {ckpt_path}")
    model = B3PoseModel(
        feature_dim=FEATURE_DIM, hidden_dim=TEMPORAL_HIDDEN_DIM, num_classes=num_classes,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect(model: B3PoseModel, loader: DataLoader):
    """Прогон модели → softmax-вероятности по 4 задачам (N,C) + rot_reg (N,)."""
    probs = {k: [] for k in TASKS}
    rot_reg = []
    for feats, *_ in loader:
        feats = feats.to(DEVICE, non_blocking=True)
        out = model(feats)
        for k in TASKS:
            probs[k].append(torch.softmax(out[k].float(), dim=1).cpu())
        rot_reg.append(out["rot_reg"].float().cpu())
    return ({k: torch.cat(v).numpy() for k, v in probs.items()},
            torch.cat(rot_reg).numpy())


def report_metrics(name: str, probs: dict, trues: dict) -> dict:
    """F1 macro + accuracy по задачам. Возвращает словарь F1 для сравнения."""
    print(f"\n  {name}")
    print(f"  {'задача':<8} {'F1':>8} {'accuracy':>10}")
    f1s = {}
    for k in TASKS:
        pred = probs[k].argmax(1)
        f1 = f1_score(trues[k], pred, average="macro", zero_division=0)
        acc = accuracy_score(trues[k], pred)
        f1s[k] = f1
        print(f"  {k:<8} {f1:>8.3f} {acc:>10.3f}")
    return f1s


def report_rot_reg(name: str, rot_reg: np.ndarray, rot_true: np.ndarray) -> None:
    """Поведение регрессионной головы: MAE + разбивка по истинным оборотам."""
    mae = float(np.abs(rot_reg - rot_true).mean())
    rounded_acc = float((np.round(rot_reg) == rot_true).mean())
    print(f"\n  rot_reg [{name}]:  MAE={mae:.3f} оборота,  "
          f"округление→целое acc={rounded_acc:.3f}")
    for val in sorted(np.unique(rot_true)):
        m = rot_true == val
        v = rot_reg[m]
        print(f"    истинно {int(val)} об (n={int(m.sum()):3d}): "
              f"rot_reg = {v.mean():.2f} ± {v.std():.2f}  "
              f"[мин {v.min():.2f}, макс {v.max():.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-seed", type=int, default=45,
                        help="одиночный чекпойнт (его использует сервис)")
    parser.add_argument("--ensemble-seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="сиды для ансамбля (усреднение вероятностей)")
    parser.add_argument("--split", choices=["val", "all"], default="val",
                        help="val = 20%% held-out (как при обучении), all = весь датасет")
    args = parser.parse_args()

    print(f"=== Оценка B3 === Device: {DEVICE}, split: {args.split}")
    df_clips, dataset, idx, labels, rotation_floats, num_classes, inv_rotation_map = \
        build_data(args.split)
    print(f"Прыжков для оценки: {len(idx)}")

    loader = DataLoader(
        Subset(dataset, idx.tolist()), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    trues = {k: labels[k][idx] for k in TASKS}
    rot_true = rotation_floats[idx]

    # --- одиночный чекпойнт seed 44 ---
    single = load_model(args.single_seed, num_classes)
    probs_single, rot_reg_single = collect(single, loader)
    del single
    torch.cuda.empty_cache()

    # --- ансамбль: усреднение softmax-вероятностей + усреднение rot_reg ---
    ens_probs = {k: np.zeros_like(probs_single[k]) for k in TASKS}
    ens_rot_reg = np.zeros_like(rot_reg_single)
    for seed in args.ensemble_seeds:
        model = load_model(seed, num_classes)
        p, rr = collect(model, loader)
        for k in TASKS:
            ens_probs[k] += p[k]
        ens_rot_reg += rr
        del model
        torch.cuda.empty_cache()
    for k in TASKS:
        ens_probs[k] /= len(args.ensemble_seeds)
    ens_rot_reg /= len(args.ensemble_seeds)

    print(f"\n{'=' * 64}\nМЕТРИКИ КЛАССИФИКАЦИИ\n{'=' * 64}")
    f1_single = report_metrics(f"одиночный seed {args.single_seed}", probs_single, trues)
    f1_ens = report_metrics(f"ансамбль {args.ensemble_seeds}", ens_probs, trues)

    print(f"\n  Δ ансамбль − одиночный:")
    for k in TASKS:
        d = f1_ens[k] - f1_single[k]
        print(f"    {k:<8} {d:+.3f}")

    print(f"\n{'=' * 64}\nРЕГРЕССИЯ ОБОРОТОВ (rot_reg)\n{'=' * 64}")
    report_rot_reg(f"одиночный seed {args.single_seed}", rot_reg_single, rot_true)
    report_rot_reg(f"ансамбль {args.ensemble_seeds}", ens_rot_reg, rot_true)

    # --- таблица по каждому прыжку ---
    sub = df_clips.iloc[idx].reset_index(drop=True)
    table = pd.DataFrame({
        "clip": sub["clip_path"].apply(lambda p: Path(str(p)).name),
        "jump_true": [INV_LABEL[j] for j in trues["jump"]],
        "jump_pred": [INV_LABEL[j] for j in ens_probs["jump"].argmax(1)],
        "rot_true": rot_true.astype(int),
        f"rot_reg_s{args.single_seed}": np.round(rot_reg_single, 2),
        "rot_reg_ens": np.round(ens_rot_reg, 2),
        "rot_cls_ens": [inv_rotation_map[c] for c in ens_probs["rot"].argmax(1)],
        "under_true": trues["under"],
        "under_pred": ens_probs["under"].argmax(1),
        "fall_true": trues["fall"],
        "fall_pred": ens_probs["fall"].argmax(1),
    })
    out_csv = CHECKPOINT_DIR / "eval_b3.csv"
    table.to_csv(out_csv, index=False)
    print(f"\n{'=' * 64}\nТАБЛИЦА ПО ПРЫЖКАМ (первые 25 из {len(table)})\n{'=' * 64}")
    print(table.head(25).to_string(index=False))
    print(f"\nПолная таблица: {out_csv}")


if __name__ == "__main__":
    main()
