"""Эксперимент: деградирует ли verifier на прорежённых окнах (reuse rotation-кэша).

Оптимизация #1 (reuse): не гонять YOLO повторно для verifier'а, а брать
keypoints из кэша rotation-прохода. Но rotation-проход семплит ~12.5 fps, а
verifier обучался на окнах ~25 fps (64 кадра подряд на нативной частоте).
Скрипт показывает, насколько просядут метрики, если кормить его 12.5-fps окнами.

Берёт готовый pose-кэш data/verifier_pose/ (25 fps) и прореживает его —
YOLO заново НЕ гоняется. Сравнивает на валидационном сплите:
  A          — 25 fps (как при обучении);
  B-nearest  — 12.5 fps, ресемпл до 64 индексным выбором (дубли кадров);
  B-interp   — 12.5 fps, ресемпл до 64 линейной интерполяцией.

Использование:
    python experiments/verifier_fps_reuse_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from scripts.train_verifier import (
    CHECKPOINT_DIR,
    FEATURE_DIM,
    SEEDS,
    SMOOTH_SIGMA,
    TEMPORAL_HIDDEN_DIM,
    VerifierModel,
    _window_cache_path,
    build_windows,
    extract_features_f4,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FRAMES = 64


def downsample_nearest(kpts: torch.Tensor) -> torch.Tensor:
    """25 fps (64 кадра) → 12.5 fps (каждый 2-й, ~32) → 64 индексным выбором."""
    coarse = kpts[::2]
    idx = np.linspace(0, len(coarse) - 1, NUM_FRAMES).round().astype(int)
    return coarse[idx]


def downsample_interp(kpts: torch.Tensor) -> torch.Tensor:
    """25 fps (64 кадра) → 12.5 fps (~32) → 64 линейной интерполяцией."""
    coarse = kpts[::2].numpy()
    n = len(coarse)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, NUM_FRAMES)
    out = np.empty((NUM_FRAMES, 17, 3), dtype=np.float32)
    for j in range(17):
        for c in range(3):
            out[:, j, c] = np.interp(dst, src, coarse[:, j, c])
    return torch.from_numpy(out)


@torch.no_grad()
def ensemble_probs(models, kpts_list) -> np.ndarray:
    """Список окон keypoints → усреднённая по сидам P(jump)."""
    feats = torch.stack([
        extract_features_f4(k, smooth_sigma=SMOOTH_SIGMA) for k in kpts_list
    ]).to(DEVICE)
    seed_probs = []
    for model in models:
        logits = model(feats)
        seed_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.mean(seed_probs, axis=0)


def metrics_table(name: str, probs: np.ndarray, labels: np.ndarray) -> float:
    print(f"\n  {name}")
    print(f"  {'thr':>6} {'P':>8} {'R':>8} {'F1':>8}")
    best_f1, best = -1.0, None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        preds = (probs >= thr).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f = f1_score(labels, preds, zero_division=0)
        print(f"  {thr:>6.2f} {p:>8.3f} {r:>8.3f} {f:>8.3f}")
        if f > best_f1:
            best_f1, best = f, (thr, p, r)
    print(f"  -> лучший F1={best_f1:.3f} @ thr={best[0]:.2f}  (P={best[1]:.3f} R={best[2]:.3f})")
    return best_f1


def main():
    print(f"Device: {DEVICE}")
    windows = build_windows()
    labels_all = np.array([w["label"] for w in windows])
    _, val_idx = train_test_split(
        np.arange(len(windows)), test_size=0.2, stratify=labels_all, random_state=42,
    )
    print(f"Валидационных окон: {len(val_idx)}")

    # ансамбль verifier'ов
    models = []
    for seed in SEEDS:
        ckpt_path = CHECKPOINT_DIR / f"seed_{seed}" / "best.pt"
        model = VerifierModel(FEATURE_DIM, TEMPORAL_HIDDEN_DIM).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
    print(f"Verifier-ансамбль: {len(models)} сидов")

    # 25fps-кэш + два прорежённых варианта
    kp_25, kp_near, kp_interp, val_labels = [], [], [], []
    for i in val_idx:
        w = windows[i]
        cache = _window_cache_path(w)
        if not cache.is_file():
            continue
        kpts = torch.load(cache, weights_only=True).float()
        kp_25.append(kpts)
        kp_near.append(downsample_nearest(kpts))
        kp_interp.append(downsample_interp(kpts))
        val_labels.append(w["label"])
    val_labels = np.array(val_labels)
    print(f"Окон с pose-кэшем: {len(val_labels)} "
          f"({int(val_labels.sum())} jump / {int((val_labels == 0).sum())} non-jump)")

    f1_25 = metrics_table("A  — 25 fps (как при обучении)", ensemble_probs(models, kp_25), val_labels)
    f1_near = metrics_table("B  — 12.5 fps, индексный ресемпл", ensemble_probs(models, kp_near), val_labels)
    f1_interp = metrics_table("B  — 12.5 fps, интерполяция", ensemble_probs(models, kp_interp), val_labels)

    print(f"\n{'=' * 54}")
    print("  ИТОГ (лучший F1 по порогам):")
    print(f"    A  25 fps                {f1_25:.3f}")
    print(f"    B  12.5 fps  nearest     {f1_near:.3f}   Δ {f1_near - f1_25:+.3f}")
    print(f"    B  12.5 fps  interp      {f1_interp:.3f}   Δ {f1_interp - f1_25:+.3f}")
    print('=' * 54)
    print("Маленькая Δ -> reuse кэша можно делать без переобучения verifier'а.")
    print("Заметная просадка -> нужен retrain на 12.5-fps окнах (train == inference).")


if __name__ == "__main__":
    main()
