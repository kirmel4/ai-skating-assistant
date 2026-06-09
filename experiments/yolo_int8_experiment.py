"""Эксперимент: теряет ли точность YOLO-pose при INT8-квантизации.

INT8-квантованная ONNX-модель быстрее на CPU, но «огрубляет» веса. Скрипт
квантует yolov8m-pose (модель rotation-прохода), переизвлекает keypoints
валидационных окон verifier'а двумя моделями (FP32-ONNX и INT8-ONNX) и
сравнивает метрики verifier'а — деградирует ли детекция.

Шаги:
  1. FP32 ONNX-экспорт yolov8m-pose (если ещё нет);
  2. калибровка на кадрах видео + статическая INT8-квантизация (onnxruntime);
  3. переизвлечение MAX_WINDOWS валидационных окон обеими моделями;
  4. verifier-ансамбль на обоих наборах → сравнение P/R/F1 + ошибка keypoints.

ВНИМАНИЕ: шаг 3 гоняет YOLO по окнам на CPU — медленно. MAX_WINDOWS можно
снизить, если долго (метрика станет шумнее, но тенденция видна).

Использование:
    python experiments/yolo_int8_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.detect_jumps import POSE_IMGSZ, POSE_MODEL, _extract_window_kpts
from scripts.train_verifier import (
    CHECKPOINT_DIR,
    FEATURE_DIM,
    SEEDS,
    SMOOTH_SIGMA,
    TEMPORAL_HIDDEN_DIM,
    VIDEO_FILES,
    VIDEOS_DIR,
    VerifierModel,
    build_windows,
    extract_features_f4,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP32_ONNX = _REPO_ROOT / "yolov8m-pose.onnx"
INT8_ONNX = _REPO_ROOT / "yolov8m-pose-int8.onnx"
N_CALIB = 96             # кадров на калибровку INT8
MAX_WINDOWS = 150        # сколько валид. окон переизвлечь (снизь, если долго)
RNG = np.random.default_rng(42)


def letterbox(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Кадр BGR → (1,3,size,size) float32 [0,1] RGB с letterbox-паддингом."""
    h, w = frame_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame_bgr, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]


def export_fp32():
    if FP32_ONNX.is_file():
        print(f"FP32 ONNX уже есть: {FP32_ONNX}")
        return
    from ultralytics import YOLO
    print("Экспорт FP32 ONNX...")
    YOLO(POSE_MODEL).export(format="onnx", dynamic=True, imgsz=POSE_IMGSZ)


def quantize_int8():
    """Статическая INT8-квантизация FP32 ONNX через onnxruntime."""
    if INT8_ONNX.is_file():
        print(f"INT8 ONNX уже есть: {INT8_ONNX}")
        return

    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    input_name = onnx.load(str(FP32_ONNX)).graph.input[0].name
    video_path = str((VIDEOS_DIR / VIDEO_FILES[10]).resolve())

    class FrameReader(CalibrationDataReader):
        """Лениво отдаёт N_CALIB кадров видео для калибровки активаций."""

        def __init__(self):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self._idxs = list(np.linspace(0, total - 1, N_CALIB).astype(int))
            self._pos = 0
            self._cap = cv2.VideoCapture(video_path)

        def get_next(self):
            while self._pos < len(self._idxs):
                i = self._idxs[self._pos]
                self._pos += 1
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = self._cap.read()
                if ok:
                    return {input_name: letterbox(fr, POSE_IMGSZ)}
            self._cap.release()
            return None

    print(f"INT8-квантизация (калибровка на {N_CALIB} кадрах)...")
    quantize_static(
        str(FP32_ONNX),
        str(INT8_ONNX),
        FrameReader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 ONNX: {INT8_ONNX}")


@torch.no_grad()
def ensemble_probs(models, kpts_list) -> np.ndarray:
    feats = torch.stack([
        extract_features_f4(k, smooth_sigma=SMOOTH_SIGMA) for k in kpts_list
    ]).to(DEVICE)
    seed_probs = [torch.softmax(m(feats), dim=-1)[:, 1].cpu().numpy() for m in models]
    return np.mean(seed_probs, axis=0)


def metrics_table(name: str, probs: np.ndarray, labels: np.ndarray) -> float:
    print(f"\n  {name}")
    print(f"  {'thr':>6} {'P':>8} {'R':>8} {'F1':>8}")
    best_f1, best = -1.0, None
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        preds = (probs >= thr).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f = f1_score(labels, preds, zero_division=0)
        print(f"  {thr:>6.2f} {p:>8.3f} {r:>8.3f} {f:>8.3f}")
        if f > best_f1:
            best_f1, best = f, (thr, p, r)
    print(f"  -> лучший F1={best_f1:.3f} @ thr={best[0]:.2f} (P={best[1]:.3f} R={best[2]:.3f})")
    return best_f1


def main():
    print(f"Device: {DEVICE}")
    export_fp32()
    quantize_int8()

    from ultralytics import YOLO
    yolo_fp32 = YOLO(str(FP32_ONNX))
    yolo_int8 = YOLO(str(INT8_ONNX))

    windows = build_windows()
    labels_all = np.array([w["label"] for w in windows])
    _, val_idx = train_test_split(
        np.arange(len(windows)), test_size=0.2, stratify=labels_all, random_state=42,
    )
    if len(val_idx) > MAX_WINDOWS:
        val_idx = RNG.choice(val_idx, MAX_WINDOWS, replace=False)
    print(f"Окон для переизвлечения: {len(val_idx)}")

    # verifier-ансамбль
    models = []
    for seed in SEEDS:
        model = VerifierModel(FEATURE_DIM, TEMPORAL_HIDDEN_DIM).to(DEVICE)
        ckpt = torch.load(CHECKPOINT_DIR / f"seed_{seed}" / "best.pt",
                          map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    caps: dict[str, cv2.VideoCapture] = {}

    def _cap(path: str) -> cv2.VideoCapture:
        if path not in caps:
            caps[path] = cv2.VideoCapture(path)
        return caps[path]

    kp_fp32, kp_int8, val_labels, kp_err = [], [], [], []
    for i in tqdm(val_idx, desc="re-extract"):
        w = windows[i]
        cap = _cap(w["video_path"])
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        a = _extract_window_kpts(cap, w["center_sec"], src_fps, total, yolo_fp32, DEVICE)
        b = _extract_window_kpts(cap, w["center_sec"], src_fps, total, yolo_int8, DEVICE)
        kp_fp32.append(a)
        kp_int8.append(b)
        val_labels.append(w["label"])
        vis = a[..., 2] > 0.2   # ошибка координат только по видимым keypoints
        if vis.any():
            kp_err.append((a[..., :2][vis] - b[..., :2][vis]).abs().mean().item())
    for c in caps.values():
        c.release()

    val_labels = np.array(val_labels)
    print(f"jump / non-jump: {int(val_labels.sum())} / {int((val_labels == 0).sum())}")
    print(f"Средняя ошибка координат keypoints FP32 vs INT8: {np.mean(kp_err):.2f} px")

    f1_fp32 = metrics_table("FP32 ONNX", ensemble_probs(models, kp_fp32), val_labels)
    f1_int8 = metrics_table("INT8 ONNX", ensemble_probs(models, kp_int8), val_labels)

    print(f"\n{'=' * 48}")
    print("  ИТОГ (лучший F1 по порогам):")
    print(f"    FP32 ONNX   {f1_fp32:.3f}")
    print(f"    INT8 ONNX   {f1_int8:.3f}   Δ {f1_int8 - f1_fp32:+.3f}")
    print('=' * 48)
    print("Маленькая Δ -> INT8 безопасен, ускорение почти даром.")
    print("Заметная просадка -> INT8 не стоит, либо нужна лучшая калибровка.")


if __name__ == "__main__":
    main()
