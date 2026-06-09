"""Визуализация keypoints на клипах прыжков.

Берёт 6 клипов из data/clips/, прогоняет YOLOv8-pose, рисует скелет
(COCO-17 — точки, соединённые линиями) и собирает один анимированный GIF:
сетка 2×3 из шести клипов.

Использование:
    python scripts/visualize_keypoints.py
    python scripts/visualize_keypoints.py --clips data/clips/3/0_4154.mp4 data/clips/4/1_547.mp4 ...
    python scripts/visualize_keypoints.py --n-frames 50 --imgsz 960 --out data/kpts.gif
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

CLIPS_DIR = _REPO_ROOT / "data" / "clips"
POSE_MODEL = "yolov8m-pose.pt"
POSE_CONF = 0.25
KP_CONF_THR = 0.3                  # порог уверенности keypoint для отрисовки

# COCO-17 скелет: пары индексов keypoints, соединяемые линией
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),              # голова
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),     # плечи + руки
    (5, 11), (6, 12), (11, 12),                  # корпус
    (11, 13), (13, 15), (12, 14), (14, 16),      # ноги
]

GRID_COLS, GRID_ROWS = 3, 2        # сетка 2×3 = 6 клипов
PANEL_W, PANEL_H = 480, 270        # 16:9 панель одного клипа


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray) -> None:
    """Рисует скелет фигуриста на кадре (BGR, in-place). kpts: (17, 3)."""
    for a, b in SKELETON:
        if kpts[a, 2] > KP_CONF_THR and kpts[b, 2] > KP_CONF_THR:
            pa = tuple(np.round(kpts[a, :2]).astype(int))
            pb = tuple(np.round(kpts[b, :2]).astype(int))
            cv2.line(frame, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
    for i in range(17):
        if kpts[i, 2] > KP_CONF_THR:
            c = tuple(np.round(kpts[i, :2]).astype(int))
            cv2.circle(frame, c, 4, (0, 128, 255), -1, cv2.LINE_AA)


def _main_person(res) -> np.ndarray | None:
    """keypoints (17,3) главного фигуриста (самый большой bbox) или None."""
    if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
        return None
    kp = res.keypoints.data.cpu().numpy()
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        person = int(areas.argmax())
    else:
        person = 0
    return kp[person]


def _letterbox_panel(frame_bgr: np.ndarray, name: str) -> np.ndarray:
    """Кадр → панель PANEL_W×PANEL_H (letterbox с сохранением пропорций) + подпись."""
    h, w = frame_bgr.shape[:2]
    r = min(PANEL_W / w, PANEL_H / h)
    nw, nh = max(int(w * r), 1), max(int(h * r), 1)
    resized = cv2.resize(frame_bgr, (nw, nh))
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    top, left = (PANEL_H - nh) // 2, (PANEL_W - nw) // 2
    panel[top:top + nh, left:left + nw] = resized
    cv2.putText(panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def process_clip(path: Path, model, device, n_frames: int, imgsz: int,
                 batch: int) -> list[np.ndarray]:
    """Клип → список кадров (BGR) с нарисованным скелетом."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(n_frames, total)).round().astype(int)
    raw = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            raw.append(fr)
    cap.release()
    if not raw:
        return []

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in raw]
    # YOLO под-батчами — иначе на занятом GPU вылетает OOM
    results = []
    for i in range(0, len(rgb), batch):
        results.extend(model(rgb[i:i + batch], verbose=False, conf=POSE_CONF,
                             imgsz=imgsz, device=device))
    out = []
    for fr, res in zip(raw, results):
        frame = fr.copy()
        kp = _main_person(res)
        if kp is not None:
            draw_skeleton(frame, kp)
        out.append(frame)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=str, nargs="+",
                        help="пути к клипам (по умолчанию 6 случайных из data/clips/)")
    parser.add_argument("--n-frames", type=int, default=40, help="кадров на клип в GIF")
    parser.add_argument("--imgsz", type=int, default=1280, help="imgsz YOLO (меньше = быстрее)")
    parser.add_argument("--batch", type=int, default=8, help="кадров за вызов YOLO (меньше при OOM)")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda:0 / cuda:1 …")
    parser.add_argument("--fps", type=float, default=12.0, help="fps итогового GIF")
    parser.add_argument("--out", type=str,
                        default=str(_REPO_ROOT / "data" / "keypoints_viz.gif"))
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.clips:
        clips = [Path(c) for c in args.clips][:6]
        if len(clips) < 6:
            parser.error("нужно ровно 6 клипов (сетка 2×3)")
    else:
        all_clips = sorted(CLIPS_DIR.glob("*/*.mp4"))
        if len(all_clips) < 6:
            parser.error(f"в {CLIPS_DIR} меньше 6 клипов")
        random.seed(42)
        clips = random.sample(all_clips, 6)

    print(f"Device: {device}, imgsz={args.imgsz}")
    for c in clips:
        print(f"  {c}")

    from ultralytics import YOLO
    model = YOLO(POSE_MODEL)
    model.to(device)

    panels_per_clip = []
    for c in clips:
        frames = process_clip(c, model, device, args.n_frames, args.imgsz, args.batch)
        if not frames:
            print(f"  ПРОПУСК (пустой клип): {c}")
            frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
        name = f"{c.parent.name}/{c.stem}"
        panels_per_clip.append([_letterbox_panel(f, name) for f in frames])

    # выравниваем число кадров по минимуму (клипы разной длины)
    n = min(len(p) for p in panels_per_clip)
    grid_frames = []
    for t in range(n):
        rows = []
        for r in range(GRID_ROWS):
            row = [panels_per_clip[r * GRID_COLS + col][t] for col in range(GRID_COLS)]
            rows.append(np.hstack(row))
        grid = np.vstack(rows)
        grid_frames.append(Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_frames[0].save(
        out, save_all=True, append_images=grid_frames[1:],
        duration=int(1000 / args.fps), loop=0,
    )
    print(f"\nGIF сохранён: {out}  ({len(grid_frames)} кадров, сетка {GRID_COLS}×{GRID_ROWS})")


if __name__ == "__main__":
    main()
