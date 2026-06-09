"""Crop-based pose-экстрактор для B3-классификатора.

Воспроизводит препроцессинг, на котором обучался B3 (train_pose_b3_final):
  окно прыжка → 64 кадра (target_fps=16, расширение до ~4 сек контекста) →
  YOLOv8m детект фигуриста → bbox + padding 0.30 + сглаживание σ=2.0 →
  кроп + ресайз 224×224 → YOLOv8x-pose → (64, 17, 3) keypoints.

Кроп критичен: B3 обучался на pose с кропа, а не с полного кадра.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from scripts.train_pose_b3_final import (
    BBOX_PADDING,
    BBOX_SMOOTH_SIGMA,
    IMAGE_SIZE,
    NUM_FRAMES,
    TARGET_FPS,
)
from scripts.train_pose_temporal import POSE_CONF, POSE_IMGSZ, POSE_MODEL
from scripts.train_videomae_phase1 import (
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL,
    _build_indices_static,
    _expand_bbox,
    _interpolate_bboxes,
)
from src.ml_service.yolo_loader import load_yolo


def _largest_person(res) -> int | None:
    """Индекс человека с самым большим bbox в результате YOLO (None если нет)."""
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(areas.argmax())


def _crop_resize(frame: np.ndarray, bbox, image_size: int) -> np.ndarray:
    """Кроп кадра по bbox + ресайз в image_size² с letterbox-паддингом."""
    h, w = frame.shape[:2]
    if bbox is None:
        crop = frame
    else:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame

    ch, cw = crop.shape[:2]
    scale = min(image_size / ch, image_size / cw)
    nh, nw = max(1, round(ch * scale)), max(1, round(cw * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

    out = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top, left = (image_size - nh) // 2, (image_size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


class CropPoseExtractor:
    """Извлекает (64,17,3) keypoints для окна прыжка — как в обучении B3."""

    def __init__(self, device: torch.device):
        self.device = device
        # детект фигуриста по полному кадру + keypoints по кропу
        self.detector = load_yolo(YOLO_MODEL, device, imgsz=YOLO_IMGSZ)
        self.pose = load_yolo(POSE_MODEL, device, imgsz=POSE_IMGSZ)

    @torch.no_grad()
    def extract_window(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        src_fps: float,
        total_frames: int,
    ) -> torch.Tensor:
        """Окно прыжка [start_sec, end_sec] → (NUM_FRAMES, 17, 3) keypoints."""
        indices = _build_indices_static(
            start_sec, end_sec, src_fps, total_frames, NUM_FRAMES, TARGET_FPS,
        )

        # --- читаем 64 кадра окна ---
        cap = cv2.VideoCapture(str(video_path))
        frames: list = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, fr = cap.read()
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ret else None)
        cap.release()

        kpts = torch.zeros(NUM_FRAMES, 17, 3, dtype=torch.float32)
        valid = [(i, f) for i, f in enumerate(frames) if f is not None]
        if not valid:
            return kpts

        # --- bbox фигуриста по полному кадру (YOLOv8m детект) ---
        vidx, vimgs = zip(*valid)
        det = self.detector(
            list(vimgs), classes=[0], verbose=False,
            conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=self.device,
        )
        h, w = vimgs[0].shape[:2]
        per_frame: list = [None] * NUM_FRAMES
        for vi, res in zip(vidx, det):
            p = _largest_person(res)
            if p is None:
                continue
            box = res.boxes.xyxy[p].cpu().numpy().tolist()
            per_frame[vi] = _expand_bbox(box, h, w, padding=BBOX_PADDING)

        bboxes = None
        if not all(b is None for b in per_frame):
            bboxes = _interpolate_bboxes(per_frame)
            if BBOX_SMOOTH_SIGMA > 0:
                arr = np.array(bboxes, dtype=np.float32)
                bboxes = gaussian_filter1d(
                    arr, sigma=BBOX_SMOOTH_SIGMA, axis=0, mode="nearest",
                ).tolist()

        # --- кроп + ресайз каждого кадра ---
        crops = np.zeros((NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for i in range(NUM_FRAMES):
            if frames[i] is None:
                continue
            crops[i] = _crop_resize(frames[i], bboxes[i] if bboxes else None, IMAGE_SIZE)

        # --- pose по кропам (YOLOv8x-pose) ---
        pose_res = self.pose(
            list(crops), verbose=False,
            imgsz=POSE_IMGSZ, conf=POSE_CONF, device=self.device,
        )
        for t, res in enumerate(pose_res):
            if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
                continue
            p = _largest_person(res)
            kpts[t] = res.keypoints.data[p if p is not None else 0].cpu().float()

        return kpts
