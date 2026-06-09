from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class VideoPreprocessor:
    def __init__(
        self,
        num_frames: int,
        target_fps: float,
        image_size: int,
    ):
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.image_size = image_size

    def __call__(self, video_path: Path) -> tuple[torch.Tensor, float]:
        """
        Returns:
            frames: torch.Tensor [1, T, C, H, W], float32, range [0, 1]
            duration_sec: float
        """
        fps, total_frames = self._read_video_meta(video_path)
        duration_sec = total_frames / fps

        frame_indices = self._build_indices(
            duration_sec=duration_sec,
            fps=fps,
            total_frames=total_frames,
        )

        frames = self._read_frames(video_path, frame_indices)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames_tensor = self._resize_with_pad(frames_tensor)
        frames_tensor = frames_tensor.unsqueeze(0)  # [1, T, C, H, W]

        return frames_tensor, duration_sec

    def _read_video_meta(self, video_path: Path) -> tuple[float, int]:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                raise ValueError(f"Invalid FPS: {fps}")

            if total_frames <= 0:
                raise ValueError(f"Invalid frame count: {total_frames}")

            return fps, total_frames
        finally:
            cap.release()

    def _build_indices(
        self,
        duration_sec: float,
        fps: float,
        total_frames: int,
    ) -> np.ndarray:
        """
        Для backend-сценария считаем, что весь загруженный файл — это один клип прыжка.
        Поэтому равномерно семплируем кадры по всей длительности видео.
        """
        if duration_sec <= 0:
            raise ValueError("Video duration must be positive")

        timestamps = np.linspace(
            0.0,
            duration_sec,
            num=self.num_frames,
            endpoint=False,
            dtype=np.float64,
        )

        indices = np.round(timestamps * fps).astype(np.int64)
        return np.clip(indices, 0, total_frames - 1)

    def _read_frames(
        self,
        video_path: Path,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        last_good: np.ndarray | None = None

        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()

                if ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_good = frame
                elif last_good is not None:
                    frame = last_good
                else:
                    raise ValueError(f"Cannot read frame {idx} from {video_path}")

                frames.append(frame)

            return np.stack(frames, axis=0)
        finally:
            cap.release()

    def _resize_with_pad(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [T, C, H, W], uint8
        returns: [T, C, image_size, image_size], float32 in [0, 1]
        """
        frames = frames.float() / 255.0

        _, _, h, w = frames.shape
        scale = min(self.image_size / h, self.image_size / w)

        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        frames = F.interpolate(
            frames,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(
            frames,
            (pad_left, pad_right, pad_top, pad_bottom),
        )
