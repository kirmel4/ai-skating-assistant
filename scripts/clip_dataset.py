from __future__ import annotations

import contextlib
import json
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import DataLoaderConfig, VideoConfig
from src.dataset import set_global_seed, seed_worker, make_generator, SEED
from src.utils.xlsx_parser import parse_excel

BASE_DIR = Path(__file__).parent.parent
CLIPS_DIR = BASE_DIR / "data" / "clips"
CLIP_OFFSETS_PATH = BASE_DIR / "data" / "clip_offsets.json"
TIMECODES_PATH = BASE_DIR / "data" / "Разметка прыжков.xlsx"

LABEL_MAP = {"T": 0, "S": 1, "Lo": 2, "F": 3, "Lz": 4, "A": 5}


def _repo_relative_posix(repo_rel: object) -> Path:
    return Path(str(repo_rel).strip().replace("\\", "/"))


@contextlib.contextmanager
def _silence_libav_stderr():
    if os.environ.get("CLIP_DECODE_VERBOSE", "").strip():
        yield
        return
    dn = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(old)
        os.close(dn)


class ClipDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_frames: int = 32,
        target_fps: float = 25.0,
        image_size: int = 224,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.image_size = image_size
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def _build_indices(self, start_sec: float, end_sec: float, fps: float, total_frames: int) -> np.ndarray:
        segment_duration = max(end_sec - start_sec, 1e-6)
        target_duration = self.num_frames / self.target_fps

        if segment_duration >= target_duration:
            window_start, window_end = start_sec, end_sec
        else:
            center = (start_sec + end_sec) / 2.0
            half = target_duration / 2.0
            window_start = min(center - half, start_sec)
            window_end = max(center + half, end_sec)

        timestamps = np.linspace(window_start, window_end, num=self.num_frames, endpoint=True)
        indices = np.round(timestamps * fps).astype(np.int64)
        return np.clip(indices, 0, total_frames - 1)

    def _read_frames(self, clip_path: str, frame_indices: np.ndarray) -> np.ndarray:
        with _silence_libav_stderr():
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open: {clip_path}")

            frames = []
            last_good = None
            try:
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        last_good = frame
                    elif last_good is not None:
                        frame = last_good
                    else:
                        raise ValueError(f"Cannot read frame {idx} from {clip_path}")
                    frames.append(frame)
            finally:
                cap.release()
        return np.stack(frames, axis=0)

    def _resize_with_pad(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float() / 255.0
        _, _, h, w = frames.shape
        scale = min(self.image_size / h, self.image_size / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h, pad_w = self.image_size - new_h, self.image_size - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        frames = F.pad(frames, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))
        return frames

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

        frames = self._read_frames(clip_path, frame_indices)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = self._resize_with_pad(frames)

        label = LABEL_MAP.get(row["jump_type"], -1)

        if not self.return_meta:
            return frames, label

        return frames, label, {
            "video_id": row["video_id"],
            "jump_id": row["jump_id"],
            "jump_type": row["jump_type"],
            "clip_path": row["clip_path"],
            "start_sec_in_clip": float(row["start_sec_in_clip"]),
            "end_sec_in_clip": float(row["end_sec_in_clip"]),
        }


def prepare_clip_dataset(
    video_config: VideoConfig,
    data_config: DataLoaderConfig,
    include_videos: list[int] | None = None,
    exclude_videos: list[int] | None = None,
) -> tuple[pd.DataFrame, ClipDataset, DataLoader]:

    if not CLIP_OFFSETS_PATH.exists():
        raise FileNotFoundError(f"{CLIP_OFFSETS_PATH} not found. Run: python -m scripts.clip_downloader")

    with open(CLIP_OFFSETS_PATH, encoding="utf-8") as f:
        clip_offsets: dict = json.load(f)

    df = parse_excel(TIMECODES_PATH)

    if include_videos:
        df = df[df["video_id"].isin(include_videos)]
    if exclude_videos:
        df = df[~df["video_id"].isin(exclude_videos)]

    df = df.reset_index(drop=True)

    def to_seconds(t):
        if pd.isna(t):
            return np.nan
        return t.hour * 3600 + t.minute * 60 + t.second

    rows, missing = [], []
    for _, row in df.iterrows():
        t_start_int = int(to_seconds(row["t_start_val"]))
        key = f"{int(row['video_id'])}_{int(row['jump_id'])}_{t_start_int}"
        if key not in clip_offsets:
            missing.append(key)
            continue

        meta = clip_offsets[key]
        clip_start = float(meta["clip_start"])
        t_start = to_seconds(row["t_start_val"])
        t_end = to_seconds(row["t_end_val"]) + 1.0

        row = row.copy()
        row["clip_path"] = _repo_relative_posix(meta["clip_path"]).as_posix()
        row["start_sec_in_clip"] = t_start - clip_start
        row["end_sec_in_clip"] = t_end - clip_start
        rows.append(row)

    if missing:
        print(f"Missing clips for {len(missing)} jumps: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    df_clips = pd.DataFrame(rows).reset_index(drop=True)
    print(f"Dataset: {len(df_clips)} jumps, {df_clips['video_id'].nunique()} videos")

    set_global_seed(SEED)

    dataset = ClipDataset(
        df=df_clips,
        num_frames=video_config.num_frames,
        target_fps=video_config.target_fps,
        image_size=video_config.image_size,
        return_meta=video_config.return_meta,
    )

    multiproc = data_config.num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers and multiproc,
        prefetch_factor=data_config.prefetch_factor if multiproc else None,
        worker_init_fn=partial(seed_worker, base_seed=SEED),
        generator=make_generator(SEED),
    )

    return df_clips, dataset, loader
