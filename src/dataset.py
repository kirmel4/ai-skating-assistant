import datetime
from functools import partial
import json
from pathlib import Path
import random
import time
from typing import Type
import numpy as np
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import DataLoaderConfig, VideoConfig
from src.utils.xlsx_parser import parse_excel


import logging

logger = logging.getLogger("dataset-builder")

VIDEOS_CONFIG_PATH = Path("data") / "video_configs.json"
VIDEOS_FOLDER_PATH = Path("data") / "videos"
TIMECODES_EXCEL_PATH = Path("data") / "Разметка прыжков.xlsx"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


SEED = 420

set_global_seed(SEED)


class _JumpDatasetBase(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        video_paths: dict,
        num_frames: int = 32,
        target_fps: float = 25.0,
        image_size: int = 224,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.image_size = image_size
        self.return_meta = return_meta

        self._video_info = {}
        self._caps = {}

        self.df["start_sec"] = self.df["t_start_val"].apply(self._time_to_seconds)
        self.df["end_sec"] = self.df["t_end_val"].apply(self._time_to_seconds) + 1.0

        self.jump_type_map = {
            "T": 0,
            "S": 1,
            "Lo": 2,
            "F": 3,
            "Lz": 4,
            "A": 5,
        }

        self.rotations_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
        }

        self.underrotation_map = {
            "clean": 0,
            "ur": 1,
        }

        self.fall_map = {
            0: 0,
            1: 1,
        }

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _time_to_seconds(t):
        if pd.isna(t):
            return np.nan
        return t.hour * 3600 + t.minute * 60 + t.second + getattr(t, "microsecond", 0) / 1e6

    def _get_video_info(self, video_id):
        # print(f"{self._video_info = }")
        # print(f"{video_id = }")
        if video_id in self._video_info:
            return self._video_info[video_id]

        path = self.video_paths[int(video_id)]
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            raise ValueError(f"Invalid FPS for video: {path}")

        info = {
            "path": path,
            "fps": fps,
            "total_frames": total_frames,
        }
        self._video_info[video_id] = info
        return info

    def _get_cap(self, video_id):
        if video_id in self._caps:
            return self._caps[video_id]

        path = self.video_paths[video_id]
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        self._caps[video_id] = {
            "cap": cap,
            "next_frame": None,
        }
        return self._caps[video_id]


    def _save_frames_to_video(self, frames, output_path, fps):
        h, w = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (w, h),
        )

        if not writer.isOpened():
            raise ValueError(f"Cannot open video writer: {output_path}")

        for frame in frames:
            # если frame grayscale
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)

        writer.release()


    def _build_indices(self, start_sec, end_sec, source_fps, total_frames):
        if np.isnan(start_sec) or np.isnan(end_sec):
            raise ValueError("start_sec or end_sec is NaN")

        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec

        segment_duration = max(end_sec - start_sec, 1e-6)
        # print(f"{segment_duration = }")
        target_duration = self.num_frames / self.target_fps
        # print(f"{target_duration = }")

        if segment_duration >= target_duration:
            window_start = start_sec
            window_end = end_sec
        else:
            center = (start_sec + end_sec) / 2.0
            half_target = target_duration / 2.0
            window_start = center - half_target
            window_end = center + half_target

            window_start = min(window_start, start_sec)
            window_end = max(window_end, end_sec)
        # print(f"{window_start = }")
        # print(f"{window_end = }")

        timestamps = np.linspace(
            window_start,
            window_end,
            num=self.num_frames,
            endpoint=True,
            dtype=np.float64,
        )
        # print(f"{timestamps = }")
        # print(f"{source_fps = }")

        indices = np.round(timestamps * source_fps).astype(np.int64)
        # print(f"{indices = }")
        indices = np.clip(indices, 0, total_frames - 1)
        # print(f"{indices = }")
        return indices

    def _read_frames(self, video_id, frame_indices):
        state = self._get_cap(video_id)
        cap = state["cap"]
        next_frame = state["next_frame"]

        frames = []

        for idx in frame_indices:
            idx = int(idx)

            if next_frame is None or idx != next_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = cap.read()

            if not ret:
                if frames:
                    frame = frames[-1]
                else:
                    raise ValueError(f"Cannot read frame {idx} from video_id={video_id}")
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
            next_frame = idx + 1

        state["next_frame"] = next_frame
        return np.stack(frames, axis=0)

    def _resize_with_pad(self, frames):
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

        frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))
        return frames
    
    def _get_jumps_videos_path(self, jump_df_row: pd.Series):
        video_artifacts_path = str(VIDEOS_FOLDER_PATH / "jumps" / f"{jump_df_row['video_id']}__{str(jump_df_row['t_start_val']).replace(':', '-')}__frames-{self.num_frames}__fps-{self.target_fps}.mp4")
        return video_artifacts_path

    def __getitem__(self, idx):
        raise NotImplementedError()

    def close(self):
        for state in self._caps.values():
            try:
                state["cap"].release()
            except Exception:
                pass
        self._caps.clear()

    def __del__(self):
        self.close()


class JumpDataset(_JumpDatasetBase):

    def __getitem__(self, idx):
        print(f"{idx = }")
        row = self.df.iloc[idx]
        video_id = row["video_id"]

        info = self._get_video_info(video_id)

        frame_indices = self._build_indices(
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            source_fps=info["fps"],
            total_frames=info["total_frames"],
        )

        frames = self._read_frames(video_id, frame_indices)

        self._save_frames_to_video(frames, self._get_jumps_videos_path(row), fps=info["fps"])

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = self._resize_with_pad(frames)

        output = {
            "frames": frames,
            "video_id": video_id,
            "jump_type": self.jump_type_map[row.get("jump_type")],
            "rotations": self.rotations_map[row.get("rotations")],
            "underrotation": self.underrotation_map[row.get("underrotation")],
            "fall": self.fall_map[row.get("fall")],
            "start_time": str(row["t_start_val"]),
            "end_time": str(row["t_end_val"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "indices": frame_indices,
            "path": info["path"],
            "fps": info["fps"],
            "total_frames": info["total_frames"]
        }

        return output


class JumpDatasetNoDL(JumpDataset):

    def _extract_basic_video_features(self, frames):
        if len(frames) < 2:
            return np.zeros(12)

        frames = np.array(frames)

        duration_frames = len(frames)

        brightness_mean = frames.mean()
        brightness_std = frames.std()

        contrast_mean = np.mean([f.std() for f in frames])
        contrast_std = np.std([f.std() for f in frames])

        diffs = []

        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            diffs.append(diff.mean())

        diffs = np.array(diffs)

        motion_mean = diffs.mean()
        motion_std = diffs.std()
        motion_max = diffs.max()
        motion_min = diffs.min()

        thirds = np.array_split(diffs, 3)

        motion_start = thirds[0].mean() if len(thirds[0]) else 0
        motion_middle = thirds[1].mean() if len(thirds[1]) else 0
        motion_end = thirds[2].mean() if len(thirds[2]) else 0

        return np.array([
            duration_frames,
            brightness_mean,
            brightness_std,
            contrast_mean,
            contrast_std,
            motion_mean,
            motion_std,
            motion_max,
            motion_min,
            motion_start,
            motion_middle,
            motion_end,
        ])
    

    def __getitem__(self, idx):
        print(f"{idx = }")
        row = self.df.iloc[idx]
        video_id = row["video_id"]

        info = self._get_video_info(video_id)

        frame_indices = self._build_indices(
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            source_fps=info["fps"],
            total_frames=info["total_frames"],
        )

        frames = self._read_frames(video_id, frame_indices)

        self._save_frames_to_video(frames, self._get_jumps_videos_path(row), fps=info["fps"])

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        
        frames_features = self._extract_basic_video_features(frames)

        label = row["jump_type"]

        if not self.return_meta:
            return frames, label

        meta = {
            "video_id": video_id,
            "jump_id": row.get("jump_id"),
            "jump_type": row.get("jump_type"),
            "start_sec": float(row["start_sec"]),
            "start_time": str(row["t_start_val"]),
            "end_sec": float(row["end_sec"]),
            "end_time": str(row["t_end_val"]),
            "indices": frame_indices,
            "path": info["path"],
            "fps": info["fps"],
            "total_frames": info["total_frames"]
        }

        return frames_features, label, meta


def get_videos_config() -> dict[str, dict[str, int | str]]:
    with open(VIDEOS_CONFIG_PATH, "rt", encoding="UTF-8") as f:
        return json.load(f)

def verify_videos() -> None:
    videos_mapping = get_videos_config()

    for video_id, info in sorted(videos_mapping.items(), key=lambda el: int(el[0])):
        video_path = Path(VIDEOS_FOLDER_PATH, info["title"])
        expected_size = info["size"]
        if video_path.exists():
            real_size = round(video_path.stat().st_size / 1024 /1024, 2)
            if str(real_size) != str(expected_size):
                raise ValueError(f"Size of video ({real_size}MB) does not match expected size ({expected_size}MB)")


def to_seconds(t: pd.Timestamp):
    return t.hour * 3600 + t.minute * 60 + t.second


def prepare_dataset(
        DatasetClass: Type,
        video_config: VideoConfig,
        data_config: DataLoaderConfig,
        include_videos: list[str | int] = None,
        exclude_videos: list[str | int] = None,
        test_size: float | None = None,
    ) -> tuple[torch.Tensor, Dataset, DataLoader]:

    verify_videos()

    if not video_config or not data_config:
        raise ValueError("One of configs is not provided.")

    df = parse_excel(TIMECODES_EXCEL_PATH)
    include_videos = set(include_videos) if include_videos else set()
    exclude_videos = set(exclude_videos) if exclude_videos else set()
    unique_videos = set(df["video_id"].unique())

    if include_videos - unique_videos:
        raise ValueError("include_videos param contains non-existing video ids.")
    if exclude_videos - unique_videos:
        raise ValueError("exclude_videos param contains non-existing video ids.") 
    if not (unique_videos - exclude_videos):
        raise ValueError("All videos are excluded.") 

    if include_videos:        
        df = df[df["video_id"].isin(include_videos)]
    
    if exclude_videos:
        df = df[df["video_id"].isin(unique_videos - exclude_videos)]

    print(f"Parsed {len(df)} timecodes for {df['video_id'].nunique()} unique videos")
    print(f"Videos in output dataset: {df['video_id'].unique().tolist()}")

    df["duration"] = df.apply(lambda row: (to_seconds(row["t_end_val"]) - to_seconds(row["t_start_val"])) + 1, axis=1) 
    df = df.sort_values(by=["video_id", "t_start_val"]).reset_index(drop=True)

    video_paths = {}
    videos_config = get_videos_config()
    for video_id, info in sorted(videos_config.items(), key=lambda el: int(el[0])):
        video_paths[int(video_id)] = str(VIDEOS_FOLDER_PATH / info["title"])

    if test_size:
        df_train, df_test = train_test_split(df, shuffle=False, random_state=data_config.seed)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        dataset_train = DatasetClass(
            df=df_train,
            video_paths=video_paths,
            num_frames=video_config.num_frames,
            target_fps=video_config.target_fps,
            image_size=video_config.image_size,
            return_meta=video_config.return_meta,
        )
        dataset_test = DatasetClass(
            df=df_test,
            video_paths=video_paths,
            num_frames=video_config.num_frames,
            target_fps=video_config.target_fps,
            image_size=video_config.image_size,
            return_meta=video_config.return_meta,
        )

        loader_train = DataLoader(
            dataset_train,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers,
            prefetch_factor=data_config.prefetch_factor,
            worker_init_fn=partial(seed_worker, base_seed=SEED),
            generator=make_generator(SEED),
        )
        loader_test = DataLoader(
            dataset_test,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers,
            prefetch_factor=data_config.prefetch_factor,
            worker_init_fn=partial(seed_worker, base_seed=SEED),
            generator=make_generator(SEED),
        )
        return df_train, df_test, dataset_train, dataset_test, loader_train, loader_test

    else:
        dataset = DatasetClass(
            df=df,
            video_paths=video_paths,
            num_frames=video_config.num_frames,
            target_fps=video_config.target_fps,
            image_size=video_config.image_size,
            return_meta=video_config.return_meta,
        )

        loader = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers,
            prefetch_factor=data_config.prefetch_factor,
            worker_init_fn=partial(seed_worker, base_seed=SEED),
            generator=make_generator(SEED),
        )

        return df, dataset, loader

