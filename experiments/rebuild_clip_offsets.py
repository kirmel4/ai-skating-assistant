"""Восстанавливает записи в clip_offsets.json для уже существующих MP4 в data/clips/.
Полезно когда clip_downloader.py упал на части видео, а файлы скопированы вручную.

Использование:
    python -m scripts.rebuild_clip_offsets               # все видео
    python -m scripts.rebuild_clip_offsets --videos 9 10 11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.xlsx_parser import parse_excel

BASE_DIR = Path(__file__).parent.parent
CLIPS_DIR = BASE_DIR / "data" / "clips"
TIMECODES_PATH = BASE_DIR / "data" / "Разметка прыжков.xlsx"
CLIP_OFFSETS_PATH = BASE_DIR / "data" / "clip_offsets.json"
BUFFER_SEC = 2.0


def time_to_seconds(t) -> float:
    if pd.isna(t):
        return 0.0
    return t.hour * 3600 + t.minute * 60 + t.second


def run(video_ids_filter: list[int] | None = None) -> None:
    df = parse_excel(TIMECODES_PATH)

    clip_offsets: dict = {}
    if CLIP_OFFSETS_PATH.exists():
        with open(CLIP_OFFSETS_PATH, encoding="utf-8") as f:
            clip_offsets = json.load(f)

    added = 0
    missing_files = 0

    for _, row in df.iterrows():
        video_id = int(row["video_id"])
        if video_ids_filter and video_id not in video_ids_filter:
            continue

        jump_id = int(row["jump_id"])
        t_start_sec = time_to_seconds(row["t_start_val"])
        t_start_int = int(t_start_sec)
        clip_start = max(0.0, t_start_sec - BUFFER_SEC)

        key = f"{video_id}_{jump_id}_{t_start_int}"
        if key in clip_offsets:
            continue

        clip_path = CLIPS_DIR / str(video_id) / f"{jump_id}_{t_start_int}.mp4"
        if not clip_path.is_file():
            missing_files += 1
            continue

        clip_offsets[key] = {
            "clip_start": clip_start,
            "clip_path": clip_path.relative_to(BASE_DIR).as_posix(),
        }
        added += 1

    with open(CLIP_OFFSETS_PATH, "w", encoding="utf-8") as f:
        json.dump(clip_offsets, f, indent=2, ensure_ascii=False)

    print(f"Added {added} entries to clip_offsets.json")
    if missing_files:
        print(f"Skipped {missing_files} jumps: MP4 not found in data/clips/")
    print(f"Total entries: {len(clip_offsets)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", type=int)
    args = parser.parse_args()
    run(video_ids_filter=args.videos)
