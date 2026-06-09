from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent.parent
CLIPS_DIR = BASE_DIR / "data" / "clips"
VIDEOS_DIR = BASE_DIR / "data" / "videos"
TIMECODES_PATH = BASE_DIR / "data" / "Разметка прыжков.xlsx"
VIDEO_URLS_PATH = BASE_DIR / "data" / "video_urls.json"
CLIP_OFFSETS_PATH = BASE_DIR / "data" / "clip_offsets.json"

BUFFER_SEC = 2.0


def check_ffmpeg() -> None:
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        print("ffmpeg not found: https://ffmpeg.org/download.html")
        sys.exit(1)


def time_to_seconds(t) -> float:
    if pd.isna(t):
        return 0.0
    return t.hour * 3600 + t.minute * 60 + t.second


def is_local(url: str) -> bool:
    return url.startswith("local:")


def is_gdrive(url: str) -> bool:
    return "drive.google.com" in url


def is_yandex_folder_url(url: str) -> bool:
    parsed = urlparse(url)
    return "disk.yandex.ru" in parsed.netloc and parsed.path.count("/") >= 3


def get_yandex_360_direct_url(public_url: str) -> str:
    from src.utils.downloaders import get_yandex_disk_direct_url
    return get_yandex_disk_direct_url(public_url)


def get_yandex_folder_direct_url(public_url: str) -> str:
    parsed = urlparse(public_url)
    parts = parsed.path.lstrip("/").split("/", 2)
    folder_key = f"https://disk.yandex.ru/d/{parts[1]}"
    file_path = "/" + unquote(parts[2])
    resp = requests.get(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download",
        params={"public_key": folder_key, "path": file_path},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["href"]


def cut_clip(source: str, output_path: Path, t_start: float, t_end: float) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"    skip: {output_path.name}")
        return True

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}",
        "-to", f"{t_end:.3f}",
        "-i", source,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
        return False

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"    OK: {output_path.name}  ({size_mb:.1f} MB, {t_start:.1f}s-{t_end:.1f}s)")
    return True


def cut_clip_gdrive(public_url: str, output_path: Path, t_start: float, t_end: float) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"    skip: {output_path.name}")
        return True

    cmd = [
        "yt-dlp",
        "--download-sections", f"*{t_start:.3f}-{t_end:.3f}",
        "--cookies-from-browser", "edge",
        "-o", str(output_path),
        "--no-playlist",
        "--quiet",
        public_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"    ERROR: timeout 180s")
        return False

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
        return False

    if not output_path.exists() or output_path.stat().st_size == 0:
        print(f"    ERROR: file empty or missing")
        return False

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"    OK: {output_path.name}  ({size_mb:.1f} MB, {t_start:.1f}s-{t_end:.1f}s)")
    return True


def run(video_ids_filter: list[int] | None = None) -> None:
    check_ffmpeg()

    with open(VIDEO_URLS_PATH, encoding="utf-8") as f:
        video_urls: dict[str, str] = json.load(f)

    from src.utils.xlsx_parser import parse_excel
    df = parse_excel(TIMECODES_PATH)

    clip_offsets: dict = {}
    if CLIP_OFFSETS_PATH.exists():
        with open(CLIP_OFFSETS_PATH, encoding="utf-8") as f:
            clip_offsets = json.load(f)

    for video_id_str, url in sorted(video_urls.items(), key=lambda x: int(x[0])):
        video_id = int(video_id_str)

        if video_ids_filter and video_id not in video_ids_filter:
            continue

        video_jumps = df[df["video_id"] == video_id]
        if video_jumps.empty:
            print(f"[video {video_id}] no jumps, skip")
            continue

        if is_local(url):
            strategy = "ffmpeg"
            source = str(VIDEOS_DIR / url[len("local:"):])
            print(f"\n[video {video_id}] local: {url[len('local:'):]}")
        elif is_gdrive(url):
            strategy = "gdrive"
            source = url
            print(f"\n[video {video_id}] gdrive")
        else:
            strategy = "ffmpeg"
            try:
                source = get_yandex_folder_direct_url(url) if is_yandex_folder_url(url) else get_yandex_360_direct_url(url)
            except Exception as e:
                print(f"[video {video_id}] failed to get url: {e}")
                continue
            print(f"\n[video {video_id}] yandex disk")

        print(f"[video {video_id}] {len(video_jumps)} jumps")

        for _, row in video_jumps.iterrows():
            jump_id = int(row["jump_id"])
            t_start_sec = time_to_seconds(row["t_start_val"])
            t_end_sec = time_to_seconds(row["t_end_val"]) + 1.0
            clip_start = max(0.0, t_start_sec - BUFFER_SEC)
            clip_end = t_end_sec + BUFFER_SEC
            t_start_int = int(t_start_sec)

            output_path = CLIPS_DIR / str(video_id) / f"{jump_id}_{t_start_int}.mp4"
            key = f"{video_id}_{jump_id}_{t_start_int}"

            if strategy == "gdrive":
                ok = cut_clip_gdrive(source, output_path, clip_start, clip_end)
            else:
                ok = cut_clip(source, output_path, clip_start, clip_end)

            if ok:
                clip_offsets[key] = {
                    "clip_start": clip_start,
                    "clip_path": output_path.relative_to(BASE_DIR).as_posix(),
                }

        with open(CLIP_OFFSETS_PATH, "w", encoding="utf-8") as f:
            json.dump(clip_offsets, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Offsets: {CLIP_OFFSETS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", type=int)
    args = parser.parse_args()
    run(video_ids_filter=args.videos)
