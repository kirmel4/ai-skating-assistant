"""Экспорт метаданных B3-классификатора для инференса.

B3 строит rotation_map / underrotation_map из обучающих данных в рантайме
(см. train_pose_b3_final.main()), а в чекпойнт эти маппинги не попадают.
Без них сервис не может ни собрать модель с правильным числом классов,
ни расшифровать id класса в человекочитаемое значение.

Скрипт повторяет те же преобразования, что и обучение, и пишет
data/b3_meta.json. Запускается один раз, обучение не требуется (секунды).

Использование:
    python scripts/export_b3_meta.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.clip_dataset import LABEL_MAP, prepare_clip_dataset
from scripts.train_dinov_2_temporal import (
    make_underrotation_map,
    normalize_underrotation_value,
)
from src.config import DataLoaderConfig, VideoConfig

OUT_PATH = _REPO_ROOT / "data" / "b3_meta.json"


def main():
    # те же параметры датасета, что в train_pose_b3_final.main()
    video_config = VideoConfig(num_frames=64, target_fps=25.0, image_size=224, return_meta=False)
    data_config = DataLoaderConfig(
        batch_size=32, shuffle=False, num_workers=0,
        pin_memory=False, persistent_workers=False, prefetch_factor=2,
    )
    df_clips, _, _ = prepare_clip_dataset(video_config, data_config, exclude_videos=[1])
    valid_mask = df_clips[["jump_type", "rotations", "underrotation", "fall"]].notna().all(axis=1)
    df_clips = df_clips[valid_mask].reset_index(drop=True)

    # rotation_map: отсортированные уникальные значения оборотов → id класса
    rotation_values = sorted(df_clips["rotations"].astype(int).unique().tolist())
    rotation_map = {v: i for i, v in enumerate(rotation_values)}

    # underrotation_map: {статус → id класса}
    under_values = df_clips["underrotation"].apply(normalize_underrotation_value).values
    underrotation_map = make_underrotation_map(under_values)

    meta = {
        "num_classes": {
            "jump": len(LABEL_MAP),
            "rot": len(rotation_map),
            "under": len(underrotation_map),
            "fall": 2,
        },
        # id класса → человекочитаемое значение
        "jump_id_to_code": {str(i): code for code, i in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])},
        "rotation_id_to_value": {str(i): float(v) for v, i in rotation_map.items()},
        "underrotation_id_to_status": {str(i): status for status, i in underrotation_map.items()},
    }

    OUT_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Датасет: {len(df_clips)} прыжков")
    print(f"B3 meta → {OUT_PATH}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
