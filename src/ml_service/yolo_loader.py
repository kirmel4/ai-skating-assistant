"""Загрузка YOLO с ONNX-ускорением на CPU.

ONNXRuntime на CPU обычно в 2-3 раза быстрее PyTorch eager. Логика:
рядом с .pt ищем .onnx-версию; если её нет — один раз экспортируем;
при любой неудаче — фолбэк на .pt. Отключается SKATING_USE_ONNX=false.

В Docker-образе .onnx экспортируются на этапе сборки (см. Dockerfile),
поэтому в рантайме экспорт не запускается.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.backend.settings import settings


def load_yolo(model_name: str, device: torch.device, imgsz: int = 640):
    """model_name — имя .pt-весов (напр. 'yolov8m-pose.pt'). Возвращает YOLO."""
    from ultralytics import YOLO

    if not settings.use_onnx:
        model = YOLO(model_name)
        model.to(device)
        return model

    onnx_path = Path(model_name).with_suffix(".onnx")

    # нет .onnx — пробуем экспортировать один раз
    if not onnx_path.is_file():
        try:
            YOLO(model_name).export(format="onnx", dynamic=True, imgsz=imgsz)
        except Exception as exc:
            print(f"[yolo] ONNX-экспорт {model_name} не удался: {exc} — работаем на .pt")

    if onnx_path.is_file():
        try:
            return YOLO(str(onnx_path))   # ONNXRuntime, CPU по умолчанию
        except Exception as exc:
            print(f"[yolo] загрузка {onnx_path} не удалась: {exc} — фолбэк на .pt")

    model = YOLO(model_name)
    model.to(device)
    return model
