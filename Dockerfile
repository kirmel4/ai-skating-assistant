# GPU-образ: torch + torchvision + CUDA + cuDNN из базового образа.
# ВНИМАНИЕ: тег при необходимости подгони под CUDA-драйвер сервера.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# зависимости сервиса (torch/torchvision — уже в базовом образе)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# OpenCV: ultralytics тянет GUI-сборку (требует системный libGL). Меняем на
# headless — она libGL не требует, поэтому apt-get в образе не нужен.
RUN pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --no-cache-dir opencv-python-headless

# веса YOLO — скачиваем в образ, чтобы не тянуть в рантайме
RUN python -c "from ultralytics import YOLO; [YOLO(m) for m in ['yolov8m.pt', 'yolov8m-pose.pt', 'yolov8x-pose.pt']]"

# код сервиса (data/ и checkpoints_* монтируются томами — см. docker-compose)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY front/ ./front/

EXPOSE 9090
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "9090"]
