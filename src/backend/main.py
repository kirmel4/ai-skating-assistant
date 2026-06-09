import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

# глушим access-лог uvicorn (GET /health, /jobs, статика) — в логах остаются
# только наши сообщения: загрузка видео (UPLOAD) и прогресс детекции
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

FRONT_DIR = Path(__file__).resolve().parents[2] / "front"

from src.backend.errors import VideoValidationError
from src.backend.schemas import AnalyzeAccepted, JobStatusResponse, PredictionResponse
from src.backend.settings import settings
from src.ml_service.jobs import AnalysisWorker, JobStore
from src.ml_service.predictor_factory import create_pipeline


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class AppState:
    store: JobStore | None = None
    worker: AnalysisWorker | None = None
    worker_task: asyncio.Task | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    state.store = JobStore(settings.jobs_db)
    stale = state.store.fail_stale()
    if stale:
        print(f"Помечено failed (рестарт во время обработки): {stale}")

    # загрузка моделей (YOLO + verifier + B3) — может занять время
    pipeline = create_pipeline(settings)
    state.worker = AnalysisWorker(state.store, pipeline)
    state.worker_task = asyncio.create_task(state.worker.run())

    yield

    state.worker_task.cancel()


app = FastAPI(
    title="AI Skating Assistant Backend",
    version="0.2.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    running = state.worker_task is not None and not state.worker_task.done()
    return {"status": "ok", "worker_running": running}


@app.post("/analyze", response_model=AnalyzeAccepted)
async def analyze_video(request: Request, video: UploadFile = File(...)) -> AnalyzeAccepted:
    """Принимает полное видео программы, ставит задачу анализа в очередь."""
    validate_upload(video)
    try:
        total_bytes = int(request.headers.get("content-length", 0))
    except (TypeError, ValueError):
        total_bytes = 0
    temp_path = await save_upload_to_temp(video, total_bytes)

    if state.store is None or state.worker is None:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Service is not initialized")

    job_id = state.store.create(str(temp_path))
    await state.worker.submit(job_id)
    return AnalyzeAccepted(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    """Статус задачи анализа; result заполнен, когда status == done."""
    if state.store is None:
        raise HTTPException(status_code=503, detail="Service is not initialized")

    job = state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result = PredictionResponse(**job["result"]) if job["result"] else None
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=result,
        error=job["error"],
    )


def validate_upload(video: UploadFile) -> None:
    if not video.filename:
        raise VideoValidationError("Missing filename")

    suffix = Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video extension: {suffix}. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )


async def save_upload_to_temp(video: UploadFile, total_bytes: int = 0) -> Path:
    suffix = Path(video.filename or "").suffix.lower()
    temp_path = settings.temp_dir / f"{uuid4().hex}{suffix}"

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    written = 0

    mb = 1024 * 1024
    fname = video.filename or "видео"
    total_mb = total_bytes / mb if total_bytes > 0 else 0.0
    # шаг логирования: ~5% для известного размера, иначе каждые 50 МБ
    log_step = max(int(total_bytes / 20), 50 * mb) if total_bytes > 0 else 50 * mb
    next_log = log_step

    try:
        with temp_path.open("wb") as f:
            while True:
                chunk = await video.read(mb)
                if not chunk:
                    break

                written += len(chunk)
                if written > max_size_bytes:
                    raise VideoValidationError(
                        f"File is too large. Max size: {settings.max_upload_size_mb} MB"
                    )

                f.write(chunk)

                if written >= next_log:
                    if total_mb > 0:
                        pct = min(written / total_bytes * 100, 100)
                        print(f"UPLOAD | file: {fname} | "
                              f"{written / mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)")
                    else:
                        print(f"UPLOAD | file: {fname} | {written / mb:.0f} MB получено")
                    next_log += log_step

        if written == 0:
            raise VideoValidationError("Uploaded file is empty")

        print(f"UPLOAD | file: {fname} | загрузка завершена: {written / mb:.1f} MB")
        return temp_path

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# Раздача фронта. Монтируется последним — API-роуты (/analyze, /jobs, /health)
# матчатся раньше, остальное (/, /app.js, /styles.css) отдаёт статика.
if FRONT_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="front")
