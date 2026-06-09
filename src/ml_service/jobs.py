"""SQLite-хранилище задач анализа + фоновый воркер.

Анализ полного видео идёт минуты — десятки минут, поэтому /analyze ставит
задачу в очередь и сразу отдаёт job_id, а воркер обрабатывает задачи по одной
(GPU не тянет параллельные прогоны). Статус и результат лежат в SQLite —
переживают переподключение клиента. При перезапуске сервиса незавершённые
задачи помечаются failed (их нужно отправить заново).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from src.backend.schemas import JumpResult, PredictionResponse
from src.ml_service.postprocess import to_prediction_response


def _stub_response() -> PredictionResponse:
    """ВРЕМЕННАЯ ЗАГЛУШКА: фиксированный результат для демонстрации фронта.
    Реальный пайплайн всё равно прогоняется (логи детекции видны), но его
    результат заменяется этим. TODO: убрать заглушку — вернуть to_prediction_response."""
    return PredictionResponse(jumps=[
        JumpResult(jump_type="Аксель", rotation_status="докрут", fall=False,
                   rotations=2.5, start_time="00:50.82", end_time="00:52.80"),
        JumpResult(jump_type="Флип", rotation_status="докрут", fall=False,
                   rotations=3.07, start_time="01:06.90", end_time="01:08.64"),
        JumpResult(jump_type="Лутц", rotation_status="докрут", fall=True,
                   rotations=2.7, start_time="01:58.68", end_time="01:59.28"),
    ])

QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Состояние задач в SQLite. Соединение открывается на каждую операцию
    (запросы мелкие, так избегаем привязки соединения к потоку)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id          TEXT PRIMARY KEY,
                    status      TEXT NOT NULL,
                    video_path  TEXT,
                    result      TEXT,
                    error       TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )

    def create(self, video_path: str) -> str:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, video_path, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, QUEUED, str(video_path), now, now),
            )
        return job_id

    def set_status(
        self,
        job_id: str,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error,
                    _now(),
                    job_id,
                ),
            )

    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job

    def fail_stale(self) -> int:
        """Незавершённые при прошлом запуске задачи → failed. Возвращает их число."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE status IN (?, ?)",
                (FAILED, "сервис был перезапущен во время обработки", _now(),
                 QUEUED, PROCESSING),
            )
            return cur.rowcount


class AnalysisWorker:
    """Один фоновый воркер: тянет задачи из очереди и гоняет пайплайн по одной."""

    def __init__(self, store: JobStore, pipeline):
        self.store = store
        self.pipeline = pipeline
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def submit(self, job_id: str) -> None:
        await self.queue.put(job_id)

    async def run(self) -> None:
        """Бесконечный цикл воркера. Запускается как задача в lifespan."""
        while True:
            job_id = await self.queue.get()
            try:
                await self._process(job_id)
            except Exception as exc:  # воркер не должен падать целиком
                self.store.set_status(job_id, FAILED, error=str(exc))
            finally:
                self.queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        video_path = job["video_path"]
        self.store.set_status(job_id, PROCESSING)
        try:
            # тяжёлый GPU-прогон — в отдельном потоке, чтобы не блокировать loop
            jumps = await run_in_threadpool(self.pipeline.analyze_video, video_path)
            # --- ВРЕМЕННАЯ ЗАГЛУШКА: результат фиксированный ---
            # реальный результат: response = to_prediction_response(jumps)
            response = _stub_response()
            self.store.set_status(job_id, DONE, result=response.model_dump())
        except Exception as exc:
            self.store.set_status(job_id, FAILED, error=str(exc))
        finally:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass
