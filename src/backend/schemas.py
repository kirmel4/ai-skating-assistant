from pydantic import BaseModel, Field


class JumpResult(BaseModel):
    jump_type: str = Field(..., examples=["Аксель"])
    rotation_status: str = Field(..., examples=["докрут"])
    fall: bool = Field(..., examples=[False])
    rotations: float = Field(..., examples=[2.5])
    start_time: str = Field(..., examples=["00:00.00"])
    end_time: str = Field(..., examples=["00:01.50"])


class PredictionResponse(BaseModel):
    jumps: list[JumpResult]


class AnalyzeAccepted(BaseModel):
    """Ответ /analyze — задача принята в обработку."""

    job_id: str
    status: str = Field(..., examples=["queued"])


class JobStatusResponse(BaseModel):
    """Ответ /jobs/{id} — статус задачи и результат (когда готов)."""

    job_id: str
    status: str = Field(..., examples=["queued", "processing", "done", "failed"])
    result: PredictionResponse | None = None
    error: str | None = None
