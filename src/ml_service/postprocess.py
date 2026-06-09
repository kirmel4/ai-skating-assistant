from src.backend.schemas import JumpResult, PredictionResponse
from src.ml_service.contracts import DetectedJump, RawPrediction


ID_TO_JUMP_TYPE = {
    0: "Тулуп",
    1: "Сальхов",
    2: "Риттбергер",
    3: "Флип",
    4: "Лутц",
    5: "Аксель",
}

# B3 обучен на 2 классах недокрута: 0=clean, 1=ur (см. data/b3_meta.json)
ID_TO_ROTATION_STATUS = {
    0: "докрут",
    1: "недокрут",
}


def seconds_to_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def to_jump_result(
    prediction: RawPrediction,
    start_sec: float,
    end_sec: float,
) -> JumpResult:
    # обороты — сырое дробное число из регрессионной головы B3 (rot_reg),
    # для акселя уже с учётом пол-оборота. Без округления к 0.5 — 2 знака.
    rotations = round(prediction.rotations_value, 2)

    return JumpResult(
        jump_type=ID_TO_JUMP_TYPE.get(prediction.jump_type_id, "Неизвестно"),
        rotation_status=ID_TO_ROTATION_STATUS.get(prediction.underrotation_id, "докрут"),
        fall=bool(prediction.fall_id),
        rotations=rotations,
        start_time=seconds_to_timecode(start_sec),
        end_time=seconds_to_timecode(end_sec),
    )


def to_prediction_response(jumps: list[DetectedJump]) -> PredictionResponse:
    """Список найденных прыжков → ответ API."""
    return PredictionResponse(
        jumps=[
            to_jump_result(jump.prediction, jump.start_sec, jump.end_sec)
            for jump in jumps
        ]
    )
