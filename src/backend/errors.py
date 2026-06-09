from fastapi import HTTPException, status


class VideoValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


class ModelNotReadyError(HTTPException):
    def __init__(self, detail: str = "Model is not ready"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class PredictionError(HTTPException):
    def __init__(self, detail: str = "Prediction failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )
