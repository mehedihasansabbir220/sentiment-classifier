from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
