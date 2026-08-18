"""Request/response models for the inference API.

Pure data contracts — no model, tokenizer, or torch code lives here.
"""

from pydantic import BaseModel, Field, field_validator

# The tokenizer truncates at 512 tokens; this bound just stops absurd payloads
# from being tokenized at all.
MAX_TEXT_LENGTH = 10_000


class HealthResponse(BaseModel):
    status: str


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Text to classify.",
        examples=["I really enjoyed this movie."],
    )

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I really enjoyed this movie.",
                    "sentiment": "positive",
                    "confidence": 0.97,
                    "probabilities": {"positive": 0.97, "negative": 0.03},
                }
            ]
        }
    }
