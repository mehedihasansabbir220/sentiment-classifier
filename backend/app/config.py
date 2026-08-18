from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _BACKEND_ROOT.parent / "model" / "sentiment-distilbert"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Sentiment Classifier API"
    cors_origins: list[str] = ["http://localhost:3000"]
    checkpoint_dir: Path = _DEFAULT_CHECKPOINT


settings = Settings()
