from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: backend/app/config.py -> backend/app -> backend -> <repo>/
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = Path("model/sentiment-distilbert")


class Settings(BaseSettings):
    # protected_namespaces=() so the `model_path` field does not collide with
    # pydantic's reserved `model_` namespace.
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "Sentiment Classifier API"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Directory holding the fine-tuned checkpoint exported from Colab.
    # Override with MODEL_PATH (CHECKPOINT_DIR is kept as a legacy alias).
    model_path: Path = Field(
        default=DEFAULT_MODEL_PATH,
        validation_alias=AliasChoices("MODEL_PATH", "CHECKPOINT_DIR"),
    )

    @property
    def resolved_model_path(self) -> Path:
        """Absolute checkpoint path.

        Relative values (including the default `model/sentiment-distilbert`)
        resolve against the repository root, so the API behaves the same no
        matter which directory uvicorn is started from.
        """
        path = self.model_path.expanduser()
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return path


settings = Settings()
