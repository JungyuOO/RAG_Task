from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = BASE_DIR / ".env"
ENV_LOCAL_FILE = BASE_DIR / ".env.local"


class PgvectorRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ENV_FILE, ENV_LOCAL_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rag_source_dir: Path
    embedding_backend: str = "ollama"

    tei_base_url: str = ""
    tei_embedding_model: str = "bge-m3"
    tei_timeout: float = 120.0
    embedding_batch_size: int = 16
    embedding_batch_char_limit: int = 24000
    embedding_parallel_workers: int = 2

    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "bge-m3"
    ollama_timeout: float = 120.0

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    @property
    def db_dsn(self) -> str:
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.db_name} "
            f"user={self.db_user} password={self.db_password}"
        )

    @field_validator("tei_timeout", "ollama_timeout")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"timeout must be positive: {value}")
        return value

    @field_validator("embedding_batch_size", "embedding_batch_char_limit", "embedding_parallel_workers")
    @classmethod
    def _positive_batch_values(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"batch value must be positive: {value}")
        return value


@lru_cache(maxsize=1)
def get_pgvector_runtime_settings() -> PgvectorRuntimeSettings:
    settings = PgvectorRuntimeSettings()
    settings.rag_source_dir.mkdir(parents=True, exist_ok=True)
    return settings
