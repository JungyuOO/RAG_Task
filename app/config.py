from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Custom RAG Task"
    cllm_base_url: str
    cllm_model: str
    cllm_api_key: str = ""
    llm_connect_timeout_seconds: float = 10.0
    llm_read_timeout_seconds: float = 30.0
    llm_write_timeout_seconds: float = 10.0
    llm_pool_timeout_seconds: float = 10.0
    llm_total_timeout_seconds: float = 90.0
    llm_timeout_cooldown_seconds: float = 3.0
    llm_failure_cooldown_seconds: float = 5.0
    llm_stream_temperature: float = 0.1
    llm_generate_temperature: float = 0.0
    llm_generate_max_tokens: int = 512
    llm_prompt_recent_turns: int = 10
    llm_prompt_context_items: int = 5
    llm_prompt_context_char_limit: int = 4000

    rag_data_dir: Path
    rag_source_dir: Path
    rag_index_dir: Path
    rag_cache_dir: Path
    rag_extract_dir: Path
    save_extracted_markdown: bool = True

    # Ollama settings for BGE-M3 embeddings.
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "bge-m3"
    ollama_timeout: float = 120.0

    chunk_size: int = 512
    chunk_overlap: int = 50
    structured_chunk_size: int = 512
    structured_chunk_overlap: int = 50
    chunking_strategy: str = "auto"
    vector_dim: int = 1024
    retrieval_top_k: int = 3
    candidate_pool_size: int = 8
    grounded_page_top_n: int = 3
    grounded_chunk_top_n: int = 3
    memory_window_turns: int = 12

    # Retrieval acceptance thresholds tuned against the current corpus.
    retrieval_min_score: float = 0.25
    retrieval_retry_min_score: float = 0.10

    # BM25 parameters.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # RRF fusion constant. Lower = rank differences matter more (default 60 is too conservative).
    rrf_k: int = 30

    # Lightweight overlap rerank before the cross-encoder reranker.
    rerank_base_weight: float = 0.8
    rerank_overlap_weight: float = 0.2
    rerank_title_weight: float = 0.15

    pdf_render_dpi: int = 170

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    cache_max_entries: int = 500
    cache_ttl_hours: int = 72

    @property
    def db_dsn(self) -> str:
        """Return the PostgreSQL DSN string."""
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.db_name} "
            f"user={self.db_user} password={self.db_password}"
        )

    @field_validator(
        "llm_connect_timeout_seconds",
        "llm_read_timeout_seconds",
        "llm_write_timeout_seconds",
        "llm_pool_timeout_seconds",
        "llm_total_timeout_seconds",
        "llm_timeout_cooldown_seconds",
        "llm_failure_cooldown_seconds",
    )
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        """Timeout values must be positive."""
        if value <= 0:
            raise ValueError(f"timeout must be positive: {value}")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_less_than_size(cls, value: int, info) -> int:
        """chunk_overlap must be smaller than chunk_size."""
        chunk_size = info.data.get("chunk_size", 512)
        if value >= chunk_size:
            raise ValueError(
                f"chunk_overlap({value}) must be smaller than chunk_size({chunk_size})"
            )
        return value

    @field_validator(
        "vector_dim",
        "retrieval_top_k",
        "candidate_pool_size",
        "memory_window_turns",
        "cache_max_entries",
    )
    @classmethod
    def _positive_int(cls, value: int) -> int:
        """Positive integer settings must be at least 1."""
        if value < 1:
            raise ValueError(f"value must be at least 1: {value}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    for directory in (
        settings.rag_data_dir,
        settings.rag_source_dir,
        settings.rag_index_dir,
        settings.rag_cache_dir,
        settings.rag_extract_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
