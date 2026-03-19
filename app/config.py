from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Custom RAG Task"
    cllm_base_url: str = "http://cllm.cywell.co.kr/v1"
    cllm_model: str = "Qwen/Qwen3.5-9B"
    llm_connect_timeout_seconds: float = 3.0
    llm_read_timeout_seconds: float = 20.0
    llm_write_timeout_seconds: float = 10.0
    llm_pool_timeout_seconds: float = 5.0
    llm_total_timeout_seconds: float = 90.0
    llm_timeout_cooldown_seconds: float = 8.0
    llm_failure_cooldown_seconds: float = 45.0
    llm_prompt_recent_turns: int = 4
    llm_prompt_context_items: int = 3
    llm_prompt_context_char_limit: int = 2200

    rag_data_dir: Path = Path("./data")
    rag_source_dir: Path = Path("./data/corpus/pdfs")
    rag_index_dir: Path = Path("./data/index")
    rag_cache_dir: Path = Path("./data/cache")
    rag_extract_dir: Path = Path("./data/extracted_markdown")
    save_extracted_markdown: bool = True

    ocr_lang: str = "korean"
    ocr_dpi: int = 220
    chunk_size: int = 700
    chunk_overlap: int = 120
    structured_chunk_size: int = 1000
    structured_chunk_overlap: int = 150
    chunking_strategy: str = "auto"
    vector_dim: int = 768
    retrieval_top_k: int = 6
    candidate_pool_size: int = 14
    grounded_page_top_n: int = 3
    grounded_chunk_top_n: int = 4
    memory_window_turns: int = 6
    retrieval_min_score: float = 0.12


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
