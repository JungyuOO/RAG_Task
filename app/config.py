from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """애플리케이션 전체 설정. 환경 변수 또는 .env 파일에서 오버라이드 가능하다."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Custom RAG Task"
    cllm_base_url: str
    cllm_model: str
    llm_connect_timeout_seconds: float = 3.0
    llm_read_timeout_seconds: float = 20.0
    llm_write_timeout_seconds: float = 10.0
    llm_pool_timeout_seconds: float = 5.0
    llm_total_timeout_seconds: float = 90.0
    llm_timeout_cooldown_seconds: float = 8.0
    llm_failure_cooldown_seconds: float = 45.0
    llm_stream_temperature: float = 0.1
    llm_generate_temperature: float = 0.0
    llm_generate_max_tokens: int = 512
    llm_prompt_recent_turns: int = 4
    llm_prompt_context_items: int = 5
    llm_prompt_context_char_limit: int = 4000

    rag_data_dir: Path
    rag_source_dir: Path
    rag_index_dir: Path
    rag_cache_dir: Path
    rag_extract_dir: Path
    save_extracted_markdown: bool = True

    # Ollama 설정 (BGE-M3 임베딩)
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
    memory_window_turns: int = 6
    # 검색 최소 점수 임계값.
    # 이 점수 미만이면 검색 결과를 사용하지 않고, 자료보기도 표시하지 않는다.
    # 코퍼스 대비 실측: 관련 청크 rerank_score 평균 0.25~0.45,
    # 비관련 청크 0.03~0.10 범위.
    retrieval_min_score: float = 0.25
    # 재질문 유도 최소 점수 — 이 점수 이상 retrieval_min_score 미만이면
    # "못 찾았다" 대신 구체적 재질문을 유도한다. 이 점수 미만이면 완전 실패.
    retrieval_retry_min_score: float = 0.10

    # 하이브리드 검색 가중치 (search() 메서드용 — RRF는 랭크 기반이므로 가중치 미사용).
    # dense를 높게 설정한 이유: BGE-M3 임베딩은 의미 유사도 판별에 유리.
    retrieval_dense_weight: float = 0.6
    retrieval_sparse_weight: float = 0.4

    # BM25 파라미터 — Okapi BM25 표준값 (Robertson et al., 1994).
    # k1=1.2: Elasticsearch/Lucene 기본값, 512자 청크에서 안정적.
    # b=0.75: 짧은 청크에 약간의 TF 부스트를 주면서 긴 청크 과대 매칭 억제.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # 리랭킹 가중치 — RRF 1차 점수 + 키워드 겹침으로 최종 순위 결정.
    # cross-encoder(BGEReranker)가 최종 재순위를 담당하므로 경량 휴리스틱만 유지.
    rerank_base_weight: float = 0.8
    rerank_overlap_weight: float = 0.2

    # PDF 페이지 이미지 렌더링 DPI
    pdf_render_dpi: int = 170

    # PostgreSQL 설정
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    # 캐시 설정
    cache_max_entries: int = 500
    cache_ttl_hours: int = 72

    @property
    def db_dsn(self) -> str:
        """PostgreSQL 연결 문자열을 반환한다."""
        return f"host={self.db_host} port={self.db_port} dbname={self.db_name} user={self.db_user} password={self.db_password}"

    @field_validator(
        "llm_connect_timeout_seconds", "llm_read_timeout_seconds",
        "llm_write_timeout_seconds", "llm_pool_timeout_seconds",
        "llm_total_timeout_seconds", "llm_timeout_cooldown_seconds",
        "llm_failure_cooldown_seconds",
    )
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        """타임아웃 값은 반드시 양수여야 한다."""
        if value <= 0:
            raise ValueError(f"타임아웃 값은 양수여야 합니다: {value}")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_less_than_size(cls, value: int, info) -> int:
        """chunk_overlap은 chunk_size보다 작아야 한다."""
        chunk_size = info.data.get("chunk_size", 512)
        if value >= chunk_size:
            raise ValueError(f"chunk_overlap({value})은 chunk_size({chunk_size})보다 작아야 합니다")
        return value

    @field_validator("vector_dim", "retrieval_top_k", "candidate_pool_size", "memory_window_turns", "cache_max_entries")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        """핵심 정수 설정은 1 이상이어야 한다."""
        if value < 1:
            raise ValueError(f"값은 1 이상이어야 합니다: {value}")
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
