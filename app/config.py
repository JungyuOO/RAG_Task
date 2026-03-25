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
    llm_prompt_recent_turns: int = 4
    llm_prompt_context_items: int = 5
    llm_prompt_context_char_limit: int = 4000

    rag_data_dir: Path
    rag_source_dir: Path
    rag_index_dir: Path
    rag_cache_dir: Path
    rag_extract_dir: Path
    save_extracted_markdown: bool = True

    # 임베딩 모델 선택. "hash": HashingEmbedder (기본값, 외부 모델 없음),
    # "e5": intfloat/multilingual-e5-small (384차원, sentence-transformers 필요)
    embedding_model: str

    chunk_size: int = 700
    chunk_overlap: int = 120
    structured_chunk_size: int = 1000
    structured_chunk_overlap: int = 150
    chunking_strategy: str = "auto"
    vector_dim: int = 768
    retrieval_top_k: int = 3
    candidate_pool_size: int = 8
    grounded_page_top_n: int = 3
    grounded_chunk_top_n: int = 3
    memory_window_turns: int = 6
    # 검색 최소 점수 임계값.
    # 코퍼스 대비 실측: 관련 청크 rerank_score 평균 0.25~0.45,
    # 비관련 청크 0.03~0.10 범위. 0.12는 비관련 청크를 걸러내면서
    # 부분 매칭(제목만 일치 등)도 허용하는 보수적 하한선.
    retrieval_min_score: float = 0.12

    # 하이브리드 검색 가중치.
    # dense(의미 유사도) 0.45 + sparse(키워드 정확도) 0.25 + title(문서 매칭) 0.15 = 0.85.
    # 나머지 0.15는 title_match_bonus, compact_overlap_bonus로 보정.
    # dense를 가장 높게 설정한 이유: SHA-256 해싱 임베딩은 바이그램+위치 가중으로
    # 토큰 순서를 반영하므로, BM25보다 문맥 유사도 판별에 유리.
    retrieval_dense_weight: float = 0.45
    retrieval_sparse_weight: float = 0.25
    retrieval_title_weight: float = 0.15

    # E5 임베딩 사용 시 가중치 (의미 검색이 강하므로 sparse/title 비중 증가)
    e5_retrieval_dense_weight: float = 0.30
    e5_retrieval_sparse_weight: float = 0.35
    e5_retrieval_title_weight: float = 0.20

    # BM25 파라미터 — Okapi BM25 표준값 (Robertson et al., 1994).
    # k1: TF 포화 계수. 높을수록 반복 출현 토큰의 영향 증가.
    #   k1=1.2는 Elasticsearch/Lucene 기본값이며,
    #   한국어 기술 문서의 짧은 청크(700자)에서 과적합 없이 안정적.
    # b: 문서 길이 정규화 계수. 1.0이면 완전 정규화, 0.0이면 정규화 없음.
    #   b=0.75는 짧은 청크에 약간의 TF 부스트를 주면서
    #   긴 청크의 과대 매칭을 억제하는 표준 균형점.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # 리랭킹 가중치 — 초기 점수를 기반으로 키워드 겹침, 제목 보너스 등을 반영.
    # base(0.68): 1차 검색 점수의 비중을 유지하되,
    # overlap(0.17): 질의-청크 간 키워드 겹침으로 정밀도 보강,
    # title(0.08+0.07): 파일명 매칭 시 출처 관련성 보정,
    # compact(0.12): 연속 부분문자열 매칭으로 구문 일치도 반영.
    rerank_base_weight: float = 0.68
    rerank_overlap_weight: float = 0.17
    rerank_title_weight: float = 0.08
    rerank_title_bonus_weight: float = 0.07
    rerank_compact_bonus_weight: float = 0.12

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
        chunk_size = info.data.get("chunk_size", 700)
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
