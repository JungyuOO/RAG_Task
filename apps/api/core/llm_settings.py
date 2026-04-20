from __future__ import annotations

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class ChatLlmSettings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cllm_base_url: str = ""
    cllm_model: str = ""
    cllm_api_key: str = ""

    llm_generate_temperature: float = 0.15
    llm_generate_max_tokens: int = 1400
    llm_intent_max_tokens: int = 200
    llm_synthesis_max_tokens: int = 600
    use_query_router: bool = True
    use_rrf_fusion: bool = True
    use_char_ngram_sparse: bool = True
    use_asymmetric_query_prompt: bool = True
    use_synonym_expansion: bool = True
    use_gap_triggered_rerank: bool = True
    rerank_gap_threshold: float = 0.15
    use_native_citation_prompt: bool = True
    use_response_cache: bool = True
    force_korean_answers: bool = True

    llm_connect_timeout_seconds: float = 5.0
    llm_read_timeout_seconds: float = 30.0
    llm_write_timeout_seconds: float = 15.0
    llm_pool_timeout_seconds: float = 5.0
    llm_total_timeout_seconds: float = 35.0


def get_chat_llm_settings() -> ChatLlmSettings:
    return ChatLlmSettings()
