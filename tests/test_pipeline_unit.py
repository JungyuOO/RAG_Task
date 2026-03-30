"""Unit tests for RagPipeline.stream_chat() — no DB, no LLM required.

All external dependencies (DB, LLM, embedder, caches) are replaced with
MagicMock instances. The pipeline object is created by bypassing __init__
so no real connections are attempted.
"""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.rag.embeddings import HashingEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.retrieval import HybridRetriever
from app.rag.types import ChatTurn
from app.services.answer_service import AnswerService
from app.services.retrieval_service import RetrievalService
from app.services.turn_policy_service import TurnPolicyService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever() -> HybridRetriever:
    return HybridRetriever(
        top_k=3,
        candidate_pool_size=8,
        dense_weight=0.45,
        sparse_weight=0.25,
        title_weight=0.15,
        bm25_k1=1.2,
        bm25_b=0.75,
        rerank_base_weight=0.68,
        rerank_overlap_weight=0.17,
        rerank_title_weight=0.08,
        rerank_title_bonus_weight=0.07,
        rerank_compact_bonus_weight=0.12,
        title_match_bonus=0.35,
    )


def _make_settings() -> MagicMock:
    """Return a MagicMock that satisfies all Settings attribute accesses in stream_chat()."""
    s = MagicMock()
    s.retrieval_min_score = 0.12
    s.llm_prompt_recent_turns = 3
    s.llm_prompt_context_items = 5
    s.llm_prompt_context_char_limit = 3000
    s.retrieval_top_k = 3
    s.candidate_pool_size = 8
    s.grounded_page_top_n = 3
    s.grounded_chunk_top_n = 3
    s.embedding_model = "hash"
    s.vector_dim = 768
    return s


def _make_session_repository(recent_turns=None) -> MagicMock:
    repo = MagicMock()
    repo.recent_turns.return_value = recent_turns or []
    repo.structured_summary.return_value = {}
    repo.topic_state.return_value = {}
    repo.memory_snapshot.return_value = {
        "session_summary": {},
        "topic_state": {},
        "recent_turns": [],
    }
    repo.build_rewrite_context.return_value = None
    repo.summary.return_value = ""
    repo.add_turn.return_value = None
    return repo


def _make_fake_index_items(source_path: str = "manual.pdf") -> list[dict]:
    embedder = HashingEmbedder(dim=768)
    text = "PV PVC persistent volume claim 설명 kubernetes storage"
    return [
        {
            "vector": embedder.encode(text),
            "chunk": {
                "chunk_id": "chunk-pv-1",
                "source_path": source_path,
                "page_number": 5,
                "metadata": {"page_start": 5, "page_end": 6},
                "tokens": text.split(),
                "text": text,
            },
        }
    ]


def _build_pipeline() -> RagPipeline:
    """Create a RagPipeline with all I/O dependencies mocked out."""
    pipeline: RagPipeline = object.__new__(RagPipeline)
    pipeline.settings = _make_settings()

    # Real retrieval components — deterministic, no I/O
    pipeline.embedder = HashingEmbedder(dim=768)
    pipeline.retriever = _make_retriever()

    # Retrieval service and answer service use no external I/O
    pipeline.retrieval_service = RetrievalService(pipeline.settings)
    pipeline.answer_service = AnswerService(pipeline.retrieval_service)
    pipeline.turn_policy_service = TurnPolicyService()

    # All repository/cache dependencies mocked
    pipeline.index_repository = MagicMock()
    pipeline.index_repository.load.return_value = []

    pipeline.embedding_cache_repository = MagicMock()
    pipeline.embedding_cache_repository.get.return_value = None
    pipeline.embedding_cache_repository.set.return_value = None

    pipeline.answer_cache_repository = MagicMock()
    pipeline.answer_cache_repository.get.return_value = None
    pipeline.answer_cache_repository.set.return_value = None

    pipeline.session_repository = _make_session_repository()

    # LLM mocked — default: empty stream
    pipeline.llm = MagicMock()
    pipeline.llm.generate = AsyncMock(return_value="")

    async def _default_stream(_messages):
        return
        yield  # pragma: no cover — makes this an async generator

    pipeline.llm.stream_chat = _default_stream

    # Agent mocks
    pipeline.query_agent = MagicMock()
    pipeline.query_agent.refine_query = AsyncMock(return_value={
        "refined_query": "",  # overridden per test
        "alternative_queries": [],
        "search_keywords": [],
    })

    pipeline.judge_agent = MagicMock()
    pipeline.judge_agent.evaluate = AsyncMock(return_value={
        "relevant": True,
        "confidence": "high",
        "clarification_message": "",
    })

    return pipeline


async def _collect_events(gen) -> list[dict]:
    events: list[dict] = []
    async for event in gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class StreamChatGreetingTest(unittest.TestCase):
    """안녕하세요 → greeting turn → 인사 응답, 검색 없음."""

    def test_stream_chat_greeting(self) -> None:
        pipeline = _build_pipeline()
        # Retrieval must NOT be called for a greeting
        pipeline.index_repository.load.return_value = []

        async def run():
            return await _collect_events(
                pipeline.stream_chat("sess-greet", "안녕하세요", append_user_turn=False)
            )

        events = asyncio.run(run())
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]
        context_events = [e for e in events if e["type"] == "context"]

        # index_repository.load should NOT have been called (no retrieval for greetings)
        pipeline.index_repository.load.assert_not_called()

        self.assertEqual(len(token_events), 1)
        self.assertIn("안녕하세요", token_events[0]["content"])
        self.assertEqual(len(done_events), 1)
        self.assertFalse(done_events[0]["cached"])
        # First context event has greeting mode
        greeting_context = [e for e in context_events if e.get("mode") == "greeting"]
        self.assertGreater(len(greeting_context), 0)


class StreamChatGeneralChatRejectedTest(unittest.TestCase):
    """오늘 날씨 어때 → general_chat turn → 거부 응답."""

    def test_stream_chat_general_chat_rejected(self) -> None:
        pipeline = _build_pipeline()

        async def run():
            return await _collect_events(
                pipeline.stream_chat("sess-general", "오늘 날씨 어때", append_user_turn=False)
            )

        events = asyncio.run(run())
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        # Retrieval should not be triggered for a general chat message
        pipeline.index_repository.load.assert_not_called()

        self.assertEqual(len(token_events), 1)
        self.assertIn("문서", token_events[0]["content"])
        self.assertEqual(len(done_events), 1)


class StreamChatDocumentQueryTest(unittest.TestCase):
    """PV PVC 설명해줘 → document_query → LLM 토큰 스트림."""

    def _setup_pipeline_with_matching_index(self) -> RagPipeline:
        pipeline = _build_pipeline()
        source_path = "/data/pdfs/storage.pdf"
        items = _make_fake_index_items(source_path)
        pipeline.index_repository.load.return_value = items

        # Make the embedder return a real vector (already using real HashingEmbedder)
        # query_agent returns a refined query that matches the index text
        pipeline.query_agent.refine_query = AsyncMock(return_value={
            "refined_query": "PV PVC persistent volume claim 설명",
            "alternative_queries": [],
            "search_keywords": ["PV", "PVC"],
        })

        # LLM yields two tokens
        async def fake_stream(_messages):
            yield "PV는 "
            yield "퍼시스턴트 볼륨입니다."

        pipeline.llm.stream_chat = fake_stream
        return pipeline

    def test_stream_chat_document_query_with_results(self) -> None:
        pipeline = self._setup_pipeline_with_matching_index()

        async def run():
            return await _collect_events(
                pipeline.stream_chat("sess-doc", "PV PVC 설명해줘", append_user_turn=False)
            )

        events = asyncio.run(run())
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]
        context_events = [e for e in events if e["type"] == "context"]

        # index was loaded
        pipeline.index_repository.load.assert_called()

        # At least the two LLM tokens must arrive
        token_contents = [e["content"] for e in token_events]
        full_answer = "".join(token_contents)
        self.assertIn("PV", full_answer)

        # Stream must finish
        self.assertEqual(len(done_events), 1)

        # At least two context events (initial + final)
        self.assertGreaterEqual(len(context_events), 2)
        # Answer was cached
        pipeline.answer_cache_repository.set.assert_called()


class StreamChatCachedAnswerTest(unittest.TestCase):
    """캐시 히트 → cached=True 토큰 반환."""

    def test_stream_chat_cached_answer(self) -> None:
        pipeline = _build_pipeline()
        source_path = "/data/pdfs/storage.pdf"
        items = _make_fake_index_items(source_path)
        pipeline.index_repository.load.return_value = items

        pipeline.query_agent.refine_query = AsyncMock(return_value={
            "refined_query": "PV PVC persistent volume claim 설명",
            "alternative_queries": [],
            "search_keywords": ["PV", "PVC"],
        })

        # Return a cached answer
        pipeline.answer_cache_repository.get.return_value = {"answer": "캐시된 PV 설명입니다."}

        # LLM must NOT be called when cache hits
        unexpected_called = []

        async def unexpected_llm(_messages):
            unexpected_called.append(True)
            yield "should not appear"

        pipeline.llm.stream_chat = unexpected_llm

        async def run():
            return await _collect_events(
                pipeline.stream_chat("sess-cached", "PV PVC 설명해줘", append_user_turn=False)
            )

        events = asyncio.run(run())
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        # All token events must carry cached=True
        self.assertTrue(all(e.get("cached") is True for e in token_events))
        # The cached text must appear
        full_answer = "".join(e["content"] for e in token_events)
        self.assertIn("캐시된", full_answer)
        # done event must also signal cached
        self.assertTrue(done_events[0]["cached"])
        # LLM was not invoked
        self.assertEqual(unexpected_called, [])


class StreamChatLlmFailureFallbackTest(unittest.TestCase):
    """LLM 예외 → fallback 텍스트 반환."""

    def test_stream_chat_llm_failure_fallback(self) -> None:
        pipeline = _build_pipeline()
        source_path = "/data/pdfs/storage.pdf"
        items = _make_fake_index_items(source_path)
        pipeline.index_repository.load.return_value = items

        pipeline.query_agent.refine_query = AsyncMock(return_value={
            "refined_query": "PV PVC persistent volume claim 설명",
            "alternative_queries": [],
            "search_keywords": ["PV", "PVC"],
        })

        # LLM raises on first call
        async def failing_stream(_messages):
            raise RuntimeError("connection refused")
            yield  # pragma: no cover

        pipeline.llm.stream_chat = failing_stream

        async def run():
            return await _collect_events(
                pipeline.stream_chat("sess-fail", "PV PVC 설명해줘", append_user_turn=False)
            )

        events = asyncio.run(run())
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        # A fallback token must be emitted
        self.assertGreater(len(token_events), 0)
        # At least one token event must have an error field
        error_events = [e for e in token_events if "error" in e]
        self.assertGreater(len(error_events), 0)
        self.assertIn("connection refused", error_events[0]["error"])
        # Stream must still terminate
        self.assertEqual(len(done_events), 1)


class TestNonKoreanQueryDetection(unittest.TestCase):
    """비한국어 질문 감지 로직 검증."""

    def test_chinese_query_detected(self):
        from app.rag.pipeline import RagPipeline
        result = RagPipeline._detect_non_korean_query("这是什么意思？")
        self.assertIsNotNone(result)
        self.assertIn("한국어", result)

    def test_korean_query_allowed(self):
        from app.rag.pipeline import RagPipeline
        result = RagPipeline._detect_non_korean_query("PV와 PVC 차이점이 뭐야?")
        self.assertIsNone(result)

    def test_english_with_korean_allowed(self):
        from app.rag.pipeline import RagPipeline
        result = RagPipeline._detect_non_korean_query("Kubernetes Storage 설명해줘")
        self.assertIsNone(result)

    def test_pure_english_technical_allowed(self):
        from app.rag.pipeline import RagPipeline
        result = RagPipeline._detect_non_korean_query("what is PV in kubernetes")
        self.assertIsNone(result)

    def test_japanese_query_detected(self):
        from app.rag.pipeline import RagPipeline
        result = RagPipeline._detect_non_korean_query("これは何ですか？")
        self.assertIsNotNone(result)


class TestLanguageEnforcement(unittest.TestCase):
    """시스템 프롬프트에 한국어 강제 지시가 포함되어 있는지 검증."""

    def test_system_prompt_contains_korean_instruction(self):
        pipeline = _build_pipeline()
        messages = pipeline._build_llm_messages(
            session_id="test",
            user_message="what is kubernetes?",
            code_example_request=False,
            response_mode="rag",
            turn_policy={},
            top_score=0.5,
            context_blocks=[],
        )
        system_content = messages[0]["content"]
        self.assertIn("한국어", system_content)


class TestContextBleeding(unittest.TestCase):
    """새로운 토픽 질문 시 이전 소스 인용이 제거되는지 검증."""

    def test_clean_turns_removes_pdf_citations(self):
        pipeline = _build_pipeline()
        pipeline.session_repository.recent_turns.return_value = [
            ChatTurn(role="user", content="쿠버네티스 PV 설명해줘"),
            ChatTurn(role="assistant", content="PV는 영구 볼륨입니다.\n[kubernetes.pdf] p.5"),
        ]
        cleaned = pipeline._build_prompt_recent_turns_clean("session-1")
        assistant_turn = next(t for t in cleaned if t["role"] == "assistant")
        self.assertNotIn("[kubernetes.pdf]", assistant_turn["content"])
        self.assertIn("PV는 영구 볼륨입니다", assistant_turn["content"])

    def test_clean_turns_preserves_user_messages(self):
        pipeline = _build_pipeline()
        pipeline.session_repository.recent_turns.return_value = [
            ChatTurn(role="user", content="[첨부파일.pdf] 내용 알려줘"),
        ]
        cleaned = pipeline._build_prompt_recent_turns_clean("session-1")
        user_turn = next(t for t in cleaned if t["role"] == "user")
        self.assertEqual(user_turn["content"], "[첨부파일.pdf] 내용 알려줘")

    def test_build_llm_messages_uses_clean_turns_when_new_topic(self):
        pipeline = _build_pipeline()
        pipeline.session_repository.recent_turns.return_value = [
            ChatTurn(role="assistant", content="답변입니다.\n[doc.pdf] p.3"),
        ]
        messages = pipeline._build_llm_messages(
            session_id="sess",
            user_message="날씨 어때?",
            code_example_request=False,
            response_mode="general",
            turn_policy={},
            top_score=0.0,
            context_blocks=[],
            is_new_topic=True,
        )
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        for msg in assistant_msgs:
            self.assertNotIn("[doc.pdf]", msg["content"])


if __name__ == "__main__":
    unittest.main()
