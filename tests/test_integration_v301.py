"""v3.0.1 전체 기능 통합 테스트"""
import unittest
from unittest.mock import AsyncMock, MagicMock
import asyncio


class TestV301Integration(unittest.TestCase):
    """v3.0.1 전체 기능 통합 테스트"""

    def test_agent_pipeline_classify_intent(self):
        """IntentAgent 분류 → 의도 반환"""
        from app.llm.intent_agent import IntentAgent
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value='{"intent": "rag", "search_query": "OCP Pod", "keywords": ["OCP", "Pod"], "confidence": 0.9}')
        agent = IntentAgent(llm_client=mock_llm)
        result = asyncio.run(agent.classify("OCP Pod 배포 방법", context={}))
        self.assertEqual(result["intent"], "rag")
        self.assertIn("search_query", result)

    def test_agent_pipeline_expand_query(self):
        """RetrievalAgent 쿼리 확장"""
        from app.llm.retrieval_agent import RetrievalAgent
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value='{"expanded_query": "OpenShift Pod deployment YAML", "alternatives": ["OCP Pod 배포"], "target_versions": [], "multi_source": false}')
        agent = RetrievalAgent(llm_client=mock_llm)
        result = asyncio.run(agent.expand("OCP Pod 배포", intent_result={"intent": "rag"}, available_sources=[]))
        self.assertIn("expanded_query", result)
        self.assertIsInstance(result["alternatives"], list)

    def test_version_manager_detect_and_filter(self):
        """VersionManager: 버전 감지 → 청크 필터링"""
        from app.rag.version_manager import VersionManager
        vm = VersionManager()
        version = vm.detect_version("ocp-4.14-networking.pdf")
        self.assertEqual(version, "4.14")
        chunks = [
            {"chunk_id": 1, "version_id": 10},
            {"chunk_id": 2, "version_id": 11},
        ]
        filtered = vm.filter_by_version(chunks, ["4.14"], {10: "4.14", 11: "4.15"})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["chunk_id"], 1)

    def test_multi_source_citation_parse_and_html(self):
        """CitationExtractor: 인용 태그 파싱 → HTML 변환"""
        from app.rag.answer_citation import CitationExtractor
        extractor = CitationExtractor()
        text = "Pod는 기본 단위입니다. [source:ocp-pods.pdf:p5:L10-20] 여러 컨테이너를 포함합니다. [source:ocp-networking.pdf:p3:L1-5]"
        rendered, citations = extractor.render_with_html_tags(text)
        self.assertEqual(len(citations), 2)
        self.assertNotIn("[source:", rendered)
        self.assertIn("citation-tag", rendered)
        self.assertEqual(citations[0]["file_name"], "ocp-pods.pdf")
        self.assertEqual(citations[1]["file_name"], "ocp-networking.pdf")

    def test_status_events_defined(self):
        """SSE status 이벤트 상수 정의 확인"""
        from app.rag.pipeline_streaming import STAGE_MESSAGES
        required_stages = ["analyzing_intent", "searching_documents", "generating_answer"]
        for stage in required_stages:
            self.assertIn(stage, STAGE_MESSAGES)
            self.assertIsInstance(STAGE_MESSAGES[stage], str)

    def test_procedure_flow_check(self):
        """PipelineOrchestrator: 절차 감지"""
        from app.rag.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        mock_answer = MagicMock()
        mock_answer.check_procedure = AsyncMock(return_value={
            "has_procedure": True, "total_steps": 5, "offer_message": "5단계로 설명해드릴까요?"
        })
        orch.answer_agent = mock_answer
        result = asyncio.run(orch.check_procedure("OCP 배포 단계", [{"text": "1단계...5단계"}]))
        self.assertTrue(result["has_procedure"])
        self.assertEqual(result["total_steps"], 5)

    def test_version_store_create_and_list(self):
        """VersionStore: 버전 생성 및 조회"""
        from app.storage.version_store import VersionStore
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_conn.fetchrow = AsyncMock(return_value={"version_id": 1})
        mock_conn.fetch = AsyncMock(return_value=[{"version_id": 1, "version_tag": "4.14"}])
        mock_pool.acquire.return_value = mock_conn
        store = VersionStore(pool=mock_pool)
        created = asyncio.run(store.create_version("/path/ocp-4.14.pdf", "ocp-4.14.pdf", "4.14"))
        self.assertEqual(created["version_id"], 1)
        versions = asyncio.run(store.list_versions())
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["version_tag"], "4.14")

    def test_retrieval_version_filter_integration(self):
        """HybridRetriever: 버전 필터 적용 검색"""
        from app.rag.retrieval import HybridRetriever
        retriever = HybridRetriever(
            top_k=5, candidate_pool_size=20,
            bm25_k1=1.5, bm25_b=0.75,
            rerank_base_weight=0.7, rerank_overlap_weight=0.3,
        )
        items = [
            {"chunk": {"chunk_id": "1", "text": "OCP pod deployment", "tokens": ["ocp", "pod"], "version_id": 10}, "vector": [0.1]*768},
            {"chunk": {"chunk_id": "2", "text": "OCP service networking", "tokens": ["ocp", "service"], "version_id": 11}, "vector": [0.1]*768},
        ]
        results = retriever.search_rrf("pod", [0.1]*768, items, target_versions=["4.14"], version_map={10: "4.14", 11: "4.15"})
        chunk_ids = [r["chunk"]["chunk_id"] for r in results]
        self.assertIn("1", chunk_ids)
        self.assertNotIn("2", chunk_ids)


if __name__ == "__main__":
    unittest.main()
