from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from apps.api.schemas.copilot_chat import CopilotChatHistoryTurn, CopilotChatResponse, CopilotChatSourceItem, CopilotChatStage
from apps.api.rag.query.query_rewrite_agent import QueryRewriteAgent
from apps.api.schemas.ocp_chat import OcpLiveChatResponse
from apps.api.schemas.ocp_live import OcpLiveResourceSummary
from apps.api.rag.query.intent_agent import IntentAgent
from apps.api.rag.generation.unified_copilot_service import UnifiedCopilotService


class _FakeLiveChatService:
    async def answer(self, **_: object) -> OcpLiveChatResponse:
        return OcpLiveChatResponse(
            connection_id="conn-1",
            cluster_url="https://api.cluster.example.com",
            mode="resource_list",
            resource="pods",
            namespace="demo",
            answer="demo namespace에서 pods 총 1개입니다. 예시: pod-a",
            items=[OcpLiveResourceSummary(name="pod-a", namespace="demo", kind="Pod")],
            artifacts=[],
        )


class _FakeDocumentRetriever:
    def __init__(self, response: CopilotChatResponse | None) -> None:
        self._response = response
        self.messages: list[str] = []
        self.allowed_source_paths: list[list[str] | None] = []

    async def answer(self, *, message: str, allowed_source_paths=None, additional_query=None) -> CopilotChatResponse | None:
        del additional_query
        self.messages.append(message)
        self.allowed_source_paths.append(list(allowed_source_paths) if allowed_source_paths is not None else None)
        return self._response


class _FakePgvectorBridge:
    def __init__(self, response: CopilotChatResponse | None) -> None:
        self._response = response
        self.messages: list[str] = []
        self.allowed_source_paths: list[list[str] | None] = []
        self.additional_queries: list[str | None] = []

    async def answer(self, *, message: str, allowed_source_paths=None, additional_query=None) -> CopilotChatResponse | None:
        self.messages.append(message)
        self.allowed_source_paths.append(list(allowed_source_paths) if allowed_source_paths is not None else None)
        self.additional_queries.append(additional_query)
        return self._response


class _FakeLlmClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.is_enabled = True
        self.settings = SimpleNamespace(
            use_query_router=False,
            use_native_citation_prompt=True,
            llm_synthesis_max_tokens=600,
            use_response_cache=True,
            force_korean_answers=True,
        )

    async def generate(self, messages, *, max_tokens=None, temperature=None, purpose=None):  # noqa: ANN001
        del messages, max_tokens, temperature, purpose
        if self.responses:
            return self.responses.pop(0)
        return ""

    async def stream_chat(self, messages, *, max_tokens=None, temperature=None, purpose=None):  # noqa: ANN001
        del messages, max_tokens, temperature, purpose
        if not self.responses:
            return
        text = self.responses.pop(0)
        for token in text.split(" "):
            yield token + " "


class UnifiedCopilotServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_finalize_cited_answer_reattaches_planned_citations(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient([]),
        )
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="auth.md · OAuth token duration",
                metadata={
                    "section_title": "OAuth token duration",
                    "preview_text": "Configure the internal OAuth server token duration",
                    "synthesis_text": "Configure the internal OAuth server token duration",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="auth.md · OAuth access tokens",
                metadata={
                    "section_title": "Listing user-owned OAuth access tokens",
                    "preview_text": "List user-owned OAuth access tokens",
                    "synthesis_text": "List user-owned OAuth access tokens",
                },
            ),
        ]

        result = service._finalize_cited_answer(
            "OAuth 토큰 유효 기간은 내부 OAuth 서버에서 조정합니다[2]",
            sources,
            paragraph_source_indexes=[[0]],
        )

        self.assertIn("[1]", result)
        self.assertNotIn("[2]", result)

    def test_prune_sources_to_citations_keeps_supporting_uncited_source(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient([]),
        )
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="service_mesh.md · Validating",
                source_path="official/en/service_mesh.md",
                metadata={
                    "section_title": "2.9.1.3. Validating your SMCP installation with the CLI",
                    "preview_text": "Validate your SMCP installation with the CLI.",
                    "synthesis_text": "Validate your SMCP installation with the CLI.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="kubernetes_nmstate.md · NodeNetworkState",
                source_path="official/en/kubernetes_nmstate.md",
                metadata={
                    "section_title": "1.1. Viewing the network state of a node by using the CLI",
                    "preview_text": "Use oc get nns to view NodeNetworkState resources.",
                    "synthesis_text": "Use oc get nns to view NodeNetworkState resources.",
                },
            ),
        ]

        answer, pruned = service._prune_sources_to_citations(
            "NodeNetworkState 객체를 확인하려면 `oc get nns` 를 사용합니다.[1]",
            sources,
        )

        self.assertEqual(answer.count("[1]"), 1)
        self.assertEqual(len(pruned), 2)

    def test_prune_sources_to_citations_drops_irrelevant_uncited_source(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient([]),
        )
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="auth.md · OAuth token list",
                source_path="official/en/authentication_and_authorization.md",
                metadata={
                    "section_title": "5.1. Listing user-owned OAuth access tokens",
                    "preview_text": "Use oc get useroauthaccesstokens to list tokens.",
                    "synthesis_text": "Use oc get useroauthaccesstokens to list user-owned OAuth access tokens.",
                },
            ),
            CopilotChatSourceItem(
                source_type="doc",
                label="auth.md · OIDC",
                source_path="official/en/authentication_and_authorization.md",
                metadata={
                    "section_title": "Chapter 8. Enabling direct authentication with an external OIDC identity provider",
                    "preview_text": "OIDC providers integrate with OpenShift.",
                    "synthesis_text": "OIDC providers integrate with OpenShift.",
                },
            ),
        ]

        _answer, pruned = service._prune_sources_to_citations(
            "유저가 소유한 OAuth 액세스 토큰은 `oc get useroauthaccesstokens` 로 조회합니다.[1]",
            sources,
        )

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].metadata["section_title"], "5.1. Listing user-owned OAuth access tokens")

    async def test_deterministic_rerank_promotes_matching_section(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient([]),
        )
        response = CopilotChatResponse(
            lane="doc_hybrid",
            mode="hybrid_rrf_doc",
            answer="raw",
            sources=[
                {
                    "source_type": "doc",
                    "label": "auth.md · OIDC",
                    "source_path": "official/en/authentication_and_authorization.md",
                    "chunk_id": "chunk-1",
                    "score": 0.96,
                    "metadata": {
                        "section_title": "Chapter 8. Enabling direct authentication with an external OIDC identity provider",
                        "preview_text": "OIDC providers integrate with OpenShift.",
                        "synthesis_text": "OIDC providers integrate with OpenShift.",
                    },
                },
                {
                    "source_type": "doc",
                    "label": "auth.md · Token list",
                    "source_path": "official/en/authentication_and_authorization.md",
                    "chunk_id": "chunk-2",
                    "score": 0.92,
                    "metadata": {
                        "section_title": "5.1. Listing user-owned OAuth access tokens",
                        "preview_text": "Use oc get useroauthaccesstokens to list tokens.",
                        "synthesis_text": "Use oc get useroauthaccesstokens to list user-owned OAuth access tokens.",
                    },
                },
            ],
        )

        reranked = await service._rerank_doc_response(
            message="유저가 가진 oauth 토큰 리스트 뽑는 방법 oauth access token listing user-owned",
            response=response,
        )

        self.assertEqual(
            reranked.sources[0].metadata["section_title"],
            "5.1. Listing user-owned OAuth access tokens",
        )

    async def test_synthesis_planning_uses_retrieval_query_for_source_order(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient(["사용자 소유 OAuth 액세스 토큰은 `oc get useroauthaccesstokens` 로 조회할 수 있습니다.[1]"]),
        )
        response = CopilotChatResponse(
            lane="doc_hybrid",
            mode="hybrid_rrf_doc",
            answer="raw",
            sources=[
                {
                    "source_type": "doc",
                    "label": "auth.md · OIDC",
                    "source_path": "official/en/authentication_and_authorization.md",
                    "relative_source_path": "data/corpus/pdfs/official/en/authentication_and_authorization.md",
                    "chunk_id": "chunk-1",
                    "score": 0.96,
                    "provenance": ["doc_new"],
                    "metadata": {
                        "section_title": "Chapter 8. Enabling direct authentication with an external OIDC identity provider",
                        "preview_text": "OIDC providers integrate with OpenShift.",
                        "synthesis_text": "OIDC providers integrate with OpenShift.",
                    },
                },
                {
                    "source_type": "doc",
                    "label": "auth.md · OAuth token list",
                    "source_path": "official/en/authentication_and_authorization.md",
                    "relative_source_path": "data/corpus/pdfs/official/en/authentication_and_authorization.md",
                    "chunk_id": "chunk-2",
                    "score": 0.92,
                    "provenance": ["doc_new"],
                    "metadata": {
                        "section_title": "5.1. Listing user-owned OAuth access tokens",
                        "preview_text": "Use oc get useroauthaccesstokens to list tokens.",
                        "synthesis_text": "Use oc get useroauthaccesstokens to list user-owned OAuth access tokens.",
                    },
                },
            ],
        )

        result = await service._synthesize_doc_response(
            message="유저가 가진 oauth 토큰 리스트 뽑는 방법",
            planning_message="유저가 가진 oauth 토큰 리스트 뽑는 방법 oauth access token listing user-owned",
            response=response,
        )

        self.assertEqual(
            result.sources[0].metadata["section_title"],
            "5.1. Listing user-owned OAuth access tokens",
        )

    async def test_llm_grounded_doc_synthesis_falls_back_to_extractive_when_citations_are_ungrounded(self) -> None:
        retriever = _FakeDocumentRetriever(
            CopilotChatResponse(
                lane="doc_hybrid",
                mode="hybrid_rrf_doc",
                answer="raw doc answer",
                sources=[
                    {
                        "source_type": "doc",
                        "label": "auth.md · OAuth token duration",
                        "source_path": "official/en/authentication_and_authorization.md",
                        "relative_source_path": "data/corpus/pdfs/official/en/authentication_and_authorization.md",
                        "chunk_id": "chunk-1",
                        "score": 0.95,
                        "provenance": ["doc_new"],
                        "metadata": {
                            "preview_text": "Configure the internal OAuth server token duration.",
                            "synthesis_text": "Configure the internal OAuth server token duration.",
                            "section_title": "3.4. Configuring the internal OAuth server’s token duration",
                        },
                    }
                ],
            )
        )
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=retriever,
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient(["General platform overview unrelated to token duration.[1]"]),
        )

        response = await service.answer(
            message="oauth 토큰 유효 기간 자체를 늘리려면 어디 설정해?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.mode, "hybrid_rrf_doc")
        self.assertNotIn("unrelated", response.answer.lower())

    async def test_llm_grounded_doc_synthesis_rejects_unsupported_command_hallucination(self) -> None:
        retriever = _FakeDocumentRetriever(
            CopilotChatResponse(
                lane="doc_hybrid",
                mode="hybrid_rrf_doc",
                answer="raw doc answer",
                sources=[
                    {
                        "source_type": "doc",
                        "label": "auth.md · OAuth token list",
                        "source_path": "official/en/authentication_and_authorization.md",
                        "relative_source_path": "data/corpus/pdfs/official/en/authentication_and_authorization.md",
                        "chunk_id": "chunk-1",
                        "score": 0.95,
                        "provenance": ["doc_new"],
                        "metadata": {
                            "preview_text": "Use oc get useroauthaccesstokens to list user-owned OAuth access tokens.",
                            "synthesis_text": "Use oc get useroauthaccesstokens to list user-owned OAuth access tokens.",
                            "section_title": "5.1. Listing user-owned OAuth access tokens",
                        },
                    }
                ],
            )
        )
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=retriever,
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient(["특정 유저 세션 종료는 `oc logout` 으로 처리합니다.[1]"]),
        )

        response = await service.answer(
            message="특정 유저 oauth 토큰 삭제해서 세션 끊으려면?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertNotIn("oc logout", response.answer)

    async def test_emits_doc_progress_stages_in_processing_order(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="new doc answer",
                    sources=[],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
        )
        stages: list[CopilotChatStage] = []

        response = await service.answer(
            message="OpenShift route가 뭐야?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
            progress=stages.append,
        )

        self.assertEqual(response.lane, "doc_new")
        self.assertEqual(
            [stage.key for stage in stages],
            [
                "analyze_question",
                "retrieve_keyword_docs",
                "retrieve_vector_docs",
                "synthesize_sources",
                "finalize_answer",
            ],
        )

    async def test_disabled_response_cache_skips_cache_calls(self) -> None:
        cache = MagicMock()
        llm = _FakeLlmClient([])
        llm.settings.use_response_cache = False
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="new doc answer",
                    sources=[],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=llm,
            response_cache=cache,
        )

        await service.answer(
            message="route 설명해줘",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        cache.get.assert_not_called()
        cache.set.assert_not_called()

    async def test_source_hints_are_forwarded_to_dense_retrieval(self) -> None:
        retriever = _FakeDocumentRetriever(None)
        bridge = _FakePgvectorBridge(None)
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=retriever,
            pgvector_bridge=bridge,
            llm_client=_FakeLlmClient([]),
        )

        await service.answer(
            message="jenkins 가 다른 프로젝트 리소스에 접근하게 하려면?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertIsNotNone(bridge.allowed_source_paths[-1])
        self.assertTrue(
            any(
                path.endswith("official\\en\\jenkins.md") or path.endswith("official/en/jenkins.md")
                for path in (bridge.allowed_source_paths[-1] or [])
            )
        )

    async def test_prefers_sparse_when_hinted_sources_disagree_with_dense(self) -> None:
        sparse = _FakeDocumentRetriever(
            CopilotChatResponse(
                lane="doc_new",
                mode="keyword_retrieval",
                answer="sparse answer",
                sources=[
                    {
                        "source_type": "doc",
                        "label": "ingress_and_load_balancing.md · Switching",
                        "source_path": str((Path.cwd() / "data" / "corpus" / "pdfs" / "official" / "en" / "ingress_and_load_balancing.md").resolve()),
                        "relative_source_path": "data/corpus/pdfs/official/en/ingress_and_load_balancing.md",
                        "chunk_id": "chunk-1",
                        "score": 0.8,
                        "provenance": ["doc_new"],
                        "metadata": {
                            "section_title": "2.6.2.1. Switching the Ingress Controller from using a Classic Load Balancer to a Network Load Balancer",
                            "preview_text": "Switch the ingress controller from CLB to NLB.",
                            "synthesis_text": "Switch the ingress controller from CLB to NLB.",
                        },
                    }
                ],
            )
        )
        dense = _FakePgvectorBridge(
            CopilotChatResponse(
                lane="doc_pgvector",
                mode="pgvector_dense",
                answer="dense answer",
                sources=[
                    {
                        "source_type": "doc",
                        "label": "security_and_compliance.md · Wrong",
                        "source_path": str((Path.cwd() / "data" / "corpus" / "pdfs" / "official" / "en" / "security_and_compliance.md").resolve()),
                        "relative_source_path": "data/corpus/pdfs/official/en/security_and_compliance.md",
                        "chunk_id": "chunk-2",
                        "score": 0.95,
                        "provenance": ["doc_pgvector"],
                        "metadata": {
                            "section_title": "Wrong section",
                            "preview_text": "Wrong context.",
                            "synthesis_text": "Wrong context.",
                        },
                    }
                ],
            )
        )
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=sparse,
            pgvector_bridge=dense,
            llm_client=_FakeLlmClient([]),
        )

        response = await service.answer(
            message="classic load balancer 쓰던걸 network load balancer 로 바꾸고 싶은데",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.mode, "keyword_retrieval")
        self.assertTrue(
            any(
                str(source.source_path).endswith("official/en/ingress_and_load_balancing.md")
                or str(source.source_path).endswith("official\\en\\ingress_and_load_balancing.md")
                for source in response.sources
            )
        )

    async def test_emits_live_progress_stages_for_live_query(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
        )
        stages: list[CopilotChatStage] = []

        response = await service.answer(
            message="show pods",
            connection_id="conn-1",
            namespace="demo",
            broker=None,  # type: ignore[arg-type]
            progress=stages.append,
        )

        self.assertEqual(response.lane, "live")
        self.assertEqual(
            [stage.key for stage in stages],
            [
                "analyze_question",
                "route_live",
                "finalize_answer",
            ],
        )

    async def test_routes_live_when_connection_and_live_markers_exist(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="show pods",
            connection_id="conn-1",
            namespace="demo",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "live")
        self.assertEqual(response.sources[0].label, "pod-a")
        self.assertEqual(response.sources[0].provenance, ["live"])

    async def test_routes_doc_when_no_live_markers(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="new doc answer",
                    sources=[],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="OpenShift route가 뭐야?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "doc_new")
        self.assertIn("new doc answer", response.answer)

    async def test_routes_doc_for_conceptual_resource_question_even_with_connection(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="pod 구성 방식 문서 응답",
                    sources=[],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="Explain pod configuration patterns",
            connection_id="conn-1",
            namespace="demo",
            recent_turns=[],
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "doc_new")
        self.assertIn("구성 방식", response.answer)

    async def test_routes_mixed_when_question_requests_doc_and_live_together(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="pod 구성 방식 문서 응답",
                    sources=[],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="Explain pod configuration patterns and also show pods in the current cluster",
            connection_id="conn-1",
            namespace="demo",
            recent_turns=[],
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "mixed")
        self.assertIn("pod-a", response.answer)
        self.assertNotIn("Explain pod configuration patterns and also show pods in the current cluster", response.answer)

    async def test_returns_guidance_when_live_query_has_no_connection(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="show pods",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "needs_connection")
        self.assertIn("Connection", response.answer)

    async def test_followup_inherits_previous_doc_lane(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="이전 문서 후속 질문 응답",
                    sources=[],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="Can you explain that in more detail?",
            connection_id="conn-1",
            namespace="demo",
            recent_turns=[
                CopilotChatHistoryTurn(role="assistant", text="Previous document answer", lane="doc_hybrid"),
            ],
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "doc_new")

    async def test_routes_live_for_yaml_detail_request_with_name_like_token(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="show me my-app-123 yaml",
            connection_id="conn-1",
            namespace="demo",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "live")

    async def test_returns_doc_no_match_when_new_retrievers_have_no_match(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(None),
            pgvector_bridge=_FakePgvectorBridge(None),
        )

        response = await service.answer(
            message="OpenShift route가 뭐야?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "doc_new")
        self.assertEqual(response.mode, "no_match")
        self.assertIn("새 문서 retrieval 경로에서 바로 사용할 근거를 찾지 못했습니다.", response.answer)

    async def test_prefers_pgvector_for_conceptual_doc_question_when_scores_are_stronger(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="new doc answer",
                    sources=[
                        {
                            "source_type": "doc",
                            "label": "doc",
                            "source_path": "manual.md",
                            "relative_source_path": "data/manual.md",
                            "chunk_id": "chunk-1",
                            "score": 1.0,
                            "provenance": ["doc_new"],
                            "metadata": {"preview_text": "new doc preview", "section_title": "New Doc"},
                        }
                    ],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(
                CopilotChatResponse(
                    lane="doc_pgvector",
                    mode="pgvector_dense",
                    answer="pgvector answer",
                    sources=[
                        {
                            "source_type": "doc",
                            "label": "pg",
                            "source_path": "manual.md",
                            "relative_source_path": "data/manual.md",
                            "chunk_id": "chunk-2",
                            "score": 2.0,
                            "provenance": ["doc_pgvector"],
                            "metadata": {"preview_text": "pgvector preview", "section_title": "PG Doc"},
                        }
                    ],
                )
            ),
        )

        response = await service.answer(
            message="route가 뭐야?",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "doc_hybrid")
        self.assertEqual(response.mode, "hybrid_rrf_doc")
        self.assertIn("pgvector preview", response.answer)
        self.assertGreaterEqual(len(response.sources), 2)
        self.assertTrue(any(source.provenance for source in response.sources))

    async def test_llm_grounded_doc_synthesis_overrides_extract_answer_when_configured(self) -> None:
        retriever = _FakeDocumentRetriever(
            CopilotChatResponse(
                lane="doc_new",
                mode="keyword_retrieval",
                answer="raw doc answer",
                sources=[
                    {
                        "source_type": "doc",
                        "label": "doc",
                        "source_path": "manual.md",
                        "relative_source_path": "data/manual.md",
                        "chunk_id": "chunk-1",
                        "score": 1.0,
                        "provenance": ["doc_new"],
                        "metadata": {"preview_text": "pod 구성은 deployment 기반이 일반적입니다.", "section_title": "Pod 구성"},
                    }
                ],
            )
        )
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=retriever,
            pgvector_bridge=_FakePgvectorBridge(None),
            intent_agent=IntentAgent(),
            query_rewrite_agent=QueryRewriteAgent(),
            llm_client=_FakeLlmClient(['이것은 요약된 응답입니다.[1]']),
        )

        response = await service.answer(
            message="Explain pod configuration patterns",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.answer, "이것은 요약된 응답입니다.[1]")
        self.assertEqual(response.mode, "keyword_retrieval_llm_grounded")

    async def test_llm_grounded_doc_synthesis_assigns_paragraph_citations_from_plan(self) -> None:
        retriever = _FakeDocumentRetriever(
            CopilotChatResponse(
                lane="doc_new",
                mode="keyword_retrieval",
                answer="raw doc answer",
                sources=[
                    {
                        "source_type": "doc",
                        "label": "nodes.md · About pods",
                        "source_path": "official/en/nodes.md",
                        "relative_source_path": "data/official/en/nodes.md",
                        "chunk_id": "chunk-1",
                        "score": 0.95,
                        "provenance": ["doc_new"],
                        "metadata": {
                            "preview_text": "Pod specs define containers and metadata.",
                            "synthesis_text": "Pod specs define containers, metadata, and restart policy.",
                            "section_title": "About pods",
                        },
                    },
                    {
                        "source_type": "doc",
                        "label": "pipelines.md · Creating pipelines",
                        "source_path": "official/en/pipelines.md",
                        "relative_source_path": "data/official/en/pipelines.md",
                        "chunk_id": "chunk-2",
                        "score": 0.91,
                        "provenance": ["doc_new"],
                        "metadata": {
                            "preview_text": "Pipelines define CI/CD tasks and resources.",
                            "synthesis_text": "Pipelines define CI/CD tasks, resources, and execution flow.",
                            "section_title": "Creating pipelines",
                        },
                    },
                ],
            )
        )
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=retriever,
            pgvector_bridge=_FakePgvectorBridge(None),
            intent_agent=IntentAgent(),
            query_rewrite_agent=QueryRewriteAgent(),
            llm_client=_FakeLlmClient(
                [
                    "파드는 컨테이너와 메타데이터를 정의합니다.\n\n파이프라인은 CI/CD 작업 흐름을 정의합니다."
                ]
            ),
        )

        response = await service.answer(
            message="pod configuration and ci cd pipeline explain",
            connection_id="",
            namespace="",
            broker=None,  # type: ignore[arg-type]
        )

        self.assertIn("정의합니다.[1]", response.answer)
        self.assertIn("정의합니다.[2]", response.answer)
        self.assertEqual(len(response.sources), 2)

    async def test_llm_planner_rewrites_doc_and_live_queries(self) -> None:
        retriever = _FakeDocumentRetriever(
            CopilotChatResponse(
                lane="doc_new",
                mode="keyword_retrieval",
                answer="doc answer",
                sources=[],
            )
        )
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=retriever,
            pgvector_bridge=_FakePgvectorBridge(None),
            llm_client=_FakeLlmClient([
                '{"lane":"mixed","doc_query":"pod 구성 방식 문서 질의","live_query":"show pods in demo namespace"}',
                '최종 혼합 응답',
            ]),
        )

        await service.answer(
            message="pod 구성 방식 설명하고 show pods 도 해줘",
            connection_id="conn-1",
            namespace="demo",
            recent_turns=[],
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(retriever.messages[-1], "pod 구성 방식 문서 질의")

    async def test_llm_mixed_synthesis_produces_single_answer_when_configured(self) -> None:
        service = UnifiedCopilotService(
            live_chat_service=_FakeLiveChatService(),
            document_retriever=_FakeDocumentRetriever(
                CopilotChatResponse(
                    lane="doc_new",
                    mode="keyword_retrieval",
                    answer="pod 구성 방식 문서 응답",
                    sources=[
                        {
                            "source_type": "doc",
                            "label": "doc",
                            "source_path": "manual.md",
                            "relative_source_path": "data/manual.md",
                            "chunk_id": "chunk-1",
                            "score": 1.0,
                            "provenance": ["doc_new"],
                            "metadata": {"preview_text": "pod 구성은 deployment 기반이 일반적입니다.", "section_title": "Pod 구성"},
                        }
                    ],
                )
            ),
            pgvector_bridge=_FakePgvectorBridge(None),
            query_rewrite_agent=QueryRewriteAgent(),
            llm_client=_FakeLlmClient([
                '{"lane":"mixed"}',
                '문서 기준으로는 deployment 구성이 일반적이고[1], 현재 클러스터에는 pod-a 가 있습니다.',
            ]),
        )

        response = await service.answer(
            message="pod 구성 방식 설명하고 show pods 도 해줘",
            connection_id="conn-1",
            namespace="demo",
            recent_turns=[],
            broker=None,  # type: ignore[arg-type]
        )

        self.assertEqual(response.lane, "mixed")
        self.assertEqual(response.mode, "doc_plus_live_llm")
        self.assertIn("deployment", response.answer)


if __name__ == "__main__":
    unittest.main()


