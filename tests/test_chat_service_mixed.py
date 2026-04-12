from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.dependencies import ChatService


class _AnswerServiceStub:
    def build_extractive_code_answer(self, context_items, requested_resource_kinds=None, **kwargs):  # noqa: ANN001, ANN003
        del context_items, requested_resource_kinds, kwargs
        return "```bash\noc get pods -n demo | grep pandas\n```"

    def build_context_payload(
        self,
        rewritten_query,
        response_mode,
        top_score,
        preferred_preview_source,
        preview_pages,
        context_items,
        grounded_pages,
        answer_citations,
        preview_finalized=False,
    ):  # noqa: ANN001
        return {
            "query": rewritten_query,
            "mode": response_mode,
            "top_score": top_score,
            "preferred_preview_source": preferred_preview_source,
            "preview_pages": preview_pages,
            "source_grounding": [],
            "grounded_pages": grounded_pages,
            "answer_citations": answer_citations,
            "preview_finalized": preview_finalized,
            "items": list(context_items or []),
        }

    @staticmethod
    def public_context_payload(payload: dict) -> dict:
        return dict(payload)


class _PipelineStub:
    def __init__(self) -> None:
        self.answer_service = _AnswerServiceStub()

    async def _prepare_retrieval_state(  # noqa: ANN001
        self,
        session_id,
        user_message,
        allowed_source_paths=None,
        *,
        uploaded_source_paths=None,
        version_tag=None,
        turn_context=None,
    ):
        del session_id, user_message, allowed_source_paths, uploaded_source_paths, version_tag, turn_context
        return {
            "query_interpretation": {
                "intent": "cli_example",
                "response_shape": "code",
                "format_constraints": ["cli"],
                "resources": ["pod"],
                "actions": ["explain"],
            },
            "turn_policy": {
                "turn_type": "document_query",
                "response_mode": "rag",
                "use_retrieval": True,
                "use_memory_rewrite": False,
                "allow_preview": True,
                "allow_citations": True,
            },
            "rewritten_query": "pandas pod command",
            "top_score": 0.32,
            "use_retrieved_context": True,
            "grounded_pages": [{"source_path": "/docs/ocp.html", "page_number": 12}],
            "ordered_context_items": [
                {
                    "chunk": {
                        "chunk_id": "c1",
                        "source_path": "/docs/ocp.html",
                        "text": "```bash\noc get pods -n demo | grep pandas\n```",
                        "page_number": 12,
                        "metadata": {"page_start": 12, "page_end": 12, "block_types": "code"},
                    }
                }
            ],
            "selected_context_items": [
                {
                    "chunk": {
                        "chunk_id": "c1",
                        "source_path": "/docs/ocp.html",
                        "text": "```bash\noc get pods -n demo | grep pandas\n```",
                        "page_number": 12,
                        "metadata": {"page_start": 12, "page_end": 12, "block_types": "code"},
                    }
                }
            ],
            "preferred_preview_source": "/docs/ocp.html",
            "response_mode": "rag",
            "doc_type": "",
        }

    @staticmethod
    def _resolve_answer_route(query_interpretation):  # noqa: ANN001
        return "extractive_code"

    @staticmethod
    def _resolve_requested_resource_kinds(query_interpretation):  # noqa: ANN001
        return {str(value).casefold() for value in query_interpretation.get("resources", []) if value}

    @staticmethod
    def _select_code_example_context_items(user_message, query_interpretation, ordered_context_items, selected_context_items):  # noqa: ANN001
        del user_message, query_interpretation, selected_context_items
        return list(ordered_context_items)

    def _finalize_answer(self, **kwargs):  # noqa: ANN003
        answer = kwargs["answer"]
        payload = self.answer_service.build_context_payload(
            kwargs["rewritten_query"],
            kwargs["response_mode"],
            kwargs["top_score"],
            kwargs["preferred_preview_source"],
            [{"page_number": 12}],
            kwargs["selected_context_items"],
            kwargs["grounded_pages"],
            [{"source_path": "/docs/ocp.html", "page_number": 12}],
            preview_finalized=True,
        )
        payload["answer_route"] = kwargs["answer_route"]
        payload["query_interpretation"] = kwargs["query_interpretation"]
        return answer, payload["answer_citations"], payload

    def stream_chat(self, **kwargs):  # noqa: ANN003
        raise AssertionError("mixed requests should not fall back to plain pipeline stream")


class _SessionRepositoryStub:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str, dict | None]] = []

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> int:
        del session_id
        self.turns.append((role, content, metadata))
        return len(self.turns)


class _OcpChatServiceStub:
    def detect_query_mode(self, session_id: str, user_message: str) -> str:
        del session_id, user_message
        return "mixed"

    async def _build_plan(self, session_id: str, user_message: str):  # noqa: ANN001
        del session_id, user_message
        return SimpleNamespace(
            mode="summary",
            resources=["pods"],
            namespace="demo",
            pattern="pandas",
            warning_only=False,
            status_check=False,
            followup=False,
            parse_strategy="rule_fastpath",
            confidence=0.92,
        )

    async def _execute_plan(self, plan):  # noqa: ANN001
        return {
            "answer_route": "ocp_status",
            "answer": "demo namespace에서 pandas 관련 Pod는 2개입니다.",
            "ocp_context": {
                "namespace": plan.namespace,
                "last_resource": "pods",
                "last_resource_names": ["pandas-api-0", "pandas-worker-0"],
                "last_result_items": [
                    {"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"},
                    {"resource": "pods", "name": "pandas-worker-0", "namespace": "demo", "kind": "Pod"},
                ],
                "last_filter_keyword": plan.pattern,
                "parse_strategy": plan.parse_strategy,
                "confidence": plan.confidence,
            },
        }

    def should_handle(self, session_id: str, user_message: str) -> bool:
        del session_id, user_message
        return False

    def stream(self, **kwargs):  # noqa: ANN003
        raise AssertionError("mixed requests should not use pure ocp stream")


class ChatServiceMixedTests(unittest.TestCase):
    def _collect(self, iterator) -> list[dict]:
        async def _run():
            items = []
            async for event in iterator:
                items.append(event)
            return items

        return asyncio.run(_run())

    def test_mixed_route_combines_document_command_and_live_ocp_result(self) -> None:
        session_repository = _SessionRepositoryStub()
        service = ChatService(
            pipeline=_PipelineStub(),
            session_repository=session_repository,
            ocp_chat_service=_OcpChatServiceStub(),
        )
        request = SimpleNamespace(
            session_id="s1",
            message="지금 내 pandas 보려면 어떤 명령어 쳐야 돼?",
            allowed_source_paths=None,
            uploaded_source_paths=None,
            append_user_turn=True,
            version_tag=None,
        )

        events = self._collect(service.stream(request))

        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[1]["type"], "status")
        self.assertEqual(events[2]["type"], "context")
        self.assertEqual(events[2]["answer_route"], "mixed_doc_ocp")
        self.assertIn("문서 기준 명령어", events[3]["content"])
        self.assertIn("oc get pods -n demo | grep pandas", events[3]["content"])
        self.assertIn("현재 OCP 결과", events[3]["content"])
        self.assertIn("pandas 관련 Pod는 2개", events[3]["content"])
        self.assertEqual(session_repository.turns[0][0], "user")
        self.assertEqual(session_repository.turns[-1][0], "assistant")

    def test_compare_style_mixed_answer_adds_comparison_sections(self) -> None:
        service = ChatService(
            pipeline=_PipelineStub(),
            session_repository=_SessionRepositoryStub(),
            ocp_chat_service=_OcpChatServiceStub(),
        )

        answer = service._compose_mixed_answer(
            user_message="그 pod yaml이랑 공식 문서의 pod yaml은 뭐가 달라?",
            document_answer="oc get pod <pod_name> -o yaml",
            ocp_answer="현재 조회된 pods 중 어떤 것을 말씀하시는지 먼저 알려 주세요. 예: a, b",
            ocp_answer_route="ocp_yaml",
        )

        self.assertIn("공식 문서 기준", answer)
        self.assertIn("현재 OCP 기준", answer)
        self.assertIn("비교 가이드", answer)

    def test_compare_style_mixed_document_query_prefers_generic_yaml_lookup(self) -> None:
        service = ChatService(
            pipeline=_PipelineStub(),
            session_repository=_SessionRepositoryStub(),
            ocp_chat_service=_OcpChatServiceStub(),
        )

        query = service._build_mixed_document_query(
            "그 pod yaml이랑 공식 문서의 pod yaml은 뭐가 달라?",
            SimpleNamespace(mode="yaml", resources=["pods"], status_check=False, warning_only=False),
        )

        self.assertIn("pod yaml", query)
        self.assertIn("oc describe pod", query)
        self.assertIn("official example", query)


if __name__ == "__main__":
    unittest.main()
