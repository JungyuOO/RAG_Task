from __future__ import annotations

import asyncio
import unittest

from app.ocp_chat import OcpChatService
from app.rag.types import ChatTurn


class _AnswerServiceStub:
    def build_context_payload(
        self,
        rewritten_query: str,
        response_mode: str,
        top_score: float,
        preferred_preview_source,
        preview_pages,
        context_items,
        grounded_pages,
        answer_citations,
        preview_finalized: bool = False,
    ) -> dict:
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
            "items": [],
        }

    @staticmethod
    def public_context_payload(payload: dict) -> dict:
        return dict(payload)


class _SessionRepositoryStub:
    def __init__(self, topic_state: dict | None = None) -> None:
        self._topic_state = topic_state or {}
        self.turns: list[ChatTurn] = []

    def topic_state(self, _session_id: str) -> dict:
        return self._topic_state

    def add_turn(self, _session_id: str, role: str, content: str, metadata: dict | None = None) -> int:
        self.turns.append(ChatTurn(role=role, content=content, metadata=metadata))
        if role == "assistant" and metadata:
            self._topic_state = {
                **self._topic_state,
                "last_namespace": str((metadata.get("ocp_context") or {}).get("namespace") or self._topic_state.get("last_namespace") or ""),
                "last_ocp_resource": str((metadata.get("ocp_context") or {}).get("last_resource") or self._topic_state.get("last_ocp_resource") or ""),
                "last_ocp_resource_names": list((metadata.get("ocp_context") or {}).get("last_resource_names") or self._topic_state.get("last_ocp_resource_names") or []),
                "last_ocp_result_items": list((metadata.get("ocp_context") or {}).get("last_result_items") or self._topic_state.get("last_ocp_result_items") or []),
                "last_ocp_filter_keyword": str((metadata.get("ocp_context") or {}).get("last_filter_keyword") or self._topic_state.get("last_ocp_filter_keyword") or ""),
            }
        return len(self.turns)


class _OcpClientStub:
    enabled = True
    default_namespace = "demo"

    async def list_resources(self, resource: str, namespace: str | None = None) -> dict:
        ns = namespace or "demo"
        fixtures = {
            "pods": [
                {"name": "pandas-api-0", "namespace": ns, "kind": "Pod", "phase": "Running", "node_name": "worker-a"},
                {"name": "pandas-worker-0", "namespace": ns, "kind": "Pod", "phase": "Pending", "node_name": "worker-b"},
            ],
            "deployments": [
                {"name": "pandas", "namespace": ns, "kind": "Deployment", "ready_replicas": 1, "replicas": 1},
            ],
            "services": [
                {"name": "pandas", "namespace": ns, "kind": "Service", "type": "ClusterIP", "cluster_ip": "10.0.0.9"},
            ],
            "routes": [
                {"name": "pandas", "namespace": ns, "kind": "Route", "host": "pandas.example.com", "to": "pandas"},
            ],
            "events": [
                {"name": "pandas-warning", "namespace": ns, "kind": "Event", "type": "Warning", "phase": "BackOff", "host": "Pod", "to": "pandas-api-0"},
            ],
        }
        items = fixtures[resource]
        return {"resource": resource, "namespace": ns, "count": len(items), "items": items}

    async def get_resource_yaml(self, resource: str, name: str, namespace: str | None = None) -> dict:
        return {
            "resource": resource,
            "namespace": namespace or "demo",
            "name": name,
            "object": {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": name, "namespace": namespace or "demo"},
            },
        }


class _LlmPlannerStub:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def generate(self, messages, max_tokens=None) -> str:  # noqa: ANN001
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response


class _SequencedLlmStub:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, messages, max_tokens=None) -> str:  # noqa: ANN001
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if not self.responses:
            return ""
        return self.responses.pop(0)


class OcpChatServiceTests(unittest.TestCase):
    def _collect(self, iterator) -> list[dict]:
        async def _run():
            events = []
            async for event in iterator:
                events.append(event)
            return events

        return asyncio.run(_run())

    def test_status_query_is_routed_to_ocp(self) -> None:
        repository = _SessionRepositoryStub()
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertTrue(service.should_handle("s1", "demo namespace의 Pod 몇 개야?"))
        events = self._collect(service.stream("s1", "demo namespace의 Pod 몇 개야?"))

        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[1]["type"], "context")
        self.assertEqual(events[1]["answer_route"], "ocp_status")
        self.assertIn("Pod", events[2]["content"])
        self.assertEqual(repository.turns[-1].metadata["ocp_context"]["last_resource"], "pods")
        self.assertEqual(repository.turns[-1].metadata["ocp_context"]["namespace"], "demo")

    def test_count_query_does_not_treat_korean_count_word_as_filter(self) -> None:
        repository = _SessionRepositoryStub()
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        events = self._collect(service.stream("s1", "demo namespace pod 몇개야 ?"))

        self.assertEqual(events[1]["answer_route"], "ocp_status")
        self.assertIn("Pod은(는) 2개", events[2]["content"])
        self.assertNotIn("`몇개야`", events[2]["content"])

    def test_followup_yaml_uses_previous_ocp_context(self) -> None:
        repository = _SessionRepositoryStub(
            topic_state={
                "last_namespace": "demo",
                "last_ocp_resource": "pods",
                "last_ocp_resource_names": ["pandas-api-0"],
                "last_ocp_result_items": [{"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"}],
            }
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertTrue(service.should_handle("s1", "그 YAML 보여줘"))
        events = self._collect(service.stream("s1", "그 YAML 보여줘"))

        self.assertEqual(events[1]["answer_route"], "ocp_yaml")
        self.assertIn("```yaml", events[2]["content"])
        self.assertIn("pandas-api-0", events[2]["content"])

    def test_followup_yaml_with_multiple_candidates_requests_clarification(self) -> None:
        repository = _SessionRepositoryStub(
            topic_state={
                "last_namespace": "demo",
                "last_ocp_resource": "pods",
                "last_ocp_resource_names": ["pandas-api-0", "pandas-worker-0"],
                "last_ocp_result_items": [
                    {"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"},
                    {"resource": "pods", "name": "pandas-worker-0", "namespace": "demo", "kind": "Pod"},
                ],
            }
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        events = self._collect(service.stream("s1", "그 yaml 보여줘"))

        self.assertEqual(events[1]["answer_route"], "ocp_yaml")
        self.assertIn("어떤 것을 말씀하시는지", events[2]["content"])
        self.assertIn("pandas-api-0", events[2]["content"])

    def test_followup_yaml_uses_explicit_pod_name_when_user_provides_one(self) -> None:
        repository = _SessionRepositoryStub(
            topic_state={
                "last_namespace": "demo",
                "last_ocp_resource": "pods",
                "last_ocp_resource_names": ["pandas-api-0", "pandas-worker-0"],
                "last_ocp_result_items": [
                    {"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"},
                    {"resource": "pods", "name": "pandas-worker-0", "namespace": "demo", "kind": "Pod"},
                ],
            }
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        events = self._collect(service.stream("s1", "build-and-push-crxvmo-build-image-pod 이거 yaml 알려줘"))

        self.assertEqual(events[1]["answer_route"], "ocp_yaml")
        self.assertIn("```yaml", events[2]["content"])
        self.assertIn("build-and-push-crxvmo-build-image-pod", events[2]["content"])

    def test_document_yaml_followup_after_doc_context_stays_on_document_lane(self) -> None:
        repository = _SessionRepositoryStub(
            topic_state={
                "last_explicit_resources": ["pod"],
                "last_answer_route": "extractive_code",
            }
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertEqual(service.detect_query_mode("s1", "그거 yaml로 보려면?"), "document")

    def test_live_pod_list_after_doc_context_stays_on_ocp_lane(self) -> None:
        repository = _SessionRepositoryStub(
            topic_state={
                "last_explicit_resources": ["pod"],
                "last_answer_route": "extractive_code",
            }
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertEqual(service.detect_query_mode("s1", "지금 pandas 관련 pod 보여줘"), "ocp")

    def test_doc_to_live_bridge_without_ocp_context_uses_mixed_lane(self) -> None:
        repository = _SessionRepositoryStub(
            topic_state={
                "last_explicit_resources": ["pod"],
                "last_answer_route": "extractive_code",
            }
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertEqual(service.detect_query_mode("s1", "그럼 지금 내 ocp 쪽 namespace에서는 어떻게 확인해"), "mixed")

    def test_procedure_style_command_question_stays_on_document_lane(self) -> None:
        repository = _SessionRepositoryStub()
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertEqual(service.detect_query_mode("s1", "etcd pod 상태 확인 명령어 알려줘"), "document")

    def test_explanation_request_falls_back_to_document_lane(self) -> None:
        repository = _SessionRepositoryStub()
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
        )

        self.assertFalse(service.should_handle("s1", "Service와 Route 차이 설명해줘"))

    def test_complex_query_uses_agent_fallback_plan(self) -> None:
        repository = _SessionRepositoryStub()
        llm = _SequencedLlmStub(
            ["""{
                "mode":"summary",
                "resources":["events","deployments"],
                "namespace":"demo",
                "pattern":"pandas",
                "warning_only":true,
                "status_check":true,
                "followup":false,
                "confidence":0.91
            }""", "demo namespace 기준으로 pandas deployment는 정상이며 warning 이벤트 1건이 있습니다. 우선 이벤트 세부 내용을 확인하는 것이 좋습니다."]
        )
        service = OcpChatService(
            ocp_api_client=_OcpClientStub(),
            session_repository=repository,
            answer_service=_AnswerServiceStub(),
            llm_client=llm,
        )

        events = self._collect(service.stream("s1", "demo namespace에서 warning 이벤트랑 pandas deployment 상태 같이 요약해줘"))

        self.assertEqual(events[1]["query_interpretation"]["parse_strategy"], "agent_fallback")
        self.assertEqual(events[1]["query_interpretation"]["resources"], ["events", "deployments"])
        self.assertIn("warning 이벤트 1건", events[2]["content"])
        self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
