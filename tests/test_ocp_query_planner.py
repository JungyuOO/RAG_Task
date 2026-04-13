from __future__ import annotations

import asyncio
import unittest

from app.ocp_chat import RuleFirstOcpPlanner


class _OcpClientStub:
    default_namespace = "demo"


class _LlmStub:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def generate(self, messages, max_tokens=None) -> str:  # noqa: ANN001
        self.calls += 1
        return self.response


class OcpQueryPlannerTests(unittest.TestCase):
    def test_rule_fastpath_handles_simple_count_question(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(planner.build_plan("demo namespace pod 몇개야?", {}))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "summary")
        self.assertEqual(plan.resources, ["pods"])
        self.assertEqual(plan.namespace, "demo")
        self.assertEqual(plan.parse_strategy, "rule_fastpath")

    def test_agent_fallback_handles_multi_resource_summary(self) -> None:
        llm = _LlmStub(
            '{"mode":"summary","resources":["events","deployments"],"namespace":"demo","pattern":"pandas","warning_only":true,"status_check":true,"confidence":0.93}'
        )
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub(), llm_client=llm)

        plan = asyncio.run(planner.build_plan("demo namespace에서 warning 이벤트랑 pandas deployment 상태 같이 요약해줘", {}))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.parse_strategy, "agent_fallback")
        self.assertEqual(plan.resources, ["events", "deployments"])
        self.assertTrue(plan.warning_only)
        self.assertEqual(plan.pattern, "pandas")
        self.assertEqual(llm.calls, 1)

    def test_command_style_question_does_not_take_ocp_status_lane(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(planner.build_plan("보통 pod 확인하는 명령어 뭐야?", {}))

        self.assertIsNone(plan)

    def test_mixed_command_and_live_status_question_keeps_ocp_plan(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())
        topic_state = {
            "last_namespace": "demo",
            "last_ocp_resource": "pods",
            "last_ocp_filter_keyword": "pandas",
        }

        self.assertTrue(planner.is_mixed_request("지금 내 pandas 보려면 어떤 명령어 쳐야 돼?", topic_state))
        plan = asyncio.run(planner.build_plan("지금 내 pandas 보려면 어떤 명령어 쳐야 돼?", topic_state))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "summary")
        self.assertEqual(plan.resources, ["pods"])
        self.assertEqual(plan.pattern, "pandas")

    def test_yaml_command_question_is_not_treated_as_mixed_live_query(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        self.assertFalse(planner.is_mixed_request("yaml 보려면 무슨 명령어 써?", {}))
        plan = asyncio.run(planner.build_plan("yaml 보려면 무슨 명령어 써?", {}))
        self.assertIsNone(plan)

    def test_document_yaml_followup_without_ocp_context_does_not_take_ocp_lane(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(planner.build_plan("그거 yaml로 보려면?", {}))

        self.assertIsNone(plan)

    def test_compare_with_official_doc_and_live_context_becomes_mixed(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())
        topic_state = {
            "last_ocp_resource": "pods",
            "last_ocp_result_items": [{"resource": "pods", "name": "p1"}],
        }

        self.assertTrue(planner.is_mixed_request("그 pod yaml이랑 공식 문서의 pod yaml은 뭐가 달라?", topic_state))

    def test_status_command_with_no_explicit_resource_defaults_to_pods(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(planner.build_plan("현재 상태 확인 명령어랑 실제 결과 같이 알려줘", {}))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.resources, ["pods"])

    def test_status_followup_can_inherit_last_document_resource(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(
            planner.build_plan(
                "그럼 지금 내 ocp 쪽 namespace에서는 어떻게 확인해",
                {"last_explicit_resources": ["pod"]},
            )
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.resources, ["pods"])

    def test_system_pod_count_question_does_not_use_system_as_filter_pattern(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(planner.build_plan("시스템에 파드 몇개 떠있어", {}))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.resources, ["pods"])
        self.assertEqual(plan.pattern, "")

    def test_non_followup_namespace_question_does_not_inherit_last_ocp_resource(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())

        plan = asyncio.run(planner.build_plan("namespace에 뭐 있어", {"last_ocp_resource": "pods"}))

        self.assertIsNone(plan)

    def test_yaml_followup_accepts_explicit_pod_name_from_user_message(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())
        topic_state = {
            "last_namespace": "demo",
            "last_ocp_resource": "pods",
            "last_ocp_resource_names": ["pandas-api-0", "pandas-worker-0"],
            "last_ocp_result_items": [
                {"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"},
                {"resource": "pods", "name": "pandas-worker-0", "namespace": "demo", "kind": "Pod"},
            ],
        }

        plan = asyncio.run(planner.build_plan("pandas-worker-0 이거 yaml 알려줘", topic_state))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "yaml")
        self.assertEqual(plan.target_name, "pandas-worker-0")

    def test_yaml_followup_accepts_explicit_kubernetes_name_even_if_not_in_candidates(self) -> None:
        planner = RuleFirstOcpPlanner(ocp_api_client=_OcpClientStub())
        topic_state = {
            "last_namespace": "demo",
            "last_ocp_resource": "pods",
            "last_ocp_resource_names": ["pandas-api-0", "pandas-worker-0"],
            "last_ocp_result_items": [
                {"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"},
                {"resource": "pods", "name": "pandas-worker-0", "namespace": "demo", "kind": "Pod"},
            ],
        }

        plan = asyncio.run(planner.build_plan("build-and-push-crxvmo-build-image-pod 이거 yaml 알려줘", topic_state))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.mode, "yaml")
        self.assertEqual(plan.target_name, "build-and-push-crxvmo-build-image-pod")


if __name__ == "__main__":
    unittest.main()
