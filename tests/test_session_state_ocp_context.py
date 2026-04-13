from __future__ import annotations

import unittest

from app.rag.types import ChatTurn
from app.session.state import build_topic_state


class SessionStateOcpContextTests(unittest.TestCase):
    def test_build_topic_state_tracks_last_ocp_context(self) -> None:
        turns = [
            ChatTurn(role="user", content="demo namespace?? pandas ?? ??? ???"),
            ChatTurn(
                role="assistant",
                content="demo namespace?? pandas ?? ???? ?????.",
                metadata={
                    "query_interpretation": {"resources": ["pods", "deployments"], "intent": "ocp_status"},
                    "answer_route": "ocp_status",
                    "ocp_context": {
                        "namespace": "demo",
                        "last_resource": "pods",
                        "last_resource_names": ["pandas-api-0", "pandas-worker-0"],
                        "last_result_items": [
                            {"resource": "pods", "name": "pandas-api-0", "namespace": "demo", "kind": "Pod"},
                            {"resource": "pods", "name": "pandas-worker-0", "namespace": "demo", "kind": "Pod"},
                        ],
                        "last_filter_keyword": "pandas",
                    },
                },
            ),
        ]

        topic_state = build_topic_state(turns)

        self.assertEqual(topic_state["last_namespace"], "demo")
        self.assertEqual(topic_state["last_ocp_resource"], "pods")
        self.assertEqual(topic_state["last_ocp_resource_names"], ["pandas-api-0", "pandas-worker-0"])
        self.assertEqual(topic_state["last_ocp_result_items"][0]["name"], "pandas-api-0")
        self.assertEqual(topic_state["last_ocp_filter_keyword"], "pandas")
        self.assertEqual(topic_state["active_lane"], "ocp")
        self.assertEqual(topic_state["active_slot"]["lane"], "ocp")
        self.assertEqual(topic_state["active_slot"]["ocp_resource"], "pods")
        self.assertEqual(topic_state["active_slot"]["namespace"], "demo")

    def test_build_topic_state_tracks_document_slot(self) -> None:
        turns = [
            ChatTurn(role="user", content="pod ?? ??? ???"),
            ChatTurn(
                role="assistant",
                content="oc get pods -A",
                metadata={
                    "answer_route": "extractive_code",
                    "query_interpretation": {
                        "resources": ["pod"],
                        "intent": "cli_example",
                        "response_shape": "code",
                        "document_group_preference": "official_ocp",
                    },
                    "source_grounding": [
                        {"file_name": "cli_tools.md", "source_path": "/docs/cli_tools.md"},
                    ],
                    "last_example_anchor": {"resource_kind": "pod"},
                },
            ),
        ]

        topic_state = build_topic_state(turns)

        self.assertEqual(topic_state["active_lane"], "document")
        self.assertEqual(topic_state["active_slot"]["lane"], "document")
        self.assertEqual(topic_state["active_slot"]["resources"], ["pod"])
        self.assertEqual(topic_state["active_slot"]["sources"], ["cli_tools.md"])
        self.assertEqual(topic_state["active_slot"]["code_resource_kind"], "pod")


if __name__ == "__main__":
    unittest.main()
