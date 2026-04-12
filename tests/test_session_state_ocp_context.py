from __future__ import annotations

import unittest

from app.rag.types import ChatTurn
from app.session.state import build_topic_state


class SessionStateOcpContextTests(unittest.TestCase):
    def test_build_topic_state_tracks_last_ocp_context(self) -> None:
        turns = [
            ChatTurn(role="user", content="demo namespace의 pandas 관련 리소스 보여줘"),
            ChatTurn(
                role="assistant",
                content="demo namespace에서 pandas 관련 리소스를 찾았습니다.",
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


if __name__ == "__main__":
    unittest.main()
