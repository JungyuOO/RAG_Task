from __future__ import annotations

import unittest

from apps.api.schemas.copilot_chat import CopilotChatHistoryTurn
from apps.api.rag.query.query_rewrite_agent import QueryRewriteAgent


class _FakeLlmClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.is_enabled = True

    async def generate(self, messages, *, max_tokens=None, temperature=None):  # noqa: ANN001
        del messages, max_tokens, temperature
        return self.response


class QueryRewriteAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_followup_rewrite_can_restrict_to_previous_sources(self) -> None:
        agent = QueryRewriteAgent(llm_client=_FakeLlmClient('{"rewritten_query":"pod security context 설명","use_previous_doc_sources":true}'))

        result = await agent.rewrite(
            message="Can you explain that in more detail?",
            recent_turns=[
                CopilotChatHistoryTurn(
                    role="assistant",
                    text="Previous answer",
                    lane="doc_hybrid",
                    source_paths=["/docs/security.md"],
                )
            ],
        )

        self.assertEqual(result.rewritten_query, "pod security context 설명")
        self.assertEqual(result.allowed_source_paths, ["/docs/security.md"])


if __name__ == "__main__":
    unittest.main()


