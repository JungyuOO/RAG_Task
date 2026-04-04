from __future__ import annotations

import unittest

from app.rag.pipeline import RagPipeline
from app.rag.pipeline import PipelineOrchestrator
from app.rag.pipeline_streaming import ChatTurnDeps, ChatTurnOrchestrator


class Phase0PipelineTests(unittest.TestCase):
    def test_pipeline_streaming_imports_resolve(self) -> None:
        self.assertIsNotNone(ChatTurnDeps)
        self.assertIsNotNone(ChatTurnOrchestrator)
        self.assertIsNotNone(PipelineOrchestrator)

    def test_pipeline_non_korean_guard_still_works(self) -> None:
        notice = RagPipeline._detect_non_korean_query("你好 pod 是什么")
        self.assertIsNotNone(notice)

    def test_pipeline_allows_korean_query(self) -> None:
        notice = RagPipeline._detect_non_korean_query("PVC가 뭐야?")
        self.assertIsNone(notice)


if __name__ == "__main__":
    unittest.main()
