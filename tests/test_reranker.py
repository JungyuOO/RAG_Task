import unittest
from unittest.mock import MagicMock, patch


class TestOllamaReranker(unittest.TestCase):
    def _make_reranker(self):
        from app.rag.reranker import OllamaReranker
        return OllamaReranker(
            base_url="http://localhost:11434",
            model="bge-reranker-v2-m3",
            top_k=2,
        )

    def _make_candidates(self, n: int) -> list[dict]:
        return [
            {
                "chunk": {
                    "chunk_id": f"chunk_{i}",
                    "text": f"document text number {i}",
                    "tokens": ["document", "text", str(i)],
                    "source_path": f"/docs/file_{i}.pdf",
                    "page_number": 1,
                    "metadata": {},
                },
                "vector": [0.1] * 1024,
                "score": 1.0 / (i + 1),
                "rerank_score": 1.0 / (i + 1),
            }
            for i in range(n)
        ]

    def test_rerank_returns_top_k(self):
        reranker = self._make_reranker()
        candidates = self._make_candidates(5)
        scores = [0.1, 0.9, 0.3, 0.7, 0.5]
        responses = [MagicMock() for _ in scores]
        for resp, score in zip(responses, scores):
            resp.status_code = 200
            resp.json.return_value = {"embeddings": [[score]]}
            resp.raise_for_status.return_value = None
        with patch("httpx.post", side_effect=responses):
            results = reranker.rerank("query", candidates)
        self.assertEqual(len(results), 2)

    def test_rerank_orders_by_score_descending(self):
        reranker = self._make_reranker()
        candidates = self._make_candidates(3)
        scores = [0.1, 0.9, 0.3]
        responses = [MagicMock() for _ in scores]
        for resp, score in zip(responses, scores):
            resp.status_code = 200
            resp.json.return_value = {"embeddings": [[score]]}
            resp.raise_for_status.return_value = None
        with patch("httpx.post", side_effect=responses):
            results = reranker.rerank("query", candidates)
        self.assertEqual(results[0]["chunk"]["chunk_id"], "chunk_1")  # score 0.9 highest

    def test_rerank_fewer_than_top_k_returns_all(self):
        reranker = self._make_reranker()  # top_k=2
        candidates = self._make_candidates(1)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.8]]}
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.post", return_value=mock_resp):
            results = reranker.rerank("query", candidates)
        self.assertEqual(len(results), 1)

    def test_rerank_empty_candidates_returns_empty(self):
        reranker = self._make_reranker()
        results = reranker.rerank("query", [])
        self.assertEqual(results, [])

    def test_rerank_score_overwrites_original(self):
        """rerank_score가 cross-encoder 점수로 덮어씌워진다."""
        reranker = self._make_reranker()
        candidates = self._make_candidates(1)
        original_score = candidates[0]["rerank_score"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.999]]}
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.post", return_value=mock_resp):
            results = reranker.rerank("query", candidates)
        self.assertNotAlmostEqual(results[0]["rerank_score"], original_score)
        self.assertAlmostEqual(results[0]["rerank_score"], 0.999)


if __name__ == "__main__":
    unittest.main()
