import math
import unittest
from unittest.mock import MagicMock, patch


class TestBGEOllamaEmbedder(unittest.TestCase):
    def _make_embedder(self):
        from app.rag.bge_embeddings import BGEOllamaEmbedder
        return BGEOllamaEmbedder(base_url="http://localhost:11434", model="bge-m3")

    def test_encode_returns_1024_dim(self):
        embedder = self._make_embedder()
        raw_vec = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [raw_vec]}
        with patch("httpx.post", return_value=mock_response):
            result = embedder.encode("테스트 쿼리")
        self.assertEqual(len(result), 1024)

    def test_encode_is_l2_normalized(self):
        embedder = self._make_embedder()
        raw_vec = [3.0, 4.0] + [0.0] * 1022  # norm = 5.0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [raw_vec]}
        with patch("httpx.post", return_value=mock_response):
            result = embedder.encode("test")
        norm = math.sqrt(sum(v * v for v in result))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_encode_batch_returns_matrix(self):
        embedder = self._make_embedder()
        texts = ["첫 번째", "두 번째", "세 번째"]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1] * 1024 for _ in texts]}
        with patch("httpx.post", return_value=mock_response):
            result = embedder.encode_batch(texts)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 1024)

    def test_encode_passage_same_output_as_encode(self):
        """passage와 query 인코딩이 동일한 API 호출 구조를 사용한다."""
        embedder = self._make_embedder()
        raw_vec = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [raw_vec]}
        with patch("httpx.post", return_value=mock_response) as mock_post:
            embedder.encode_passage("passage text")
        mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
