from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import httpx

from app.rag.bge_embedding_server import BGETEIEmbedder, EmbeddingPayloadTooLargeError


class BgeEmbeddingServerTests(unittest.TestCase):
    def test_encode_batch_maps_400_length_error_to_payload_too_large(self) -> None:
        embedder = BGETEIEmbedder("http://tei.example.com")
        request = httpx.Request("POST", "http://tei.example.com/embed")
        response = httpx.Response(400, request=request, text="Input validation error: token length exceeds limit")

        with patch("app.rag.bge_embedding_server.httpx.post", side_effect=httpx.HTTPStatusError("bad request", request=request, response=response)):
            with self.assertRaises(EmbeddingPayloadTooLargeError):
                embedder.encode_batch(["x" * 1000, "y" * 1000])

    def test_encode_batch_leaves_other_400_errors_untouched(self) -> None:
        embedder = BGETEIEmbedder("http://tei.example.com")
        request = httpx.Request("POST", "http://tei.example.com/embed")
        response = httpx.Response(400, request=request, text="unsupported input type")

        with patch("app.rag.bge_embedding_server.httpx.post", side_effect=httpx.HTTPStatusError("bad request", request=request, response=response)):
            with self.assertRaises(httpx.HTTPStatusError):
                embedder.encode_batch(["abc"])

    def test_encode_batch_accepts_embeddings_response(self) -> None:
        embedder = BGETEIEmbedder("http://tei.example.com")
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = '{"embeddings": [[1.0, 0.0], [0.0, 1.0]]}'
        response.json.return_value = {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}
        response.status_code = 200
        response.headers = {"Content-Type": "application/json"}

        with patch("app.rag.bge_embedding_server.httpx.post", return_value=response):
            vectors = embedder.encode_batch(["a", "b"])

        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(vectors[0][0], 1.0)
        self.assertAlmostEqual(vectors[1][1], 1.0)

    def test_encode_batch_does_not_map_empty_input_400_to_payload_too_large(self) -> None:
        embedder = BGETEIEmbedder("http://tei.example.com")
        request = httpx.Request("POST", "http://tei.example.com/embed")
        response = httpx.Response(400, request=request, text='{"error":"Input validation error: `inputs` cannot be empty","error_type":"Empty"}')

        with patch("app.rag.bge_embedding_server.httpx.post", side_effect=httpx.HTTPStatusError("bad request", request=request, response=response)):
            with self.assertRaises(httpx.HTTPStatusError):
                embedder.encode_batch([""])


if __name__ == "__main__":
    unittest.main()
