import unittest


def _make_item(chunk_id: str, text: str, vector: list[float]) -> dict:
    from app.rag.utils import tokenize
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "text": text,
            "tokens": tokenize(text),
            "source_path": f"/docs/{chunk_id}.pdf",
            "page_number": 1,
            "metadata": {"page_start": 1, "page_end": 1},
        },
        "vector": vector,
    }


class TestRRFRetrieval(unittest.TestCase):
    def _make_retriever(self, top_k=3):
        from app.rag.retrieval import HybridRetriever
        return HybridRetriever(
            top_k=top_k,
            candidate_pool_size=10,
            dense_weight=0.45,
            sparse_weight=0.25,
            title_weight=0.15,
            bm25_k1=1.2,
            bm25_b=0.75,
            rerank_base_weight=0.68,
            rerank_overlap_weight=0.17,
            rerank_title_weight=0.08,
            rerank_title_bonus_weight=0.07,
            rerank_compact_bonus_weight=0.12,
            title_match_bonus=0.35,
        )

    def test_search_rrf_returns_top_k(self):
        retriever = self._make_retriever(top_k=2)
        items = [
            _make_item("a", "kubernetes pod deployment", [1.0, 0.0, 0.0]),
            _make_item("b", "docker container image", [0.0, 1.0, 0.0]),
            _make_item("c", "helm chart release", [0.0, 0.0, 1.0]),
            _make_item("d", "kubernetes service cluster", [0.9, 0.1, 0.0]),
        ]
        results = retriever.search_rrf("kubernetes", [1.0, 0.0, 0.0], items)
        self.assertEqual(len(results), 2)

    def test_search_rrf_scores_are_positive(self):
        retriever = self._make_retriever(top_k=3)
        items = [
            _make_item("a", "kubernetes pod", [1.0, 0.0]),
            _make_item("b", "docker container", [0.0, 1.0]),
            _make_item("c", "helm chart", [0.5, 0.5]),
        ]
        results = retriever.search_rrf("kubernetes pod", [1.0, 0.0], items)
        for r in results:
            self.assertGreater(r["rerank_score"], 0)

    def test_search_rrf_highest_relevant_first(self):
        retriever = self._make_retriever(top_k=3)
        items = [
            _make_item("match", "kubernetes pod deployment scaling", [1.0, 0.0]),
            _make_item("nomatch", "weather forecast temperature", [0.0, 1.0]),
        ]
        results = retriever.search_rrf("kubernetes pod", [1.0, 0.0], items)
        self.assertEqual(results[0]["chunk"]["chunk_id"], "match")

    def test_search_rrf_empty_index_returns_empty(self):
        retriever = self._make_retriever()
        results = retriever.search_rrf("query", [0.1, 0.2], [])
        self.assertEqual(results, [])

    def test_search_rrf_result_has_rerank_score(self):
        retriever = self._make_retriever(top_k=2)
        items = [
            _make_item("a", "kubernetes pod", [1.0, 0.0]),
            _make_item("b", "storage volume", [0.0, 1.0]),
        ]
        results = retriever.search_rrf("kubernetes", [1.0, 0.0], items)
        for r in results:
            self.assertIn("rerank_score", r)
            self.assertIn("chunk", r)


if __name__ == "__main__":
    unittest.main()
