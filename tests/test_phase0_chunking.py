from __future__ import annotations

import unittest

from app.rag.chunking import TextChunker
from app.rag.chunking_markdown import StructuredMarkdownChunker
from app.rag.types import Document


class Phase0ChunkingTests(unittest.TestCase):
    def test_text_chunker_reexport_still_works(self) -> None:
        chunker = TextChunker(chunk_size=12, overlap=2)
        documents = [
            Document(
                doc_id="doc-1",
                source_path="sample.pdf",
                page_number=1,
                text="abcdefghijklmno",
                metadata={"file_name": "sample.pdf"},
            )
        ]

        chunks = chunker.split(documents)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["chunking_strategy"], "page_window")
        self.assertEqual(chunks[0].metadata["page_start"], 1)

    def test_structured_chunker_keeps_heading_with_following_content(self) -> None:
        chunker = StructuredMarkdownChunker(chunk_size=1200, overlap=120)
        markdown_text = """## Page 5

Volume Types:

- Thin
- Thick
- Snapshot

Volume descriptions continue here with more explanatory prose.
"""
        documents = [
            Document(
                doc_id="doc-2",
                source_path="sample.pdf",
                page_number=5,
                text="Volume Types Thin Thick Snapshot Volume descriptions continue here.",
            )
        ]

        chunks = chunker.split(documents, markdown_text=markdown_text)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["block_types"], "heading,list")
        self.assertEqual(chunks[1].metadata["block_types"], "heading,paragraph")
        self.assertEqual(chunks[0].metadata["section_title"], "Volume Types:")

    def test_structured_chunker_merges_cross_page_yaml(self) -> None:
        chunker = StructuredMarkdownChunker(chunk_size=1200, overlap=120)
        markdown_text = """## Page 3

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: demo-ingress
spec:
  rules:
  - host: demo.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-svc
            port:
              number: 8080
```

---

## Page 4

      - path: /static
        pathType: Prefix
        backend:
          service:
            name: static-svc
            port:
              number: 9000
"""
        documents = [
            Document(
                doc_id="doc-3",
                source_path="sample.pdf",
                page_number=3,
                text="Ingress YAML",
            )
        ]

        chunks = chunker.split(documents, markdown_text=markdown_text)
        code_chunks = [chunk for chunk in chunks if chunk.metadata["block_types"] == "code"]

        self.assertEqual(len(code_chunks), 1)
        self.assertEqual(code_chunks[0].metadata["page_start"], 3)
        self.assertEqual(code_chunks[0].metadata["page_end"], 4)
        self.assertIn("name: static-svc", code_chunks[0].text)


if __name__ == "__main__":
    unittest.main()
