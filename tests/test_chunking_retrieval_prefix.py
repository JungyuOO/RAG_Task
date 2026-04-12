from app.rag.types import Document
from app.rag.chunking_markdown import StructuredMarkdownChunker


def test_retrieval_text_prefixed_with_section_path():
    chunker = StructuredMarkdownChunker(chunk_size=1000, overlap=160, min_chunk_chars=0)
    markdown = "\n".join([
        "## Page 1",
        "",
        "# Chapter 2. Changing the MTU",
        "",
        "## 2.1.2. MTU value selection",
        "",
        "To avoid selecting an MTU value that is not acceptable by a node, verify the maximum MTU value accepted by the network interface using the ip -d link command. " * 3,
        "",
    ])
    chunks = chunker.split([Document(doc_id="test-doc", source_path="t.pdf", text="", page_number=1)], markdown_text=markdown)
    assert chunks, "청크가 비어있음"
    retrieval = chunks[-1].metadata["retrieval_text"]
    assert retrieval.lower().startswith("section:")
    assert "MTU value selection" in retrieval
