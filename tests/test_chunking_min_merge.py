from app.rag.types import Document
from app.rag.chunking_markdown import StructuredMarkdownChunker


def _chunker() -> StructuredMarkdownChunker:
    return StructuredMarkdownChunker(chunk_size=1000, overlap=160, min_chunk_chars=300)


def _doc(path: str = "test.pdf") -> Document:
    return Document(doc_id="test-doc", source_path=path, text="", page_number=1)


def test_heading_boundary_is_skipped_when_current_is_too_short():
    chunker = _chunker()
    markdown = "\n".join([
        "## Page 1",
        "",
        "# Section A",
        "",
        "Short body A.",
        "",
        "# Section B",
        "",
        "Body B is a reasonably long paragraph with enough characters to comfortably exceed the minimum chunk threshold many times over. " * 5,
        "",
    ])
    chunks = chunker.split([_doc()], markdown_text=markdown)
    # Section A의 짧은 본문은 Section B 본문과 한 청크로 머지되어야 한다.
    assert any("Section A" in c.text and "Section B" in c.text for c in chunks)


def test_post_pass_merges_tiny_chunk_with_neighbor_same_section():
    chunker = _chunker()
    long_body = ("Long paragraph content line. " * 30).strip()
    markdown = "\n".join([
        "## Page 1",
        "",
        "# Parent",
        "",
        "## Child",
        "",
        "tiny.",
        "",
        "## Child",
        "",
        long_body,
        "",
    ])
    chunks = chunker.split([_doc()], markdown_text=markdown)
    tiny_alone = [c for c in chunks if c.text.strip().endswith("tiny.") and len(c.text) < 50]
    assert not tiny_alone, "작은 청크가 이웃과 머지되지 않았습니다"
