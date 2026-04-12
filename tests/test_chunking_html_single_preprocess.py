from __future__ import annotations

from app.rag.chunking_markdown import StructuredMarkdownChunker
from app.rag.types import Document


def _chunker() -> StructuredMarkdownChunker:
    return StructuredMarkdownChunker(chunk_size=1000, overlap=160, min_chunk_chars=0)


def test_html_single_preprocess_collapses_short_text_fences_and_removes_copylink_noise() -> None:
    chunker = _chunker()
    markdown = "\n".join(
        [
            "## Page 4",
            "## Chapter 1. ExampleCopy linkLink copied to clipboard!",
            "",
            "```text",
            "PodNetworkConnectivity",
            "```",
            "",
            "This is the explanatory paragraph.",
            "",
        ]
    )

    chunks = chunker.split([Document(doc_id="d1", source_path="advanced_networking.md", text="", page_number=1)], markdown_text=markdown)

    assert chunks
    first = chunks[0].text
    assert "Copy linkLink copied to clipboard!" not in first
    assert "PodNetworkConnectivity" in first
    assert "```text" not in first


def test_html_single_preprocess_preserves_structured_yaml_text_fence() -> None:
    chunker = _chunker()
    markdown = "\n".join(
        [
            "## Page 10",
            "## Example section",
            "",
            "```text",
            "apiVersion: v1 kind: Pod metadata: name: sample spec: containers: - name: app image: demo",
            "```",
            "",
        ]
    )

    chunks = chunker.split([Document(doc_id="d1", source_path="advanced_networking.md", text="", page_number=1)], markdown_text=markdown)

    assert chunks
    first = chunks[0].text
    assert "apiVersion:" in first
    assert "kind: Pod" in first


def test_html_single_yaml_field_line_does_not_become_section_heading() -> None:
    chunker = _chunker()
    markdown = "\n".join(
        [
            "## Page 6",
            "## Chapter 1. Example",
            "",
            "```text",
            "sourcePlacement:",
            "```",
            "",
            "```text",
            "$ oc get pods -n demo -o wide",
            "```",
            "",
        ]
    )

    chunks = chunker.split([Document(doc_id="d1", source_path="advanced_networking.md", text="", page_number=1)], markdown_text=markdown)

    assert chunks
    assert chunks[0].metadata.get("section_title") != "sourcePlacement:"


def test_html_single_camel_case_yaml_field_line_does_not_become_section_heading() -> None:
    chunker = _chunker()
    markdown = "\n".join(
        [
            "## Page 6",
            "## Chapter 1. Example",
            "",
            "```text",
            "sourcePlacement:",
            "```",
            "",
            "```text",
            "nodeSelector: app=demo",
            "```",
            "",
        ]
    )

    chunks = chunker.split([Document(doc_id="d1", source_path="advanced_networking.md", text="", page_number=1)], markdown_text=markdown)

    assert chunks
    assert all(chunk.metadata.get("section_title") != "sourcePlacement:" for chunk in chunks)
