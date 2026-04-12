from app.rag.chunking_markdown import StructuredMarkdownChunker


def _make_chunker() -> StructuredMarkdownChunker:
    return StructuredMarkdownChunker(chunk_size=1000, overlap=160, min_chunk_chars=300)


def test_drop_toc_page_with_dot_leaders():
    chunker = _make_chunker()
    page_entries = [
        [
            ("# CHAPTER 2. CHANGING THE MTU", 5),
            ("2.1. ABOUT THE CLUSTER MTU", 5),
            ("2.1.1. Service interruption considerations", 5),
            ("2.1.2. MTU value selection", 5),
            (". . . . . . . . . . . . . . . . . . . . . . . . . . .", 5),
            ("16", 5),
        ],
        [
            ("Normal paragraph content that should survive the TOC filter.", 6),
        ],
    ]
    chunker._drop_toc_pages(page_entries)
    assert page_entries[0] == []
    assert len(page_entries[1]) == 1


def test_drop_toc_page_by_contents_marker():
    chunker = _make_chunker()
    page_entries = [
        [
            ("Table of Contents", 2),
            ("1. Intro", 2),
            ("2. Usage", 2),
        ],
    ]
    chunker._drop_toc_pages(page_entries)
    assert page_entries[0] == []


def test_keep_chapter_start_page_without_dot_leaders():
    chunker = _make_chunker()
    page_entries = [
        [
            ("# CHAPTER 2. CHANGING THE MTU FOR THE CLUSTER NETWORK", 10),
            ("This chapter describes how to change the maximum transmission unit.", 10),
            ("The MTU value determines packet size on the cluster network.", 10),
        ],
    ]
    chunker._drop_toc_pages(page_entries)
    assert len(page_entries[0]) == 3


def test_prune_dot_leader_and_page_number_lines_in_body():
    from app.rag.chunking import MarkdownBlock
    chunker = _make_chunker()
    annotated = [
        ("This is a real paragraph sentence that belongs in the chunk.", 7),
        (". . . . . . . . . . . . . . . . .", 7),
        ("42", 7),
        ("Another real sentence that should also survive the prune.", 7),
    ]
    blocks = chunker._parse_annotated_markdown_blocks(annotated)
    joined = " ".join(block.text for block in blocks)
    assert "real paragraph sentence" in joined
    assert "Another real sentence" in joined
    assert ". ." not in joined
    assert "42" not in joined.split()
