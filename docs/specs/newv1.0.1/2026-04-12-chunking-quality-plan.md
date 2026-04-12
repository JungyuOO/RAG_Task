# 청킹 품질 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TOC 페이지를 인덱싱에서 원천 제외하고, 최소 청크 크기(300자)를 보장하며, 섹션 경로를 검색 텍스트에 주입하고, 프론트 청크 뷰어가 인덱스의 원본 청크를 그대로 표시하도록 청킹 파이프라인을 개선한다.

**Architecture:** `StructuredMarkdownChunker`의 기존 스트리밍 경계 로직은 유지하되 (1) TOC 페이지 사전 drop 훅과 dot-leader 라인 정제를 블록 파싱 전에 추가하고, (2) heading 경계 조건에 최소 크기 가드를 달고, (3) `split()` 결과에 post-pass 머저를 붙인다. 동시에 `routes_library.py`의 뷰어 후처리 필터(`_is_low_signal_chunk_payload`)를 제거하여 인덱스와 뷰어가 1:1로 일치하게 만든다.

**Tech Stack:** Python 3.11, FastAPI, PostgreSQL, pytest, Vanilla JS.

**Spec:** `docs/specs/newv1.0.1/2026-04-12-chunking-quality-design.md`

---

## File Structure

**Modify:**
- `app/config.py` — 3개 필드 (chunk_size, overlap, min_chunk_chars)
- `.env.example` — 동일 키 3개
- `app/rag/pipeline.py:388-391` — `StructuredMarkdownChunker` 생성자에 `min_chunk_chars` 전달
- `app/rag/chunking_markdown.py` — `__init__`, `split`, `_should_force_boundary`, `_blocks_from_markdown` / 신규: `_drop_toc_pages`, `_merge_small_chunks`, `_is_toc_page`
- `app/rag/chunking_markdown_support.py` — `_parse_annotated_markdown_blocks` (dot-leader prune), `_build_retrieval_text` (섹션 경로 prefix)
- `app/api/routes_library.py` — `_is_low_signal_chunk_payload` 삭제, `list_chunks` 단순화
- `app/web/js/library.js:265` — 뷰어 헤더 카피 교체

**Create (tests):**
- `tests/test_chunking_toc_drop.py` — TOC 페이지 drop 및 dot-leader 정제 단위 테스트
- `tests/test_chunking_min_merge.py` — 최소 청크 머저 동작 단위 테스트
- `tests/test_chunking_retrieval_prefix.py` — section path 주입 단위 테스트

---

## Task 1: 설정 필드 추가

**Files:**
- Modify: `app/config.py:64-65`
- Modify: `.env.example`
- Modify: `app/rag/pipeline.py:388-391`

- [ ] **Step 1: `Settings`에 필드 추가/수정**

`app/config.py:64-65`의 기존 2개 라인을 아래 3개 라인으로 교체한다.

```python
    structured_chunk_size: int = 1000
    structured_chunk_overlap: int = 160
    structured_chunk_min_chars: int = 300
```

- [ ] **Step 2: `.env.example`에 동일 키 추가**

파일 하단에 섹션을 추가한다 (키가 없다면 새로 만든다).

```bash
# Chunking
RAG_STRUCTURED_CHUNK_SIZE=1000
RAG_STRUCTURED_CHUNK_OVERLAP=160
RAG_STRUCTURED_CHUNK_MIN_CHARS=300
```

설정 키 접두어가 다르면 기존 `.env.example`의 케이스에 맞춰 이름을 조정한다(예: `STRUCTURED_CHUNK_SIZE`).

- [ ] **Step 3: `RagPipeline.__init__`에서 `min_chunk_chars` 전달**

`app/rag/pipeline.py:388-391`을 아래로 교체:

```python
        self.structured_chunker = StructuredMarkdownChunker(
            chunk_size=settings.structured_chunk_size,
            overlap=settings.structured_chunk_overlap,
            min_chunk_chars=settings.structured_chunk_min_chars,
        )
```

- [ ] **Step 4: 파이썬 import 검사**

Run: `python -c "from app.config import Settings; from app.rag.pipeline import RagPipeline; print('ok')"`
Expected: `ok` (import 실패 없어야 함. `StructuredMarkdownChunker`는 아직 새 인자를 안 받으므로 Task 3 Step 1 완료 후 재실행)

Step 4는 Task 3 이후로 미뤄도 된다. 지금은 설정 필드만 커밋.

- [ ] **Step 5: 커밋**

```bash
git add app/config.py .env.example app/rag/pipeline.py
git commit -m "feat(config): add structured_chunk_min_chars, raise size to 1000"
```

---

## Task 2: TOC 페이지 판별 + drop 훅 (단위 테스트 선)

**Files:**
- Modify: `app/rag/chunking_markdown.py`
- Create: `tests/test_chunking_toc_drop.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_chunking_toc_drop.py` 신규:

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_chunking_toc_drop.py -v`
Expected: FAIL — `AttributeError: 'StructuredMarkdownChunker' object has no attribute '_drop_toc_pages'` 또는 생성자 인자 누락.

- [ ] **Step 3: `StructuredMarkdownChunker.__init__` 확장**

`app/rag/chunking_markdown.py:15-18`:

```python
    def __init__(self, *, chunk_size: int, overlap: int, min_chunk_chars: int = 0, max_block_chars: int = 2000) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_chars = min_chunk_chars
        self.max_block_chars = max_block_chars
```

- [ ] **Step 4: `_drop_toc_pages` 및 `_is_toc_page` 추가**

`chunking_markdown.py`의 클래스 본문에 다음 메서드를 추가한다 (`_apply_page_boundary_policies` 근처).

```python
    _TOC_CONTENTS_MARKERS = ("table of contents", "contents", "목차")
    _NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+){1,4}\.?\s+")
    _DOT_LEADER_RE = re.compile(r"^(?:\.\s*){6,}$")
    _PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

    def _drop_toc_pages(self, page_entries: list[list[tuple[str, int]]]) -> None:
        for idx, lines in enumerate(page_entries):
            if self._is_toc_page(lines):
                page_entries[idx] = []

    def _is_toc_page(self, lines: list[tuple[str, int]]) -> bool:
        meaningful = [raw.strip() for raw, _page in lines if raw and raw.strip()]
        if not meaningful:
            return False

        lowered = [line.casefold() for line in meaningful]
        if any(marker in line for line in lowered for marker in self._TOC_CONTENTS_MARKERS):
            return True

        if any(line.startswith("```") for line in meaningful):
            return False
        if any(re.match(r"^\|?[\s:|\-]{3,}\|?$", line) for line in meaningful):
            return False

        numbered = sum(1 for line in meaningful if self._NUMBERED_HEADING_RE.match(line.lstrip("# ").strip()))
        dot_leaders = sum(1 for line in meaningful if self._DOT_LEADER_RE.match(line))
        page_numbers = sum(1 for line in meaningful if self._PAGE_NUMBER_RE.match(line))

        if numbered < 2:
            return False
        return dot_leaders >= 1 or page_numbers >= 2
```

- [ ] **Step 5: `_blocks_from_markdown`에서 훅 호출**

`chunking_markdown.py:147` 근처, `self._apply_page_boundary_policies(page_entries)` **직전**에 한 줄 추가:

```python
        self._drop_toc_pages(page_entries)
        self._apply_page_boundary_policies(page_entries)
```

- [ ] **Step 6: 테스트 재실행 → 성공 확인**

Run: `python -m pytest tests/test_chunking_toc_drop.py -v`
Expected: 3개 PASS.

- [ ] **Step 7: 커밋**

```bash
git add app/rag/chunking_markdown.py tests/test_chunking_toc_drop.py
git commit -m "feat(chunking): drop TOC pages before block parsing"
```

---

## Task 3: Dot-leader / 페이지번호 라인 사전 정제

**Files:**
- Modify: `app/rag/chunking_markdown_support.py`
- Modify: `tests/test_chunking_toc_drop.py` (테스트 추가)

- [ ] **Step 1: 테스트 추가**

`tests/test_chunking_toc_drop.py` 하단에 아래 테스트 추가:

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_chunking_toc_drop.py::test_prune_dot_leader_and_page_number_lines_in_body -v`
Expected: FAIL — 정제되지 않은 dot-leader / "42" 토큰이 여전히 블록 텍스트에 남음.

- [ ] **Step 3: 정제 로직 추가**

`app/rag/chunking_markdown_support.py::_parse_annotated_markdown_blocks` 상단의 `for section in self._split_annotated_sections(annotated_lines):` 직후, `lines = [...]`를 만들기 **전**에 section 자체를 정제한다. 현재 코드:

```python
        for section in self._split_annotated_sections(annotated_lines):
            lines = [line.strip() for line, _page in section if line.strip()]
            pages = [page for line, page in section if line.strip()]
```

아래로 교체:

```python
        _DOT_LEADER_RE = re.compile(r"^(?:\.\s*){6,}$")
        _PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
        for section in self._split_annotated_sections(annotated_lines):
            cleaned_section = [
                (raw, page)
                for raw, page in section
                if raw.strip()
                and not _DOT_LEADER_RE.match(raw.strip())
                and not _PAGE_NUMBER_RE.match(raw.strip())
            ]
            if not cleaned_section:
                continue
            lines = [line.strip() for line, _page in cleaned_section]
            pages = [page for _line, page in cleaned_section]
```

`import re`가 파일 상단에 이미 있으므로 추가 import 불필요.

동일 패턴을 `_parse_markdown_blocks`에도 적용한다. 해당 함수의 `for section in self._split_markdown_sections(body_text):` 직후:

```python
        for section in self._split_markdown_sections(body_text):
            raw_lines = section.splitlines()
            _DOT_LEADER_RE = re.compile(r"^(?:\.\s*){6,}$")
            _PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
            lines = [
                line.strip()
                for line in raw_lines
                if line.strip()
                and not _DOT_LEADER_RE.match(line.strip())
                and not _PAGE_NUMBER_RE.match(line.strip())
            ]
            if not lines:
                continue
```

- [ ] **Step 4: 테스트 재실행 → 성공**

Run: `python -m pytest tests/test_chunking_toc_drop.py -v`
Expected: 모두 PASS.

- [ ] **Step 5: 커밋**

```bash
git add app/rag/chunking_markdown_support.py tests/test_chunking_toc_drop.py
git commit -m "feat(chunking): prune dot-leader and lone page-number lines"
```

---

## Task 4: 최소 청크 크기 경계 완화 + post-pass 머저

**Files:**
- Modify: `app/rag/chunking_markdown.py`
- Create: `tests/test_chunking_min_merge.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_chunking_min_merge.py` 신규:

```python
from app.rag.types import Document
from app.rag.chunking_markdown import StructuredMarkdownChunker


def _chunker() -> StructuredMarkdownChunker:
    return StructuredMarkdownChunker(chunk_size=1000, overlap=160, min_chunk_chars=300)


def _doc(path: str = "test.pdf") -> Document:
    return Document(source_path=path, text="", page_number=1)


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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_chunking_min_merge.py -v`
Expected: FAIL — heading 경계에서 flush되어 짧은 청크가 단독으로 남음.

- [ ] **Step 3: `_should_force_boundary` 완화**

`chunking_markdown.py:97-117`의 함수 본문 시작 부분에 heading 조건을 수정한다. 기존:

```python
        if next_block.kind == "heading":
            return True
```

교체:

```python
        if next_block.kind == "heading":
            current_chars = sum(len(b.text) for b in current_blocks)
            if self.min_chunk_chars and current_chars < self.min_chunk_chars:
                return False
            return True
```

- [ ] **Step 4: `split()`에 post-pass 머저 추가**

`chunking_markdown.py::split`의 마지막 `return chunks` 직전에 다음 한 줄을 추가:

```python
        chunks = self._merge_small_chunks(chunks, doc_id, documents[0].source_path)
        return chunks
```

- [ ] **Step 5: `_merge_small_chunks` 구현**

`StructuredMarkdownChunker` 클래스 본문에 아래 메서드를 추가한다. `split()`은 블록 리스트를 보존하지 않으므로, 머저는 **기존 Chunk의 `metadata["raw_text"]`와 `metadata["section_path"]`, `page_start`/`page_end`**를 사용해 판단하고, 머지 시 두 청크의 원본 블록을 복원하기 위해 `split()`가 각 Chunk의 metadata에 `_source_blocks`를 임시로 저장하도록 한다.

먼저 `_build_chunk`에서 metadata에 `_source_blocks`를 임시 보관한다. `chunking_markdown_support.py::_build_chunk`의 metadata dict에 다음 라인을 추가(사전 구축 직후):

```python
        metadata["_source_blocks"] = list(blocks)
```

그리고 `chunking_markdown.py`에 `_merge_small_chunks` 추가:

```python
    def _merge_small_chunks(self, chunks: list, doc_id: str, source_path: str) -> list:
        if not self.min_chunk_chars or len(chunks) < 2:
            self._strip_source_blocks(chunks)
            return chunks

        result: list = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            if len(current.text) >= self.min_chunk_chars:
                result.append(current)
                i += 1
                continue

            cur_section = str(current.metadata.get("section_path") or "")
            cur_page = current.metadata.get("page_start")
            cur_blocks = current.metadata.get("_source_blocks") or []

            # (a) 다음 청크와 같은 section_path prefix면 머지
            if i + 1 < len(chunks):
                nxt = chunks[i + 1]
                nxt_section = str(nxt.metadata.get("section_path") or "")
                if cur_section and nxt_section.startswith(cur_section):
                    merged_blocks = cur_blocks + (nxt.metadata.get("_source_blocks") or [])
                    merged = self._build_chunk(doc_id, source_path, merged_blocks, len(result))
                    result.append(merged)
                    i += 2
                    continue

            # (b) 직전 청크가 있고 같은 페이지면 머지
            if result and result[-1].metadata.get("page_start") == cur_page:
                prev = result.pop()
                merged_blocks = (prev.metadata.get("_source_blocks") or []) + cur_blocks
                merged = self._build_chunk(doc_id, source_path, merged_blocks, len(result))
                result.append(merged)
                i += 1
                continue

            # (c) 둘 다 불가 → 그대로
            result.append(current)
            i += 1

        self._strip_source_blocks(result)
        return result

    @staticmethod
    def _strip_source_blocks(chunks: list) -> None:
        for chunk in chunks:
            if isinstance(chunk.metadata, dict):
                chunk.metadata.pop("_source_blocks", None)
```

- [ ] **Step 6: 테스트 재실행**

Run: `python -m pytest tests/test_chunking_min_merge.py -v`
Expected: 2개 PASS. 실패 시 `section_path` 비교(정확히 "Parent > Child")가 기대대로 세팅되는지 확인 — heading level 판정이 issue라면 prefix 매칭을 `nxt_section == cur_section`으로 좁혀도 무방.

- [ ] **Step 7: Task 1 Step 4 import 검사 재실행**

Run: `python -c "from app.config import Settings; from app.rag.pipeline import RagPipeline; print('ok')"`
Expected: `ok`

- [ ] **Step 8: 커밋**

```bash
git add app/rag/chunking_markdown.py app/rag/chunking_markdown_support.py tests/test_chunking_min_merge.py
git commit -m "feat(chunking): enforce min_chunk_chars via boundary + post-pass merge"
```

---

## Task 5: 섹션 경로를 retrieval_text에 주입

**Files:**
- Modify: `app/rag/chunking_markdown_support.py`
- Create: `tests/test_chunking_retrieval_prefix.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_chunking_retrieval_prefix.py` 신규:

```python
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
    chunks = chunker.split([Document(source_path="t.pdf", text="", page_number=1)], markdown_text=markdown)
    assert chunks, "청크가 비어있음"
    retrieval = chunks[-1].metadata["retrieval_text"]
    assert retrieval.lower().startswith("section:")
    assert "MTU value selection" in retrieval
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_chunking_retrieval_prefix.py -v`
Expected: FAIL — retrieval_text에 `section:` prefix 없음.

- [ ] **Step 3: `_build_retrieval_text` 수정**

`chunking_markdown_support.py::_build_retrieval_text`의 함수 본문 시작 부분을 교체. 기존:

```python
    def _build_retrieval_text(self, blocks: list[MarkdownBlock]) -> str:
        parts: list[str] = []
        for block in blocks:
```

교체:

```python
    def _build_retrieval_text(self, blocks: list[MarkdownBlock]) -> str:
        parts: list[str] = []
        heading_path: tuple[str, ...] = ()
        for block in reversed(blocks):
            if block.heading_path:
                heading_path = block.heading_path
                break
        if heading_path:
            parts.append("section: " + " > ".join(heading_path))
        for block in blocks:
```

- [ ] **Step 4: 테스트 재실행**

Run: `python -m pytest tests/test_chunking_retrieval_prefix.py -v`
Expected: PASS.

- [ ] **Step 5: 회귀 테스트 실행**

Run: `python -m pytest tests/test_chunking_toc_drop.py tests/test_chunking_min_merge.py tests/test_chunking_retrieval_prefix.py -v`
Expected: 모두 PASS.

- [ ] **Step 6: 커밋**

```bash
git add app/rag/chunking_markdown_support.py tests/test_chunking_retrieval_prefix.py
git commit -m "feat(chunking): inject section path prefix into retrieval_text"
```

---

## Task 6: 뷰어 후처리 필터 제거 및 응답 단순화

**Files:**
- Modify: `app/api/routes_library.py:23-57` (삭제), `:330-367` (단순화)

- [ ] **Step 1: `_is_low_signal_chunk_payload` 삭제**

`app/api/routes_library.py`의 23~57행 전체(`def _is_low_signal_chunk_payload` 및 본문)를 삭제한다. 동시에 파일 상단 `import re`가 이 함수 외에 쓰이고 있지 않으면 그대로 두고, 다른 곳에서 사용하면 유지한다. 확인:

Run: `grep -n "^import re\|^from re\|\bre\\." app/api/routes_library.py`
Expected: `re` 사용처가 더 없다면 `import re`도 삭제 가능. 하나라도 남아있다면 import 유지.

- [ ] **Step 2: `list_chunks` 단순화**

`routes_library.py:330-367`의 엔드포인트를 아래로 교체:

```python
@router.get("/api/library/{file_name}/chunks")
async def list_chunks(
    file_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source_path: str | None = Query(None),
    container: AppContainer = Depends(get_container),
):
    lookup_source_path = source_path or str(resolve_library_pdf(container.settings, file_name))
    all_items = container.pipeline.index_repository.list_all_chunks(lookup_source_path)
    total = len(all_items)
    offset = max(page - 1, 0) * page_size
    page_items = all_items[offset : offset + page_size]
    chunks = []
    for item in page_items:
        chunk = item["chunk"]
        metadata = chunk.get("metadata", {})
        chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number") or metadata.get("page_start"),
                "token_count": len(chunk.get("tokens", [])),
                "html_anchor": metadata.get("html_anchor", ""),
                "block_anchor": metadata.get("primary_block_anchor", ""),
                "block_types": metadata.get("block_types", ""),
                "section_title": metadata.get("section_title", ""),
            }
        )
    return {
        "file_name": file_name,
        "source_path": lookup_source_path,
        "chunks": chunks,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
```

차이: (a) `_is_low_signal_chunk_payload` 필터 제거, (b) `text` 필드가 `metadata.display_text` fallback 없이 `chunk.text`만 사용.

- [ ] **Step 3: 서버 기동 스모크 테스트**

Run: `python -c "from app.api.routes_library import router; print('ok')"`
Expected: `ok`

- [ ] **Step 4: 커밋**

```bash
git add app/api/routes_library.py
git commit -m "refactor(api): chunk viewer shows index chunks as-is"
```

---

## Task 7: 프론트 청크 뷰어 헤더 카피 교체

**Files:**
- Modify: `app/web/js/library.js:265`

- [ ] **Step 1: 헤더 카피 교체**

`app/web/js/library.js:265`의 문자열:

```javascript
      '<div><h3>' + escapeHtml(data.file_name) + ' chunk list (' + data.total + ')</h3><div class="chunk-viewer-copy">정리된 청크 텍스트와 섹션 정보를 확인하고, 필요하면 해당 페이지 PDF를 바로 엽니다.</div></div>' +
```

교체:

```javascript
      '<div><h3>' + escapeHtml(data.file_name) + ' chunk list (' + data.total + ')</h3><div class="chunk-viewer-copy">인덱스에 저장된 원본 청크를 그대로 표시합니다. 필요하면 해당 페이지 PDF를 바로 엽니다.</div></div>' +
```

- [ ] **Step 2: 커밋**

```bash
git add app/web/js/library.js
git commit -m "chore(web): update chunk viewer header copy to reflect raw display"
```

---

## Task 8: 재인덱싱 및 수동 검증

**Files:** (없음 — 런타임 검증)

- [ ] **Step 1: 서버 기동**

Run: `uvicorn app.main:app --reload`
별도 터미널에서 이후 단계 수행.

- [ ] **Step 2: 라이브러리 재인덱싱 트리거**

브라우저에서 `http://localhost:8000/` → 라이브러리 화면 → 재인덱싱 버튼 클릭. 또는:

Run: `python scripts/build_index.py` (해당 스크립트가 복원되어 있다면)

Expected: 진행상태가 success로 종료.

- [ ] **Step 3: TOC 청크 사라짐 확인**

대상 PDF(OpenShift Container Platform 4.20 Advanced networking)의 청크 뷰어를 연다. 검증:

- p.5에 있었던 `6.1.1.1. Supported platforms`, `6.2. ENABLING BGP ROUTING`, `6.1. ABOUT BGP ROUTING` 같은 단일 heading 청크가 **목록에 등장하지 않는다**.
- 뷰어 상단 총 청크 수가 재인덱싱 전보다 **감소**했다.

- [ ] **Step 4: 최소 청크 크기 분포 확인**

DB에 직접 쿼리하거나, Python REPL에서:

```python
from app.dependencies import build_container
from app.config import Settings
c = build_container(Settings())
items = c.pipeline.index_repository.list_all_chunks("<target source_path>")
short = [i for i in items if len(i["chunk"].get("text", "")) < 300]
print(len(items), len(short))
```

Expected: `short`가 전체의 10% 미만(고립 섹션 허용). 재인덱싱 전과 비교해 유의미하게 감소.

- [ ] **Step 5: 검색 품질 스팟체크**

채팅 UI에서 `"MTU value selection"` 질의. Expected: 답변 근거 카드에 `2.1.2. MTU value selection` 섹션을 가진 청크가 포함됨.

- [ ] **Step 6: 뷰어 ↔ 인덱스 일치 확인**

- 뷰어의 임의 청크 1개의 `chunk_id`와 text를 복사.
- `GET /api/library/{file}/chunks/{chunk_id}`로 상세 조회.
- 응답의 `text`가 뷰어 표시와 **문자열 단위로 동일**.
- 뷰어의 `total`이 `indexed_chunks` 값과 동일 (후처리 필터 차이 없음).

- [ ] **Step 7: 전체 테스트 재실행**

Run: `python -m pytest tests/test_chunking_toc_drop.py tests/test_chunking_min_merge.py tests/test_chunking_retrieval_prefix.py -v`
Expected: 전부 PASS.

- [ ] **Step 8: 최종 커밋 (필요 시)**

검증 과정에서 발견된 사소한 수정이 있다면 커밋. 없다면 skip.

---

## 완료 기준

- [ ] Task 1~7 전부 완료
- [ ] 새 테스트 3개 파일 모두 PASS
- [ ] 대상 PDF 재인덱싱 후 TOC 유래 소형 청크가 뷰어에서 사라졌음
- [ ] 뷰어 `total` ≒ `indexed_chunks`
- [ ] `"MTU value selection"` 질의에서 해당 섹션이 검색 top-K 안에 들어옴
