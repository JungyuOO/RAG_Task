# 청킹 품질 개선 설계 (newv1.0.1)

작성일: 2026-04-12
상태: Draft

## 1. 배경과 문제

현재 `StructuredMarkdownChunker`는 공식 OCP 문서에서 다음과 같은 저품질 청크를 대량 생성한다.

- 목차(TOC) 페이지의 섹션 제목이 각각 단일 heading 블록으로 파싱되어, 페이지 1개당 5~43 토큰짜리 청크가 여러 개 생성된다.
- `_should_force_boundary`가 heading을 만나면 무조건 경계를 만들기 때문에, "heading + 짧은 본문 1개"가 그대로 flush 되어 최소 크기 개념이 없다.
- dot-leader(`. . . .`)가 `_clean_display_text`에서만 제거되고, 블록 파싱 단계 및 `retrieval_text`에는 남아 검색 품질을 떨어뜨린다.
- `heading_path`가 metadata에만 저장되고 `retrieval_text`에는 포함되지 않아, 섹션 경로 신호(BM25/dense 공통)를 못 쓴다.
- 청크 크기 상한(`structured_chunk_size=900`)이 공식 문서엔 짧아 평균 청크가 더 작게 수렴한다.

또한 프론트 청크 뷰어(`/api/library/{file_name}/chunks`)는 `_is_low_signal_chunk_payload`로 TOC/챕터/dot-line 청크를 후처리 필터링하고, 응답에서 `metadata.display_text`를 대신 내려주어 **인덱스에 저장된 청크와 뷰어에 보이는 청크가 서로 다르다**. 인덱싱 단계에서 TOC를 원천 제거하면 이 필터는 존재 이유가 사라진다.

## 2. 목표

1. 목차 페이지를 인덱싱 단계에서 완전히 제외한다.
2. 청크 최소 크기(`min_chunk_chars=300`)를 경계 로직과 post-pass 머저로 보장한다.
3. 섹션 경로(`heading_path`)를 `retrieval_text` 선두에 주입하여 검색 신호를 강화한다.
4. 청크 크기 상한을 `900 → 1000`으로 소폭 상향한다.
5. 청크 뷰어는 인덱스에 저장된 청크를 **가공 없이** 그대로 표시한다.

## 3. 비목표

- 청킹 전체 구조를 "heading-first packing"으로 재설계하지 않는다(리팩터 범위 과다).
- 임베딩/리트리벌 가중치 조정은 본 스펙 범위 밖이다.
- PDF 추출(`ingestion_pdf`) 로직은 수정하지 않는다.

## 4. 설계

### 4.1 TOC 페이지 완전 제외

**위치**: `app/rag/chunking_markdown.py::StructuredMarkdownChunker._blocks_from_markdown`

`_apply_page_boundary_policies` 호출 **직전**에 `_drop_toc_pages(page_entries)` 훅을 추가한다.

페이지 단위 TOC 판별 기준 (페이지 내 비어있지 않은 라인 기준):

- 넘버드 헤딩 라인(`^\d+(?:\.\d+){1,4}\.?\s+`) 개수 ≥ 2, **그리고**
- dot-leader 라인(`(?:\.\s*){6,}`) ≥ 1, 또는 페이지번호 단독 라인(`^\d{1,4}$`) ≥ 2, **그리고**
- 코드펜스(```` ``` ````)가 페이지 안에 없다, **그리고**
- 마크다운 표 분리자 라인이 없다.

추가로, 페이지 내 어떤 라인이 `"table of contents"`, `"contents"`, `"목차"` 중 하나를 포함하면 위 신호 없이도 TOC로 판정한다 (heading 라인만 있는 TOC 대비).

판정된 페이지는 `page_entries`에서 **라인 전체를 빈 리스트로 치환**한다(다른 페이지 인덱스 오프셋을 보존하기 위해 삭제가 아닌 비움). 경계 병합 정책(`_apply_page_boundary_policies`)은 `_find_next_nonempty_page`로 이미 빈 페이지를 건너뛰므로 추가 수정 불필요.

### 4.2 dot-leader / 페이지번호 라인 사전 정제

**위치**: `chunking_markdown_support.py::_parse_annotated_markdown_blocks`

`_split_annotated_sections` 결과로 얻은 각 section의 lines를 아래 패턴에 걸리면 drop한 뒤 남은 라인으로 블록을 만든다:

- `re.fullmatch(r"(?:\.\s*){6,}", line.strip())`
- `re.fullmatch(r"\d{1,4}", line.strip())` — 단독 페이지번호 라인

drop 후 section이 비면 블록 생성을 skip한다. 이로써 일반 본문 페이지에 섞여 있던 잔여 TOC/페이지번호 잡음도 청크에 들어가지 않는다.

### 4.3 최소 청크 크기 + post-pass 머저

#### 4.3.1 설정

`app/config.py::Settings`에 필드 추가:

```python
structured_chunk_size: int = 1000       # 900 → 1000
structured_chunk_overlap: int = 160     # 120 → 160
structured_chunk_min_chars: int = 300   # 신설
```

`.env.example`에 동일 키 추가. `dependencies.py`에서 `StructuredMarkdownChunker` 생성 시 `min_chunk_chars=settings.structured_chunk_min_chars` 전달.

`StructuredMarkdownChunker.__init__` 시그니처를 `min_chunk_chars` 인자를 받도록 확장한다.

#### 4.3.2 경계 로직 완화

`_should_force_boundary`의 heading 분기를 다음과 같이 수정한다:

```python
if next_block.kind == "heading":
    current_chars = sum(len(b.text) for b in current_blocks)
    if current_chars < self.min_chunk_chars:
        return False   # 너무 짧으면 heading을 새 블록으로 계속 누적
    return True
```

lone-heading 보호 로직(`is_lone_heading`)은 유지.

#### 4.3.3 Post-pass 머저

`split()` 루프가 끝난 뒤 아래 단계를 추가한다:

```python
chunks = self._merge_small_chunks(chunks, doc_id, source_path)
```

`_merge_small_chunks` 동작:

1. 각 청크의 `len(chunk.text) < min_chunk_chars`를 검사한다.
2. 해당 청크의 `heading_path`를 `metadata["section_path"]` 문자열에서 복원한다.
3. 머지 대상 선택 우선순위:
   - (a) 다음 청크의 `section_path`가 현재 `section_path`로 시작하거나 같으면 다음과 머지,
   - (b) 직전 청크의 `page_start == 현재 page_start`면 직전과 머지,
   - (c) 둘 다 불가면 그대로 둔다.
4. 머지는 양쪽 청크의 "원본 블록 시퀀스"가 필요하므로, `split()` 단계에서 각 청크의 `metadata["_blocks"]`에 블록 리스트를 임시 보관하고 머저 이후 제거한다(외부로 새지 않게 후처리 직전에 pop).
5. 머지 결과를 `_build_chunk(doc_id, source_path, merged_blocks, new_order)`로 재조립하여 `chunk_id`/메타데이터 일관 유지.
6. 한 패스만 수행한다(머지 후 여전히 작은 청크가 나올 수는 있지만, 무한 루프 회피 및 단순성을 위해 단일 패스).

#### 4.3.4 Chunk ID 안정성

`chunk_id`는 `stable_hash(f"{doc_id}:{order}:{page_start}:{page_end}:{raw[:40]}")`로 재생성된다. 기존 인덱스와 충돌하므로 **재인덱싱이 필요**하며, 이는 7절 테스트 계획에 포함한다.

### 4.4 섹션 경로를 retrieval_text에 주입

**위치**: `chunking_markdown_support.py::_build_retrieval_text`

함수 시작에서 `blocks`의 마지막 `heading_path`를 가져와, 존재하면 `"section: " + " > ".join(heading_path)` 라인을 `parts` 맨 앞에 prepend한다. 이후 기존 블록별 정규화 로직은 그대로 유지.

`display_text`에는 주입하지 않는다(뷰어 중복 방지).

### 4.5 프론트 청크 뷰어: 원본 그대로

**백엔드** (`app/api/routes_library.py`):

- `_is_low_signal_chunk_payload` 함수와 그 호출처(`filtered_items = [...]`)를 **삭제**한다.
- `list_chunks`는 `all_items`를 직접 페이지네이션한다.
- 응답 `chunks[i]["text"]`는 `metadata.display_text` fallback을 제거하고 `chunk.get("text", "")`만 사용한다(인덱스에 저장된 본문 그대로).
- 나머지 필드(`chunk_id`, `page_number`, `token_count`, `html_anchor`, `block_anchor`, `block_types`, `section_title`)는 현행 유지.

**프론트** (`app/web/js/library.js`):

- `_renderChunkListModal`의 청크 본문 렌더링 로직은 수정 불필요(이미 `chunk.text`를 escape하여 그대로 출력).
- 상단 설명 카피 `"정리된 청크 텍스트와 섹션 정보를 확인하고, 필요하면 해당 페이지 PDF를 바로 엽니다."`를 `"인덱스에 저장된 원본 청크를 그대로 표시합니다. 필요하면 해당 페이지 PDF를 바로 엽니다."`로 교체한다.

## 5. 영향 범위

- `app/config.py` — 필드 3개 (변경/추가)
- `.env.example` — 키 3개
- `app/dependencies.py` — chunker 생성자 인자 1개 추가
- `app/rag/chunking_markdown.py` — `split`, `_should_force_boundary`, `_blocks_from_markdown`, `_merge_small_chunks`(신규), `_drop_toc_pages`(신규)
- `app/rag/chunking_markdown_support.py` — `_parse_annotated_markdown_blocks`, `_build_retrieval_text`
- `app/api/routes_library.py` — `_is_low_signal_chunk_payload` 삭제, `list_chunks` 단순화
- `app/web/js/library.js` — 뷰어 헤더 카피 1건 교체

## 6. 예상 리스크 및 완화

| 리스크 | 완화 |
|---|---|
| dot-leader 필터가 본문 한 줄을 손실 | `(?:\.\s*){6,}` 및 `^\d{1,4}$`로 단독 라인만 타깃 |
| min_chunk_chars 머저가 컨텍스트 오염 | 같은 `section_path` prefix 또는 동일 페이지 조건에서만 머지 |
| 기존 인덱스 `chunk_id` 불일치 | 라이브러리 재인덱싱 수행(7절) |
| TOC 판별이 챕터 시작 페이지(넘버드 소제목만 있는 페이지)를 오탐 | dot-leader 또는 페이지번호 단독 라인 신호를 **필수**로 요구하여 챕터 페이지와 구분 |

## 7. 테스트 계획

- 대상: `data/corpus/pdfs/` 내 OpenShift Container Platform 4.20 Advanced networking PDF (본 스펙 트리거가 된 문서).
- 절차: 재인덱싱 후 라이브러리 뷰어에서 청크 목록을 확인한다.
- 검증:
  1. p.5의 TOC 청크 6~7개가 **전부 사라진다**(목록에 등장하지 않음).
  2. 전체 청크 중 `len(text) < 300` 개수가 20개 미만으로 감소(고립 섹션 허용).
  3. 질의 `"MTU value selection"`에 대해 리트리벌 top-5에 `"2.1.2. MTU value selection"` 섹션 청크가 포함된다.
  4. `/api/library/{file}/chunks` 응답의 `text`가 인덱스 행의 `chunk.text`와 문자열 단위로 동일함(수동 1건 확인).
  5. 뷰어에 표시되는 청크 수 ≒ `indexed_chunks` (필터로 인한 차이 없음).

## 8. 롤아웃

1. 코드 변경 및 단위 수준 검증.
2. 로컬 재인덱싱(`scripts/build_index.py` 또는 `/api/library/reindex`).
3. 7절 검증 수행.
4. dev 브랜치에 커밋.
