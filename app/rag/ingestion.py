from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings
from app.rag.artifacts import extracted_markdown_path
from app.rag.types import Document
from app.rag.utils import normalize_text, stable_hash

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


# YAML 코드 블록 시작을 판별하는 키워드 패턴
_YAML_START_RE = re.compile(
    r"^(apiVersion|kind|metadata|spec|rules|subjects|roleRef|data|stringData"
    r"|items|parameters|provisioner|template|containers|volumes|ports"
    r"|env|resources|status|selector|replicas|strategy|storage"
    r"|certificate|description)\s*:",
)
# 번호 매기기 헤딩 패턴: "1. 제목", "1.1 제목" 등 (50자 이하)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+\.(?:\d+\.?)*)\s+(.+)$")


class DocumentIngestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ingest_paths(self, paths: list[Path], progress_callback=None) -> tuple[list[Document], list[Path]]:
        documents: list[Document] = []
        skipped: list[Path] = []

        for path in paths:
            if not path.exists() or not path.is_file():
                skipped.append(path)
                continue

            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pdf_documents, markdown_sections = self._ingest_pdf(path, progress_callback=progress_callback)
                documents.extend(pdf_documents)
                self._export_pdf_markdown(path, pdf_documents, markdown_sections)
            elif suffix in {".txt", ".md"}:
                text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
                if not text:
                    skipped.append(path)
                    continue
                documents.append(
                    Document(
                        doc_id=stable_hash(str(path)),
                        source_path=str(path),
                        page_number=None,
                        text=text,
                        metadata={"file_name": path.name, "loader": "text"},
                    )
                )
            else:
                skipped.append(path)

        return documents, skipped

    def _ingest_pdf(self, path: Path, progress_callback=None) -> tuple[list[Document], list[dict[str, str | int]]]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required to parse PDF files.")

        documents: list[Document] = []
        markdown_sections: list[dict[str, str | int]] = []

        with fitz.open(path) as pdf:
            is_slide = self._is_slide_pdf(pdf)
            footer_pattern = self._detect_footer_pattern(pdf) if not is_slide else None

            total_pages = len(pdf)
            for index, page in enumerate(pdf, start=1):
                if is_slide:
                    structured_md = self._extract_slide_page(page)
                    loader = "pdf_slide"
                else:
                    structured_md = self._extract_structured_page(page, footer_pattern)
                    loader = "pdf_text"

                text = normalize_text(structured_md)
                if not text:
                    continue

                documents.append(
                    Document(
                        doc_id=stable_hash(f"{path}:{index}"),
                        source_path=str(path),
                        page_number=index,
                        text=text,
                        metadata={"file_name": path.name, "loader": loader},
                    )
                )
                markdown_sections.append(
                    {
                        "page_number": index,
                        "loader": loader,
                        "chars": len(text),
                        "text": structured_md,
                    }
                )
                if progress_callback:
                    progress_callback("extract", index, total_pages)

        self._merge_cross_page_tables(documents, markdown_sections)
        return documents, markdown_sections

    # ------------------------------------------------------------------
    # 0) 슬라이드형 PDF 감지 및 좌표 기반 추출
    # ------------------------------------------------------------------

    def _is_slide_pdf(self, pdf) -> bool:
        """가로형 레이아웃 + 이미지 블록 비율이 높으면 슬라이드형 PDF로 판정한다."""
        if len(pdf) == 0:
            return False

        slide_pages = 0
        sample_count = min(len(pdf), 5)
        for i in range(sample_count):
            page = pdf[i]
            # 가로형(landscape) 체크
            if page.rect.width <= page.rect.height:
                continue
            # 이미지 블록 vs 텍스트 블록 비율
            blocks = page.get_text("dict")["blocks"]
            img_blocks = sum(1 for b in blocks if b.get("type") == 1)
            text_blocks = sum(1 for b in blocks if "lines" in b)
            if img_blocks > text_blocks * 2:
                slide_pages += 1

        return slide_pages >= max(2, sample_count // 2)

    def _extract_slide_page(self, page) -> str:
        """슬라이드형 페이지에서 모든 텍스트 span의 좌표를 이용해 텍스트를 재조립한다."""
        # (y, x, bbox_x_end, font_size, text)
        spans: list[tuple[float, float, float, float, str]] = []

        # 슬라이드 경계 바깥으로 넘어간 텍스트도 포함하기 위해 clip 영역 확장
        page_rect = page.rect
        wider_clip = fitz.Rect(
            page_rect.x0, page_rect.y0,
            page_rect.width * 1.5, page_rect.height * 1.2,
        )

        for block in page.get_text("dict", clip=wider_clip)["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for s in line["spans"]:
                    text = s["text"].strip()
                    if text:
                        spans.append((
                            s["origin"][1],   # y
                            s["origin"][0],   # x start
                            s["bbox"][2],     # x end (실제 텍스트 끝 좌표)
                            s["size"],        # font size
                            text,
                        ))

        if not spans:
            return ""

        # y좌표로 라인 그룹핑 (±tolerance 픽셀 이내는 같은 줄)
        spans.sort(key=lambda x: (x[0], x[1]))
        tolerance = 6.0
        lines: list[list[tuple[float, float, float, float, str]]] = []
        current_line: list[tuple[float, float, float, float, str]] = [spans[0]]
        current_y = spans[0][0]

        for span in spans[1:]:
            if abs(span[0] - current_y) <= tolerance:
                current_line.append(span)
            else:
                lines.append(current_line)
                current_line = [span]
                current_y = span[0]
        lines.append(current_line)

        # 각 라인 내에서 x좌표 순으로 정렬하고 텍스트 연결
        result: list[str] = []
        for line_spans in lines:
            line_spans.sort(key=lambda x: x[1])
            parts: list[str] = []
            prev_x_end = 0.0
            prev_text = ""
            for y, x, x_end, size, text in line_spans:
                if parts:
                    gap = x - prev_x_end
                    # 간격이 있거나, span 경계가 단어 경계인 경우 공백 삽입
                    # (쉼표/마침표 등 뒤에는 항상 공백, 그 외 span 경계에도 공백)
                    needs_space = (
                        gap > size * 0.1
                        or not self._should_merge_spans(prev_text, text)
                    )
                    if needs_space:
                        parts.append(" ")
                parts.append(text)
                prev_x_end = x_end
                prev_text = text

            line_text = "".join(parts).strip()
            if line_text:
                result.append(line_text)

        return "\n".join(result)

    @staticmethod
    def _should_merge_spans(prev: str, curr: str) -> bool:
        """두 span을 공백 없이 이어붙여야 하는지 판단한다."""
        if not prev or not curr:
            return True
        last_char = prev[-1]
        first_char = curr[0]
        # 쉼표, 마침표, 콜론 뒤는 붙이기 (이미 구두점 자체가 별도 span)
        if last_char in ".,;:!?":
            return True
        # 괄호: 여는 괄호 뒤, 닫는 괄호 앞은 붙이기
        if last_char in "([{" or first_char in ")]}":
            return True
        # 하이픈 연결은 붙이기
        if last_char == "-" or first_char == "-":
            return True
        return False

    # ------------------------------------------------------------------
    # 1) 페이지 푸터/헤더 패턴 감지
    # ------------------------------------------------------------------

    def _detect_footer_pattern(self, pdf) -> str | None:
        """전체 페이지의 마지막 2줄을 분석하여 반복되는 푸터 텍스트를 감지한다."""
        if len(pdf) < 2:
            return None

        candidates: dict[str, int] = {}
        for page in pdf:
            lines = page.get_text("text").strip().splitlines()
            if len(lines) < 3:
                continue
            # 마지막 줄이 숫자(페이지 번호)이고 그 앞 줄이 짧은 텍스트이면 푸터 후보
            last = lines[-1].strip()
            second_last = lines[-2].strip()
            if last.isdigit() and 1 <= len(second_last) <= 60:
                candidates[second_last] = candidates.get(second_last, 0) + 1

        if not candidates:
            return None

        most_common = max(candidates, key=candidates.get)
        if candidates[most_common] >= max(2, len(pdf) // 2):
            return most_common
        return None

    # ------------------------------------------------------------------
    # 2) 구조화된 페이지 추출 (테이블 + 텍스트 + 코드 블록)
    # ------------------------------------------------------------------

    def _extract_structured_page(self, page, footer_pattern: str | None) -> str:
        """페이지에서 테이블, 본문 텍스트, 코드 블록을 구조화하여 마크다운으로 변환한다."""
        # 테이블 영역 감지
        table_rects: list[tuple] = []
        table_markdowns: list[tuple[float, str]] = []
        try:
            tables = page.find_tables()
            for table in tables.tables:
                table_rects.append(table.bbox)
                md = self._table_to_markdown(table)
                if md:
                    table_markdowns.append((table.bbox[1], md))  # y좌표로 정렬용
        except Exception:
            pass

        # 텍스트 블록 추출 (dict 모드로 좌표 정보 포함)
        blocks = page.get_text("dict")["blocks"]
        text_segments: list[tuple[float, str]] = []

        for block in blocks:
            if "lines" not in block:
                continue

            # 테이블 영역과 겹치는 블록은 건너뛰기
            bx0, by0, bx1, by1 = block["bbox"]
            if self._overlaps_any_table(bx0, by0, bx1, by1, table_rects):
                continue

            block_lines = self._extract_block_lines(block)
            if block_lines:
                text_segments.append((by0, "\n".join(block_lines)))

        # 테이블과 텍스트를 y좌표 순서로 병합
        all_segments: list[tuple[float, str]] = text_segments + table_markdowns
        all_segments.sort(key=lambda x: x[0])

        raw_text = "\n\n".join(seg[1] for seg in all_segments)

        # 후처리: 푸터 제거, 헤딩 복원, 코드 블록 감지
        processed = self._remove_footer(raw_text, footer_pattern)
        processed = self._restore_headings(processed)
        processed = self._detect_code_blocks(processed)

        return processed.strip()

    def _overlaps_any_table(
        self, x0: float, y0: float, x1: float, y1: float, table_rects: list[tuple],
    ) -> bool:
        """블록 영역이 테이블 영역과 겹치는지 검사한다."""
        for tx0, ty0, tx1, ty1 in table_rects:
            # 수직 겹침이 블록 높이의 50% 이상이면 테이블 내부로 판정
            overlap_top = max(y0, ty0)
            overlap_bottom = min(y1, ty1)
            overlap_height = max(0, overlap_bottom - overlap_top)
            block_height = max(1, y1 - y0)
            if overlap_height / block_height > 0.5:
                return True
        return False

    def _extract_block_lines(self, block: dict) -> list[str]:
        """텍스트 블록에서 폰트 크기와 x-offset을 활용하여 구조화된 라인을 추출한다."""
        lines: list[str] = []
        page_left = 72.0  # 일반적인 PDF 좌측 마진

        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue

            line_text = "".join(s["text"] for s in spans).rstrip()
            if not line_text.strip():
                continue

            # x-offset 기반 들여쓰기 레벨 계산 (리스트 보존)
            x_offset = spans[0]["origin"][0]
            indent_level = max(0, int((x_offset - page_left) / 18))

            # 리스트 항목이 아닌 경우에만 들여쓰기 적용
            stripped = line_text.strip()
            if indent_level > 0 and not re.match(r"^(\d+\.\s|[-*]\s|#)", stripped):
                line_text = "  " * indent_level + stripped
            else:
                line_text = stripped

            lines.append(line_text)

        return lines

    # ------------------------------------------------------------------
    # 3) 테이블 → 마크다운 파이프 테이블
    # ------------------------------------------------------------------

    def _table_to_markdown(self, table) -> str:
        """PyMuPDF 테이블 객체를 마크다운 파이프 테이블로 변환한다."""
        rows = table.extract()
        if not rows:
            return ""

        # 셀 내용 정리 (None → 빈 문자열, 줄바꿈 → 공백)
        cleaned: list[list[str]] = []
        for row in rows:
            cleaned.append([
                (cell or "").replace("\n", " ").strip() for cell in row
            ])

        if not cleaned:
            return ""

        col_count = max(len(row) for row in cleaned)
        # 모든 행의 열 수를 맞춤
        for row in cleaned:
            while len(row) < col_count:
                row.append("")

        header = cleaned[0]
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in cleaned[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    # ------------------------------------------------------------------
    # 4) 푸터 제거
    # ------------------------------------------------------------------

    def _remove_footer(self, text: str, footer_pattern: str | None) -> str:
        """페이지 하단의 반복 푸터(문서 제목 + 페이지 번호)를 제거한다."""
        lines = text.splitlines()
        if len(lines) < 2:
            return text

        # 마지막 줄이 순수 숫자(페이지 번호)이면 제거
        while lines and lines[-1].strip().isdigit():
            lines.pop()

        # 감지된 푸터 패턴과 일치하는 마지막 줄 제거
        if footer_pattern and lines:
            if lines[-1].strip() == footer_pattern:
                lines.pop()

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 5) 헤딩 구조 복원
    # ------------------------------------------------------------------

    def _restore_headings(self, text: str) -> str:
        """번호 패턴("1. 제목", "1.1 제목")을 마크다운 헤딩으로 변환한다."""
        result_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                result_lines.append("")
                continue

            match = _NUMBERED_HEADING_RE.match(stripped)
            if match:
                number_part = match.group(1)
                title_part = match.group(2).strip()
                # 제목이 50자 이하이고 콜론/마침표로 끝나지 않으면 (본문이 아닌) 헤딩
                if len(title_part) <= 50 and not title_part.endswith((".", "다.", "함.")):
                    depth = number_part.count(".")
                    level = min(depth + 2, 4)  # ## (depth=1), ### (depth=2), #### (depth=3)
                    result_lines.append(f"{'#' * level} {number_part} {title_part}")
                    continue

            result_lines.append(line)
        return "\n".join(result_lines)

    # ------------------------------------------------------------------
    # 6) 코드 블록 감지 (YAML 등)
    # ------------------------------------------------------------------

    def _detect_code_blocks(self, text: str) -> str:
        """YAML/설정 파일 패턴을 감지하여 ```yaml 펜스로 래핑한다."""
        lines = text.splitlines()
        result: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # 이미 펜스된 코드 블록은 건너뛰기
            if line.strip().startswith("```"):
                result.append(line)
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    result.append(lines[i])
                    i += 1
                if i < len(lines):
                    result.append(lines[i])
                    i += 1
                continue

            # YAML 시작 패턴 감지
            if _YAML_START_RE.match(line.strip()):
                code_lines = [line]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.strip()
                    # 빈 줄, 들여쓰기된 줄, YAML 키-값, 리스트 항목(- ), 주석(#)
                    is_yaml = (
                        not next_stripped
                        or next_line.startswith(" ")
                        or next_line.startswith("\t")
                        or _YAML_START_RE.match(next_stripped)
                        or next_stripped.startswith("- ")
                        or next_stripped.startswith("# ")
                        or re.match(r"^[\w.-]+\s*:", next_stripped)
                    )
                    if not is_yaml:
                        break
                    code_lines.append(next_line)
                    j += 1

                # 빈 줄로 끝나는 경우 제거
                while code_lines and not code_lines[-1].strip():
                    code_lines.pop()

                if len(code_lines) >= 2:
                    result.append("```yaml")
                    result.extend(code_lines)
                    result.append("```")
                    i = j
                    continue

            result.append(line)
            i += 1

        return "\n".join(result)

    # ------------------------------------------------------------------
    # 7) 페이지 경계 테이블 병합
    # ------------------------------------------------------------------

    def _merge_cross_page_tables(
        self,
        documents: list[Document],
        markdown_sections: list[dict],
    ) -> None:
        """인접 페이지에 걸친 불완전한 테이블을 병합한다.

        Page N이 테이블로 끝나고 Page N+1이 같은 컬럼 수의 테이블로 시작하면,
        N+1의 테이블 행을 N의 테이블에 합치고 N+1에서는 제거한다.
        """
        if len(markdown_sections) < 2:
            return

        for i in range(len(markdown_sections) - 1):
            curr_text = str(markdown_sections[i]["text"])
            next_text = str(markdown_sections[i + 1]["text"])

            # 현재 페이지의 마지막 테이블 찾기
            curr_table_end = self._find_trailing_table(curr_text)
            if curr_table_end is None:
                continue

            # 다음 페이지의 첫 테이블 찾기
            next_table_start = self._find_leading_table(next_text)
            if next_table_start is None:
                continue

            curr_table_lines = curr_table_end["lines"]
            next_table_lines = next_table_start["lines"]

            # 컬럼 수 비교
            curr_col_count = curr_table_lines[0].count("|") - 1
            next_col_count = next_table_lines[0].count("|") - 1
            if curr_col_count != next_col_count:
                continue

            # N+1 테이블에서 본문 행만 추출 (헤더, 구분선 제거)
            body_rows = self._extract_table_body_rows(next_table_lines)
            if not body_rows:
                continue

            # 현재 페이지 테이블에 행 추가
            merged_curr_text = (
                curr_text[:curr_table_end["end_pos"]].rstrip()
                + "\n" + "\n".join(body_rows)
                + curr_text[curr_table_end["end_pos"]:]
            )

            # 다음 페이지에서 선행 테이블 제거
            merged_next_text = next_text[next_table_start["end_pos"]:].lstrip("\n")

            # 업데이트
            markdown_sections[i]["text"] = merged_curr_text.strip()
            markdown_sections[i]["chars"] = len(merged_curr_text.strip())
            markdown_sections[i + 1]["text"] = merged_next_text.strip()
            markdown_sections[i + 1]["chars"] = len(merged_next_text.strip())

            # Document 객체도 동기화
            for doc in documents:
                if doc.page_number == markdown_sections[i]["page_number"]:
                    doc.text = normalize_text(merged_curr_text)
                elif doc.page_number == markdown_sections[i + 1]["page_number"]:
                    doc.text = normalize_text(merged_next_text)

    def _find_trailing_table(self, text: str) -> dict | None:
        """텍스트 끝에 있는 마크다운 테이블을 찾아 라인과 위치를 반환한다."""
        lines = text.rstrip().splitlines()
        if not lines:
            return None

        # 끝에서부터 파이프 테이블 행을 수집
        table_lines = []
        for j in range(len(lines) - 1, -1, -1):
            stripped = lines[j].strip()
            if stripped and "|" in stripped:
                table_lines.insert(0, stripped)
            elif stripped:
                break
            # 빈 줄은 건너뜀 (테이블과 다른 내용 사이)

        if len(table_lines) < 2:
            return None

        # 구분선(| --- | --- |) 확인
        separator_re = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
        has_separator = any(separator_re.match(line) for line in table_lines)
        if not has_separator:
            return None

        # 테이블 시작 위치 계산
        start_pos = text.rfind(table_lines[0])
        end_pos = len(text.rstrip())

        return {"lines": table_lines, "start_pos": start_pos, "end_pos": end_pos}

    def _find_leading_table(self, text: str) -> dict | None:
        """텍스트 시작에 있는 마크다운 테이블을 찾아 라인과 위치를 반환한다."""
        lines = text.lstrip().splitlines()
        if not lines:
            return None

        table_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and "|" in stripped:
                table_lines.append(stripped)
            elif stripped:
                break
            # 빈 줄이면 테이블 끝

        if len(table_lines) < 1:
            return None

        # 테이블 끝 위치 계산
        end_pos = 0
        remaining = text.lstrip()
        for tl in table_lines:
            idx = remaining.find(tl, end_pos)
            if idx >= 0:
                end_pos = idx + len(tl)

        # lstrip으로 제거된 공백 보정
        leading_whitespace = len(text) - len(text.lstrip())
        end_pos += leading_whitespace

        return {"lines": table_lines, "end_pos": end_pos}

    def _extract_table_body_rows(self, table_lines: list[str]) -> list[str]:
        """테이블에서 헤더와 구분선을 제외한 본문 행만 추출한다."""
        separator_re = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
        body_rows = []
        found_separator = False
        for line in table_lines:
            if separator_re.match(line):
                found_separator = True
                continue
            if found_separator:
                body_rows.append(line)
            # separator 이전의 줄은 헤더이므로 skip

        # separator가 없으면 (불완전한 테이블 continuation), 모든 행이 body
        if not found_separator:
            # 첫 행이 빈 셀이면 skip (빈 헤더)
            for line in table_lines:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if any(c for c in cells):  # 내용이 있는 행만
                    body_rows.append(line)

        return body_rows

    # ------------------------------------------------------------------
    # 마크다운 내보내기
    # ------------------------------------------------------------------

    def _export_pdf_markdown(
        self,
        path: Path,
        documents: list[Document],
        markdown_sections: list[dict[str, str | int]],
    ) -> None:
        if not self.settings.save_extracted_markdown:
            return

        output_path = extracted_markdown_path(self.settings.rag_extract_dir, path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# {path.name}",
            "",
            f"- source_path: `{path}`",
            f"- extracted_pages: {len(documents)}",
            "",
        ]

        if not documents:
            lines.extend(
                [
                    "> No text was extracted from this PDF.",
                    "",
                ]
            )
        else:
            for section in markdown_sections:
                lines.extend(
                    [
                        f"## Page {section['page_number']}",
                        "",
                        f"- loader: `{section['loader']}`",
                        f"- chars: {section['chars']}",
                        "",
                        str(section["text"]).strip(),
                        "",
                        "---",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
