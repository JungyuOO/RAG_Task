from __future__ import annotations

import logging
import time
from pathlib import Path

from app.config import Settings

_logger = logging.getLogger("rag.startup")
from app.rag.ingestion_pdf_extract import PdfExtractionSupport, PdfMergeSupport
from app.rag.types import Document
from app.rag.utils import normalize_text, stable_hash

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


class PdfExtractor(PdfExtractionSupport, PdfMergeSupport):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract_pdf(self, path: Path, progress_callback=None) -> tuple[list[Document], list[dict[str, str | int]]]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required to parse PDF files.")

        documents: list[Document] = []
        markdown_sections: list[dict[str, str | int]] = []

        t0 = time.perf_counter()
        with fitz.open(path) as pdf:
            is_slide = self._is_slide_pdf(pdf)
            footer_pattern = self._detect_footer_pattern(pdf) if not is_slide else None
            header_patterns = self._detect_header_patterns(pdf) if not is_slide else set()
            total_pages = len(pdf)
            t_open = time.perf_counter()
            _logger.info("[Timing][PDF:%s] 파일 열기+분석: %.2fs, 총 %d 페이지 (슬라이드=%s)",
                         path.name, t_open - t0, total_pages, is_slide)

            slow_pages: list[tuple[int, float]] = []
            for index, page in enumerate(pdf, start=1):
                tp = time.perf_counter()
                if is_slide:
                    structured_markdown = self._extract_slide_page(page)
                    loader = "pdf_slide"
                else:
                    structured_markdown = self._extract_structured_page(page, footer_pattern, header_patterns)
                    loader = "pdf_text"
                page_elapsed = time.perf_counter() - tp

                if page_elapsed > 1.0:
                    slow_pages.append((index, page_elapsed))

                text = normalize_text(structured_markdown)
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
                        "text": structured_markdown,
                    }
                )
                if progress_callback:
                    progress_callback("extract", index, total_pages)

        t_pages = time.perf_counter()
        _logger.info("[Timing][PDF:%s] 페이지 추출 완료: %.2fs (페이지당 평균 %.3fs)",
                     path.name, t_pages - t_open,
                     (t_pages - t_open) / max(total_pages, 1))
        if slow_pages:
            top = sorted(slow_pages, key=lambda x: x[1], reverse=True)[:5]
            _logger.warning("[Timing][PDF:%s] 느린 페이지 top5: %s",
                            path.name, [(f"p{p}", f"{e:.2f}s") for p, e in top])

        self._merge_cross_page_tables(documents, markdown_sections)
        self._merge_cross_page_yaml_blocks(documents, markdown_sections)
        t_merge = time.perf_counter()
        _logger.info("[Timing][PDF:%s] 테이블/YAML 병합: %.2fs", path.name, t_merge - t_pages)
        return documents, markdown_sections

    def export_markdown(self, path: Path, documents: list[Document], markdown_sections: list[dict[str, str | int]]) -> None:
        self.export_artifacts(path, documents, markdown_sections)

    def export_artifacts(self, path: Path, documents: list[Document], markdown_sections: list[dict[str, str | int]]) -> None:
        self._export_pdf_markdown(path, documents, markdown_sections)
        self._export_pdf_html(path, documents, markdown_sections)
        self._export_pdf_metadata(path, documents, markdown_sections)

    def _is_slide_pdf(self, pdf) -> bool:
        if len(pdf) == 0:
            return False

        slide_pages = 0
        sample_count = min(len(pdf), 5)
        for index in range(sample_count):
            page = pdf[index]
            if page.rect.width <= page.rect.height:
                continue
            blocks = page.get_text("dict")["blocks"]
            img_blocks = sum(1 for block in blocks if block.get("type") == 1)
            text_blocks = sum(1 for block in blocks if "lines" in block)
            if img_blocks > text_blocks * 2:
                slide_pages += 1

        return slide_pages >= max(2, sample_count // 2)

    def _extract_slide_page(self, page) -> str:
        spans: list[tuple[float, float, float, float, str]] = []
        page_rect = page.rect
        wider_clip = fitz.Rect(page_rect.x0, page_rect.y0, page_rect.width * 1.5, page_rect.height * 1.2)

        for block in page.get_text("dict", clip=wider_clip)["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        spans.append((span["origin"][1], span["origin"][0], span["bbox"][2], span["size"], text))

        if not spans:
            return ""

        spans.sort(key=lambda item: (item[0], item[1]))
        tolerance = 6.0
        lines: list[list[tuple[float, float, float, float, str]]] = []
        current_line = [spans[0]]
        current_y = spans[0][0]

        for span in spans[1:]:
            if abs(span[0] - current_y) <= tolerance:
                current_line.append(span)
            else:
                lines.append(current_line)
                current_line = [span]
                current_y = span[0]
        lines.append(current_line)

        result: list[str] = []
        for line_spans in lines:
            line_spans.sort(key=lambda item: item[1])
            parts: list[str] = []
            prev_x_end = 0.0
            prev_text = ""
            for _y, x, x_end, size, text in line_spans:
                if parts:
                    gap = x - prev_x_end
                    needs_space = gap > size * 0.1 or not self._should_merge_spans(prev_text, text)
                    if needs_space:
                        parts.append(" ")
                parts.append(text)
                prev_x_end = x_end
                prev_text = text

            line_text = "".join(parts).strip()
            if line_text:
                result.append(line_text)

        return "\n".join(result)


class DocumentIngestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pdf_extractor = PdfExtractor(settings)

    def ingest_paths(self, paths: list[Path], progress_callback=None) -> tuple[list[Document], list[Path]]:
        documents: list[Document] = []
        skipped: list[Path] = []

        for path in paths:
            if not path.exists() or not path.is_file():
                skipped.append(path)
                continue

            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pdf_documents, markdown_sections = self.pdf_extractor.extract_pdf(path, progress_callback=progress_callback)
                documents.extend(pdf_documents)
                self.pdf_extractor.export_markdown(path, pdf_documents, markdown_sections)
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
