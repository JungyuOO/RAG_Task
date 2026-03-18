from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.rag.artifacts import extracted_markdown_path
from app.rag.ocr import OcrEngine
from app.rag.types import Document
from app.rag.utils import normalize_text, stable_hash

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


class DocumentIngestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ocr_engine = OcrEngine(settings)

    def ingest_paths(self, paths: list[Path]) -> tuple[list[Document], list[Path]]:
        documents: list[Document] = []
        skipped: list[Path] = []

        for path in paths:
            if not path.exists() or not path.is_file():
                skipped.append(path)
                continue

            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pdf_documents, markdown_sections = self._ingest_pdf(path)
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

    def _ingest_pdf(self, path: Path) -> tuple[list[Document], list[dict[str, str | int]]]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required to parse PDF files.")

        pdf = fitz.open(path)
        documents: list[Document] = []
        markdown_sections: list[dict[str, str | int]] = []
        for index, page in enumerate(pdf, start=1):
            extracted_text = page.get_text("text")
            text = normalize_text(extracted_text)
            loader = "pdf_text"

            if len(text) < 40:
                pix = page.get_pixmap(dpi=self.settings.ocr_dpi)
                extracted_text = self.ocr_engine.extract_text_from_pixmap(pix.tobytes("png"))
                text = normalize_text(extracted_text)
                if text:
                    loader = "pdf_ocr"

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
                    "text": self._render_markdown_text(extracted_text),
                }
            )
        pdf.close()
        return documents, markdown_sections

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

    def _render_markdown_text(self, text: str) -> str:
        rendered = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
        rendered = "\n".join(line.rstrip() for line in rendered.splitlines())
        rendered = rendered.strip()
        return rendered or normalize_text(text)
