from __future__ import annotations

import io
from functools import cached_property

import numpy as np

from app.config import Settings

try:
    from PIL import Image
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover
    Image = None
    PaddleOCR = None


class OcrEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return Image is not None and PaddleOCR is not None

    @cached_property
    def client(self):
        if not self.available:
            return None
        return PaddleOCR(
            lang=self.settings.ocr_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_text_from_pixmap(self, pixmap_bytes: bytes) -> str:
        if not self.available or self.client is None:
            return ""

        try:
            image = Image.open(io.BytesIO(pixmap_bytes)).convert("RGB")
            image_array = np.array(image)
            result = self.client.predict(image_array)
        except Exception:
            return ""

        lines: list[str] = []
        for page_result in result or []:
            rec_texts = page_result.get("rec_texts", [])
            for text in rec_texts:
                stripped = str(text).strip()
                if stripped:
                    lines.append(stripped)
        return "\n".join(lines)
