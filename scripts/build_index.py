from __future__ import annotations

from app.config import get_settings
from app.rag.pipeline import RagPipeline


def main() -> None:
    settings = get_settings()
    pipeline = RagPipeline(settings)
    source_files = [
        path
        for path in settings.rag_source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    result = pipeline.rebuild_index(source_files)
    print(result)


if __name__ == "__main__":
    main()
