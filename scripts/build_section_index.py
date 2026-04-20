from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus" / "pdfs" / "official" / "en"
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "section_index.json"
DEFAULT_HARD_OUTPUT = ROOT / "tests" / "data" / "section_hard_pairs.json"
RELATIVE_PREFIX = "official/en"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "by", "with",
    "using", "how", "what", "is", "are", "be", "your", "this", "that", "these",
    "those", "from", "at", "as", "it", "its", "can", "you", "if", "when", "into",
}


def _tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.casefold())
    return {tok for tok in cleaned.split() if len(tok) >= 3 and tok not in STOPWORDS}


def _first_body_preview(body_lines: list[str], limit: int = 240) -> str:
    joined = " ".join(line.strip() for line in body_lines if line.strip())
    cleaned = re.sub(r"\s+", " ", joined).strip()
    return cleaned[:limit]


def parse_markdown(path: Path, relative_source_path: str) -> list[dict]:
    sections: list[dict] = []
    section_path: list[str] = []
    current_title = ""
    current_level = 0
    current_body: list[str] = []
    in_code = False

    def flush() -> None:
        if not current_title:
            return
        sections.append(
            {
                "source_path": relative_source_path,
                "level": current_level,
                "section_title": current_title,
                "section_path": list(section_path),
                "body_preview": _first_body_preview(current_body),
            }
        )

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            current_body.append(raw)
            continue
        if in_code:
            current_body.append(raw)
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            section_path = section_path[: max(level - 1, 0)]
            section_path.append(title)
            current_title = title
            current_level = level
            current_body = []
            continue
        current_body.append(raw)
    flush()
    return sections


def detect_hard_pairs(sections: list[dict], min_shared: int = 2) -> list[dict]:
    by_parent: dict[tuple[str, tuple[str, ...]], list[dict]] = defaultdict(list)
    for section in sections:
        parent = tuple(section["section_path"][:-1])
        by_parent[(section["source_path"], parent)].append(section)

    hard_pairs: list[dict] = []
    for (source_path, parent), siblings in by_parent.items():
        if len(siblings) < 2:
            continue
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                left = siblings[i]
                right = siblings[j]
                shared = _tokens(left["section_title"]).intersection(_tokens(right["section_title"]))
                if len(shared) >= min_shared:
                    hard_pairs.append(
                        {
                            "source_path": source_path,
                            "parent": list(parent),
                            "left_title": left["section_title"],
                            "right_title": right["section_title"],
                            "left_preview": left["body_preview"],
                            "right_preview": right["body_preview"],
                            "shared_tokens": sorted(shared),
                        }
                    )
    hard_pairs.sort(key=lambda item: (item["source_path"], -len(item["shared_tokens"])))
    return hard_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hard-output", default=str(DEFAULT_HARD_OUTPUT))
    parser.add_argument("--min-shared-tokens", type=int, default=2)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    md_files = sorted(corpus_dir.glob("*.md"))
    all_sections: list[dict] = []
    for md in md_files:
        relative = f"{RELATIVE_PREFIX}/{md.name}"
        all_sections.extend(parse_markdown(md, relative))

    hard_pairs = detect_hard_pairs(all_sections, min_shared=args.min_shared_tokens)

    summary = {
        "doc_count": len(md_files),
        "section_count": len(all_sections),
        "hard_pair_count": len(hard_pairs),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_sections, ensure_ascii=False, indent=2), encoding="utf-8")

    hard_output_path = Path(args.hard_output)
    hard_output_path.parent.mkdir(parents=True, exist_ok=True)
    hard_output_path.write_text(json.dumps(hard_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
