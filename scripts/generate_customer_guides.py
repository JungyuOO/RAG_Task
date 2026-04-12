from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.rag.utils import normalize_markdown_display_text


OPENAI_CHAT_BASE_URL = "https://api.openai.com/v1"


def load_settings() -> dict[str, str]:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    return {
        "api_key": api_key,
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip(),
    }


def iter_markdown_files(root: Path, version: str) -> list[Path]:
    prefix = f"OpenShift_Container_Platform-{version}-"
    return sorted(path for path in root.glob("*.md") if path.name.startswith(prefix))


def read_markdown_corpus(paths: list[Path]) -> list[dict[str, str]]:
    return [{"file_name": path.name, "text": path.read_text(encoding="utf-8")} for path in paths]


def extract_headings(text: str, limit: int = 40) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            if heading and heading not in headings:
                headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def build_topic_discovery_context(corpus: list[dict[str, str]], *, max_docs: int = 8) -> str:
    sections: list[str] = []
    for doc in corpus[:max_docs]:
        headings = extract_headings(doc["text"], limit=24)
        preview = "\n".join(headings[:24])
        sections.append(f"[Source: {doc['file_name']}]\n{preview}")
    return "\n\n".join(sections)


def call_openai_chat(settings: dict[str, str], messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    headers = {
        "Authorization": f"Bearer {settings['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": temperature,
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{OPENAI_CHAT_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def sanitize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.strip().casefold())
    return normalized.strip("-")


def propose_topics_with_openai(version: str, corpus: list[dict[str, str]], settings: dict[str, str], *, count: int) -> list[dict[str, object]]:
    discovery_context = build_topic_discovery_context(corpus)
    system_prompt = (
        "You analyze official OpenShift documentation headings and propose practical customer-handbook topics.\n"
        "Return only JSON.\n"
    )
    user_prompt = (
        f"Target OCP version: {version}\n"
        f"Need {count} topic proposals.\n"
        "Use only topics clearly supported by the official headings/context below.\n"
        "Prefer practical customer-facing areas such as deployment, networking, storage, access control, routing, troubleshooting, and operations.\n"
        "For each topic return:\n"
        "- slug: short ascii key\n"
        "- title: Korean guide title\n"
        "- keywords: list of grounded search keywords\n"
        "- customer_needs: 3 short Korean strings describing what a customer team would need\n\n"
        "JSON format:\n"
        "{\"topics\":[{\"slug\":\"...\",\"title\":\"...\",\"keywords\":[\"...\"],\"customer_needs\":[\"...\",\"...\",\"...\"]}]}\n\n"
        "Official headings/context:\n"
        + discovery_context
    )
    raw = call_openai_chat(
        settings,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    parsed = json.loads(raw)
    topics = parsed.get("topics", [])
    if not isinstance(topics, list) or not topics:
        raise SystemExit("Failed to derive topic proposals from extracted markdown.")

    cleaned: list[dict[str, object]] = []
    for item in topics[:count]:
        slug = sanitize_slug(str(item.get("slug") or "topic"))
        title = str(item.get("title") or slug).strip()
        keywords = [str(value).strip() for value in item.get("keywords", []) if str(value).strip()]
        customer_needs = [str(value).strip() for value in item.get("customer_needs", []) if str(value).strip()]
        if not keywords:
            continue
        cleaned.append(
            {
                "slug": slug,
                "title": title,
                "keywords": keywords,
                "customer_needs": customer_needs[:3] or [
                    "운영 체크포인트가 필요함",
                    "실무 예시가 필요함",
                    "점검 및 트러블슈팅 기준이 필요함",
                ],
            }
        )
    if not cleaned:
        raise SystemExit("No usable topic proposals were produced.")
    return cleaned


def build_manual_topics_from_args(topics_arg: str) -> list[dict[str, object]]:
    topics: list[dict[str, object]] = []
    for raw in [value.strip() for value in topics_arg.split(",") if value.strip()]:
        topics.append(
            {
                "slug": sanitize_slug(raw),
                "title": f"{raw} 고객사 운영 가이드",
                "keywords": [raw],
                "customer_needs": [
                    "실무 운영 기준이 필요함",
                    "고객사 예시 코드가 필요함",
                    "점검 및 트러블슈팅 기준이 필요함",
                ],
            }
        )
    return topics


def extract_relevant_excerpt(text: str, keywords: list[str], *, max_chars: int = 10000) -> str:
    lines = text.splitlines()
    lowered_keywords = [keyword.casefold() for keyword in keywords]
    hits: list[int] = []
    for idx, line in enumerate(lines):
        lowered = line.casefold()
        if any(keyword in lowered for keyword in lowered_keywords):
            hits.append(idx)

    if not hits:
        excerpt = "\n".join(lines[: min(len(lines), 200)])
    else:
        windows: list[str] = []
        for hit in hits[:10]:
            start = max(hit - 8, 0)
            end = min(hit + 24, len(lines))
            windows.append("\n".join(lines[start:end]))
        excerpt = "\n\n---\n\n".join(windows)

    excerpt = re.sub(r"\n{3,}", "\n\n", excerpt).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit("\n", 1)[0].strip()
    return excerpt


def collect_official_context(corpus: list[dict[str, str]], keywords: list[str], *, limit: int = 4) -> list[dict[str, str]]:
    scored: list[tuple[int, dict[str, str]]] = []
    for doc in corpus:
        lowered = doc["text"].casefold()
        score = sum(lowered.count(keyword.casefold()) for keyword in keywords)
        if score <= 0:
            continue
        excerpt = extract_relevant_excerpt(doc["text"], keywords)
        scored.append((score, {"file_name": doc["file_name"], "excerpt": excerpt}))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in scored[:limit]]


def build_generation_prompt(version: str, topic: dict[str, object], official_context: list[dict[str, str]]) -> list[dict[str, str]]:
    context_sections = [f"[Source: {item['file_name']}]\n{item['excerpt']}" for item in official_context]
    system_prompt = (
        "You write customer-internal OpenShift handbooks.\n"
        "Use the official OCP excerpts as the source of truth, but rewrite them as an internal customer manual.\n"
        "Return Korean markdown only.\n"
        "The document must feel like a real customer team's internal handbook, not a generic summary.\n"
        "When you include YAML or CLI examples, never copy names verbatim from the source.\n"
        "Rewrite namespace, labels, app names, config names, and route hosts into plausible fictional customer values.\n"
        "Keep OCP semantics correct.\n"
        "Do not mention AI generation.\n"
        "Write enough detail for a 10-20 page handbook when rendered to PDF.\n"
    )
    user_prompt = (
        f"Target OCP version: {version}\n"
        f"Guide title: {topic['title']}\n"
        f"Customer needs: {', '.join(topic['customer_needs'])}\n\n"
        "Write a customer handbook with these Korean sections in order:\n"
        "1. 개요\n"
        "2. 적용 범위와 전제 조건\n"
        "3. 실제 고객사 운영/개발 관점 설명\n"
        "4. 고객사 예시 YAML/CLI\n"
        "5. 운영 팁\n"
        "6. 점검 체크리스트\n"
        "7. 자주 하는 실수와 대응\n"
        "8. 참고한 공식 문서 요약\n\n"
        "Tone requirements:\n"
        "- Write clear Korean for operators and delivery teams.\n"
        "- Use labels like '권장', '주의', '점검', '운영 기준' where helpful.\n"
        "- Include concrete operational advice, not just paraphrased summaries.\n"
        "- Avoid markdown noise such as repeated separators or unnecessary blank lines.\n\n"
        "Official context:\n"
        + "\n\n".join(context_sections)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def postprocess_generated_markdown(markdown: str) -> str:
    lines = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    output: list[str] = []
    in_code_block = False
    last_blank = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            output.append(stripped or "```")
            in_code_block = not in_code_block
            last_blank = False
            continue

        if in_code_block:
            output.append(line.rstrip())
            last_blank = False
            continue

        if not stripped:
            if output and not last_blank:
                output.append("")
                last_blank = True
            continue

        if stripped.startswith("#"):
            cleaned_heading = re.sub(r"\s+", " ", stripped)
            if output and output[-1] != "":
                output.append("")
            output.append(cleaned_heading)
            last_blank = False
            continue

        if "|" in stripped:
            if re.fullmatch(r"\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$", stripped):
                output.append(stripped)
            else:
                cells = [re.sub(r"\s+", " ", cell.strip()) for cell in stripped.strip("|").split("|")]
                output.append("| " + " | ".join(cell for cell in cells if cell) + " |")
            last_blank = False
            continue

        cleaned = normalize_markdown_display_text(stripped)
        if cleaned:
            output.append(cleaned)
            last_blank = False

    while output and not output[-1].strip():
        output.pop()
    return "\n".join(output).strip() + "\n"


def build_output_text(version: str, topic: dict[str, object], generated_markdown: str, official_context: list[dict[str, str]]) -> str:
    sources = "\n".join(f"- {item['file_name']}" for item in official_context)
    cleaned_markdown = postprocess_generated_markdown(generated_markdown)
    front_matter = (
        "---\n"
        f"product: OCP\nversion: {version}\n"
        "doc_type: operation_manual\n"
        "document_group: customer_generated\n"
        f"topic: {topic['slug']}\n"
        "generated_by: openai_script\n"
        "---\n\n"
    )
    source_footer = f"\n## 참고한 공식 문서\n{sources}\n"
    return front_matter + cleaned_markdown + source_footer


def generate_guides(version: str, topics: list[dict[str, object]], source_root: Path, output_root: Path, settings: dict[str, str]) -> list[Path]:
    source_files = iter_markdown_files(source_root, version)
    if not source_files:
        raise SystemExit(f"No extracted markdown files found for version {version}.")

    corpus = read_markdown_corpus(source_files)
    output_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for topic in topics:
        official_context = collect_official_context(corpus, list(topic["keywords"]), limit=4)
        if not official_context:
            continue
        messages = build_generation_prompt(version, topic, official_context)
        generated_markdown = call_openai_chat(settings, messages, temperature=0.2)
        output_text = build_output_text(version, topic, generated_markdown, official_context)
        output_path = output_root / f"ocp-{version}-{sanitize_slug(str(topic['slug']))}-customer-guide.md"
        output_path.write_text(output_text, encoding="utf-8")
        created.append(output_path)
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate customer-facing OCP guides from extracted official markdown.")
    parser.add_argument("--version", default="4.15", help="OCP version, for example 4.15")
    parser.add_argument("--topics", default="", help="Optional comma-separated topic hints. If omitted, topics are proposed from extracted markdown.")
    parser.add_argument("--auto-topics", type=int, default=3, help="How many topic proposals to generate when --topics is omitted.")
    parser.add_argument("--source-root", default="data/extracted_markdown", help="Extracted markdown directory")
    parser.add_argument("--output-root", default="data/corpus/pdfs/generated", help="Generated markdown output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    source_root = Path(args.source_root)
    source_files = iter_markdown_files(source_root, args.version)
    corpus = read_markdown_corpus(source_files)

    if args.topics.strip():
        topics = build_manual_topics_from_args(args.topics)
    else:
        topics = propose_topics_with_openai(args.version, corpus, settings, count=max(args.auto_topics, 1))

    created = generate_guides(
        version=args.version,
        topics=topics,
        source_root=source_root,
        output_root=Path(args.output_root),
        settings=settings,
    )
    print(
        json.dumps(
            {
                "selected_topics": topics,
                "generated_files": [str(path) for path in created],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
