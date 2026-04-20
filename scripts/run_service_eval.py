from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "data" / "service_eval_dataset_v1.json"
DEFAULT_OUTPUT_JSON = ROOT / "tests" / "results" / "service-eval" / "latest.json"
DEFAULT_OUTPUT_MD = ROOT / "tests" / "results" / "service-eval" / "latest.md"
_HANGUL_RE = re.compile(r"[가-힣]")
_GENERIC_SUBTOPIC_TOKENS = {
    "what", "is", "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
    "how", "does", "do", "can", "should", "tell", "me", "about", "from", "that", "this",
    "it", "be", "are", "at", "by", "only", "just", "into", "also", "than", "difference",
    "설명", "설명해줘", "알려줘", "뭐야", "무엇", "어떻게", "하는지", "되는지", "정리", "요약", "먼저",
    "그리고", "이랑", "하고", "또", "관련", "같이", "따로", "기준", "정도", "위해", "대한", "에서",
    "있는", "없는", "보여줘", "말해줘", "흐름", "개요", "싶어", "해줘", "짧게", "번호로",
    "필드", "의미", "출력", "고려", "제약사항", "주의", "점",
}


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_case(case: dict, response: dict, latency_ms: float, turn_trace: list[dict]) -> dict:
    sources = response.get("sources") or []
    answer = str(response.get("answer") or "")
    citation_alignment = evaluate_citation_alignment(answer, sources)
    source_match = any(_source_matches_case(source, case) for source in sources)
    section_match = any(_section_matches_case(source, case) for source in sources)
    retrieved_section_titles = [
        str((source.get("metadata") or {}).get("section_title") or "")
        for source in sources
    ]
    retrieved_source_paths = [
        str(source.get("relative_source_path") or source.get("source_path") or "")
        for source in sources
    ]
    matching_source_count = sum(1 for source in sources if _source_matches_case(source, case))
    source_purity = round(matching_source_count / max(len(sources), 1), 3) if sources else 0.0
    primary_source_match = _source_matches_case(sources[0], case) if sources else False
    primary_section_match = _section_matches_case(sources[0], case) if sources else False
    korean_answer = evaluate_korean_answer(case, answer)
    followup_source_retained = evaluate_followup_source_retention(turn_trace)
    subtopics = extract_subtopics(case)
    subtopic_coverage_count = sum(
        1 for subtopic in subtopics if any(source_supports_subtopic(source, subtopic) for source in sources)
    )
    subtopic_coverage = round(subtopic_coverage_count / max(len(subtopics), 1), 3) if subtopics else 0.0
    relevant_source_count = sum(
        1 for source in sources if any(source_supports_subtopic(source, subtopic) for subtopic in subtopics)
    )
    source_relevance_ratio = round(relevant_source_count / max(len(sources), 1), 3) if sources else 0.0
    strict_ok = (
        bool(sources)
        and citation_alignment
        and source_match
        and korean_answer
        and followup_source_retained
        and subtopic_coverage >= 1.0
        and source_relevance_ratio >= 0.67
    )
    return {
        "id": case["id"],
        "question": case["question"],
        "scenario": case.get("scenario", "single_turn"),
        "turn_count": len(case.get("turns") or [case["question"]]),
        "latency_ms": round(latency_ms, 1),
        "lane": response.get("lane"),
        "mode": response.get("mode"),
        "source_count": len(sources),
        "has_citation": "[" in answer and "]" in answer,
        "citation_alignment": citation_alignment,
        "source_match": source_match,
        "section_match": section_match,
        "ok": bool(sources) and (source_match or section_match) and citation_alignment,
        "strict_ok": strict_ok,
        "primary_source_match": primary_source_match,
        "primary_section_match": primary_section_match,
        "matching_source_count": matching_source_count,
        "source_purity": source_purity,
        "korean_answer": korean_answer,
        "followup_source_retained": followup_source_retained,
        "subtopics": subtopics,
        "subtopic_coverage": subtopic_coverage,
        "subtopic_coverage_count": subtopic_coverage_count,
        "relevant_source_count": relevant_source_count,
        "source_relevance_ratio": source_relevance_ratio,
        "expected_source_path": case.get("source_path"),
        "expected_section_title": case.get("section_title"),
        "retrieved_source_paths": retrieved_source_paths,
        "retrieved_section_titles": retrieved_section_titles,
        "answer_preview": answer[:400],
        "turn_trace": turn_trace,
    }


def _source_matches_case(source: dict, case: dict) -> bool:
    expected = str(case.get("source_path") or "")
    actual_rel = str(source.get("relative_source_path") or "")
    actual_abs = str(source.get("source_path") or "")
    expected_file = expected.rsplit("/", 1)[-1]
    return (
        actual_rel.endswith(expected)
        or actual_abs.endswith(expected)
        or actual_rel.endswith("/" + expected_file)
        or actual_abs.endswith("/" + expected_file)
        or actual_rel.endswith("\\" + expected_file)
        or actual_abs.endswith("\\" + expected_file)
    )


def _section_matches_case(source: dict, case: dict) -> bool:
    expected = str(case.get("section_title") or "").casefold()
    actual = str((source.get("metadata") or {}).get("section_title") or "").casefold()
    return bool(expected and expected in actual)


def evaluate_korean_answer(case: dict, answer: str) -> bool:
    turns = case.get("turns") or [case.get("question", "")]
    expects_korean = any(_HANGUL_RE.search(str(turn or "")) for turn in turns)
    if not expects_korean:
        return True
    hangul_count = len(_HANGUL_RE.findall(answer))
    return hangul_count >= 8


def evaluate_followup_source_retention(turn_trace: list[dict]) -> bool:
    if len(turn_trace) <= 1:
        return True
    previous_paths: set[str] = set()
    for turn in turn_trace[:-1]:
        previous_paths.update(str(path or "") for path in (turn.get("retrieved_source_paths") or []))
    final_paths = {str(path or "") for path in (turn_trace[-1].get("retrieved_source_paths") or [])}
    if not previous_paths or not final_paths:
        return False
    return bool(previous_paths.intersection(final_paths))


def extract_subtopics(case: dict) -> list[str]:
    turns = case.get("turns") or [case.get("question", "")]
    combined = " ".join(str(turn or "").strip() for turn in turns if str(turn or "").strip())
    if not combined:
        return []
    normalized = combined
    for marker in (" 그리고 ", "이랑 ", " 하고 ", " 및 ", " also ", " and ", "/", ","):
        normalized = normalized.replace(marker, "|")
    parts = [part.strip() for part in normalized.replace("?", "|").split("|") if part.strip()]
    return parts[:4] or [combined]


def source_supports_subtopic(source: dict, subtopic: str) -> bool:
    clause_tokens = _subtopic_tokens(subtopic)
    if not clause_tokens:
        return False
    source_context = " ".join(
        [
            str((source.get("metadata") or {}).get("section_title") or source.get("label") or ""),
            str((source.get("metadata") or {}).get("preview_text") or ""),
            str((source.get("metadata") or {}).get("synthesis_text") or ""),
        ]
    )
    source_tokens = _subtopic_tokens(source_context)
    if not source_tokens:
        return False
    overlap = len(clause_tokens.intersection(source_tokens))
    if overlap == 0:
        return False
    title_tokens = _subtopic_tokens(str((source.get("metadata") or {}).get("section_title") or source.get("label") or ""))
    title_overlap = len(clause_tokens.intersection(title_tokens))
    coverage = overlap / max(len(clause_tokens), 1)
    return title_overlap >= 1 or coverage >= 0.45 or (len(clause_tokens) <= 3 and overlap >= 1)


def _subtopic_tokens(text: str) -> set[str]:
    return {
        token
        for token in _stable_tokens(text)
        if token not in _GENERIC_SUBTOPIC_TOKENS
    }


def _stable_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9가-힣_-]+", str(text or "").casefold())
        if len(token) >= 2
    }


def evaluate_citation_alignment(answer: str, sources: list[dict]) -> bool:
    if not answer.strip():
        return False
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
    if not paragraphs:
        return False
    has_citation = False
    for paragraph in paragraphs:
        citations = [int(match) for match in re.findall(r"\[(\d+)\]", paragraph)]
        if not citations:
            continue
        has_citation = True
        paragraph_tokens = _stable_tokens(re.sub(r"\[(\d+)\]", "", paragraph))
        if not paragraph_tokens:
            return False
        for citation in citations:
            if citation < 1 or citation > len(sources):
                return False
            source = sources[citation - 1]
            source_context = " ".join(
                [
                    str((source.get("metadata") or {}).get("section_title") or source.get("label") or ""),
                    str((source.get("metadata") or {}).get("preview_text") or ""),
                    str((source.get("metadata") or {}).get("synthesis_text") or ""),
                ]
            )
            source_tokens = _stable_tokens(source_context)
            overlap = paragraph_tokens.intersection(source_tokens)
            if len(overlap) < 1:
                return False
    return has_citation


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9가-힣_-]+", str(text or "").casefold())
        if len(token) >= 2
    }


def write_markdown(results: list[dict], output_path: Path) -> None:
    lines = ["# Service Eval", ""]
    for item in results:
        lines.extend(
            [
                f"## {item['id']}",
                f"- question: {item['question']}",
                f"- scenario: {item['scenario']}",
                f"- turn_count: {item['turn_count']}",
                f"- lane/mode: {item['lane']} / {item['mode']}",
                f"- latency_ms: {item['latency_ms']}",
                f"- source_count: {item['source_count']}",
                f"- ok: {item['ok']}",
                f"- strict_ok: {item['strict_ok']}",
                f"- citation_alignment: {item['citation_alignment']}",
                f"- source_match: {item['source_match']}",
                f"- section_match: {item['section_match']}",
                f"- primary_source_match: {item['primary_source_match']}",
                f"- primary_section_match: {item['primary_section_match']}",
                f"- source_purity: {item['source_purity']}",
                f"- korean_answer: {item['korean_answer']}",
                f"- followup_source_retained: {item['followup_source_retained']}",
                f"- subtopic_coverage: {item['subtopic_coverage']}",
                f"- source_relevance_ratio: {item['source_relevance_ratio']}",
                f"- preview: {item['answer_preview']}",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))[: args.limit]
    client = httpx.Client(timeout=120)
    results: list[dict] = []

    for case in dataset:
        started = time.perf_counter()
        turns = case.get("turns") or [case["question"]]
        history = []
        response = None
        turn_trace: list[dict] = []
        for turn in turns:
            response = client.post(
                f"{args.base_url.rstrip('/')}/api/v1/chat/query",
                json={
                    "message": turn,
                    "connection_id": "",
                    "namespace": "",
                    "history": history,
                },
            )
            response.raise_for_status()
            assistant = response.json()
            turn_trace.append(
                {
                    "message": turn,
                    "lane": assistant.get("lane"),
                    "mode": assistant.get("mode"),
                    "retrieved_source_paths": [
                        source.get("relative_source_path") or source.get("source_path") or ""
                        for source in (assistant.get("sources") or [])
                    ],
                    "retrieved_section_titles": [
                        (source.get("metadata") or {}).get("section_title") or ""
                        for source in (assistant.get("sources") or [])
                    ],
                    "answer_preview": str(assistant.get("answer") or "")[:240],
                }
            )
            history.extend(
                [
                    {"role": "user", "text": turn, "lane": ""},
                    {
                        "role": "assistant",
                        "text": assistant.get("answer", ""),
                        "lane": assistant.get("lane", ""),
                        "source_paths": [
                            source.get("source_path") or ""
                            for source in (assistant.get("sources") or [])
                        ],
                        "resource_names": [
                            source.get("label") or ""
                            for source in (assistant.get("sources") or [])
                            if source.get("source_type") == "live"
                        ],
                        "namespace": next(
                            (
                                source.get("namespace") or ""
                                for source in (assistant.get("sources") or [])
                                if source.get("namespace")
                            ),
                            "",
                        ),
                    },
                ]
            )
        latency_ms = (time.perf_counter() - started) * 1000
        assert response is not None
        results.append(score_case(case, response.json(), latency_ms, turn_trace))

    summary = {
        "count": len(results),
        "ok_count": sum(1 for item in results if item["ok"]),
        "strict_ok_count": sum(1 for item in results if item["strict_ok"]),
        "korean_answer_count": sum(1 for item in results if item["korean_answer"]),
        "primary_source_match_count": sum(1 for item in results if item["primary_source_match"]),
        "followup_source_retained_count": sum(1 for item in results if item["followup_source_retained"]),
        "avg_latency_ms": round(statistics.mean(item["latency_ms"] for item in results), 1) if results else 0.0,
        "results": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results, Path(args.output_md))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
