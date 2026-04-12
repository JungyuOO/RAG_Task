"""
현재 인덱싱된 PDF(OCP 4.15 전체 + 4.16 일부) 기준 멀티턴 10회 테스트.

v.3.0.1 spec 기준 검증:
  1. 10턴 이상 멀티턴 대화가 유지되는지 (세션 메모리 / 쿼리 재작성)
  2. 답변의 문장마다 인라인 출처 태그([file.pdf p.N] 형식)가 달리는지
  3. 실제 RAG가 문서를 참조하여 답변했는지 (answer_citations payload 확인)

출력:
  - 콘솔: 턴별 Q/A 요약 + 최종 통계
  - 파일: tests/results/multiturn_<session_id>.json (사람이 읽기 쉬운 구조)

사용법:
  python scripts/test_multiturn_citations.py
  python scripts/test_multiturn_citations.py --base-url http://localhost:8000 --version 4.15
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Iterator


# 10턴 시나리오 (후속 질문에서 대명사/생략으로 메모리 의존성 유발)
TURNS_4_15 = [
    "OCP 4.15 아키텍처에서 컨트롤 플레인은 어떤 구성 요소로 이루어져 있나요?",
    "그 중에서 etcd는 어떤 역할을 하나요?",                              # 앞 답변의 "구성 요소" 참조
    "etcd 백업은 어떻게 수행하나요?",                                     # 동일 토픽 유지
    "복원(restore) 절차도 알려주세요.",                                   # "백업" 대비
    "OCP 4.15에서 사용자 인증 방식에는 어떤 것들이 지원되나요?",
    "그 중 LDAP 연동 설정은 어떻게 하나요?",                              # "그 중" = 인증 방식
    "RBAC에서 ClusterRole과 Role의 차이는 무엇인가요?",
    "AWS에 설치할 때 필요한 최소 권한 IAM 정책은 무엇인가요?",
    "네트워크 정책(NetworkPolicy)의 기본 동작 방식은 어떤가요?",
    "지금까지 얘기한 내용 중 etcd 백업 명령어만 다시 정리해줘.",        # 명시적 과거 참조
]


# 인라인 인용 태그 패턴 (answer_citation.py:108-125 와 동일 패턴군)
CITATION_PATTERNS = [
    re.compile(r"\[[^\[\]\n]+?\.pdf\]\s*p\.\d+(?:-\d+)?", re.IGNORECASE),
    re.compile(r"\[[^\[\]\n]+?\.pdf\s+p\.\d+(?:-\d+)?\]", re.IGNORECASE),
    re.compile(r"\[source:[^:]+:p\d+:L\d+-\d+\]", re.IGNORECASE),
]

# 한국어/영어 문장 분리 (보수적으로)
SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=음\.)\s+|(?<=임\.)\s+|(?<=니다\.)\s+"
)

LLM_FAILURE_MARKERS = (
    "LLM 응답 생성에 실패했습니다",
    "검색된 문맥 기준으로 핵심만",
    "답변 생성 중 오류",
)


def has_citation(text: str) -> bool:
    return any(p.search(text) for p in CITATION_PATTERNS)


def find_citations_in_text(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in CITATION_PATTERNS:
        hits.extend(pattern.findall(text))
    return hits


def split_sentences(answer: str) -> list[str]:
    """Sources 라인과 코드 블록은 검증 대상에서 제외"""
    lines = []
    in_code = False
    for line in answer.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.strip().startswith("Sources:"):
            continue
        lines.append(line)
    text = " ".join(line.strip() for line in lines if line.strip())
    if not text:
        return []
    raw = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) >= 5]


def evaluate_sentences(answer: str) -> list[dict]:
    """문장별 평가 결과"""
    sentences = split_sentences(answer)
    evaluations = []
    for sent in sentences:
        citations_found = find_citations_in_text(sent)
        evaluations.append({
            "text": sent,
            "has_citation": bool(citations_found),
            "inline_citations": citations_found,
        })
    return evaluations


def detect_llm_failure(answer: str) -> bool:
    return any(marker in answer for marker in LLM_FAILURE_MARKERS)


def stream_chat(base_url: str, session_id: str, message: str, version_tag: str | None) -> Iterator[dict]:
    payload = {"session_id": session_id, "message": message}
    if version_tag:
        payload["version_tag"] = version_tag
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Id": "test-multiturn-client",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_bytes in resp:
            raw = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not raw or not raw.startswith("data: "):
                continue
            try:
                yield json.loads(raw[len("data: "):])
            except json.JSONDecodeError:
                continue


def run_turn(base_url: str, session_id: str, turn_idx: int, message: str, version_tag: str | None) -> dict:
    print(f"\n{'=' * 78}")
    print(f"[Turn {turn_idx}] Q: {message}")
    print("-" * 78)

    answer_parts: list[str] = []
    context_event: dict | None = None
    status_events: list[dict] = []
    saw_done = False
    token_cached_count = 0
    t0 = time.time()

    replaced_answer: str | None = None
    for event in stream_chat(base_url, session_id, message, version_tag):
        etype = event.get("type")
        if etype == "token":
            answer_parts.append(event.get("content", ""))
            if event.get("cached"):
                token_cached_count += 1
        elif etype == "replace_answer":
            replaced_answer = event.get("content", "")
        elif etype == "context":
            context_event = event
        elif etype == "status":
            status_events.append({"stage": event.get("stage"), "message": event.get("message")})
        elif etype == "done":
            saw_done = True
            break

    elapsed = time.time() - t0
    answer = replaced_answer if replaced_answer is not None else "".join(answer_parts).strip()
    print(f"A: {answer[:600]}{'...' if len(answer) > 600 else ''}")

    sentence_eval = evaluate_sentences(answer)
    total_sents = len(sentence_eval)
    cited_sents = sum(1 for s in sentence_eval if s["has_citation"])
    missing = total_sents - cited_sents
    llm_failed = detect_llm_failure(answer)

    ctx = context_event or {}
    answer_citations = ctx.get("answer_citations", [])
    grounded_pages = ctx.get("grounded_pages", [])
    items = ctx.get("items", [])

    cited_pages = [
        {
            "file_name": c.get("file_name"),
            "page_number": c.get("page_number"),
            "score": c.get("score"),
            "origin": c.get("origin"),
        }
        for c in answer_citations
    ]
    retrieved_sources = []
    version_re = re.compile(r"ocp-(\d+\.\d+)", re.IGNORECASE)
    for item in items[:10]:
        source_path = item.get("source_path", "") or ""
        version_match = version_re.search(source_path)
        retrieved_sources.append({
            "file_name": Path(source_path).name,
            "page_number": item.get("page_number"),
            "section_title": item.get("section_title"),
            "rerank_score": item.get("rerank_score"),
            "dense_score": item.get("dense_score"),
            "sparse_score": item.get("sparse_score"),
            "version_tag": version_match.group(1) if version_match else None,
        })

    version_mismatch = False
    if version_tag and retrieved_sources:
        top_versions = [s["version_tag"] for s in retrieved_sources[:5] if s["version_tag"]]
        if top_versions and not any(v == version_tag for v in top_versions):
            version_mismatch = True

    print("-" * 78)
    print(
        f"[elapsed] {elapsed:.1f}s | 문장={total_sents} | "
        f"인용있는문장={cited_sents} | 누락={missing} | "
        f"payload.answer_citations={len(answer_citations)} | "
        f"LLM실패={llm_failed} | version_mismatch={version_mismatch}"
    )
    if cited_pages:
        top5 = ", ".join(f"{p['file_name']}:p.{p['page_number']}" for p in cited_pages[:5])
        print(f"[인용 페이지] {top5}{' ...' if len(cited_pages) > 5 else ''}")

    return {
        "turn": turn_idx,
        "question": message,
        "answer": answer,
        "answer_length": len(answer),
        "elapsed_sec": round(elapsed, 2),
        "done": saw_done,
        "token_cached_count": token_cached_count,
        "llm_generation_failed": llm_failed,
        "version_mismatch": version_mismatch,
        "sentence_stats": {
            "total": total_sents,
            "with_citation": cited_sents,
            "missing_citation": missing,
            "coverage_pct": round((cited_sents / total_sents * 100) if total_sents else 0.0, 1),
        },
        "sentence_evaluation": sentence_eval,
        "answer_citations": cited_pages,
        "retrieved_sources": retrieved_sources,
        "grounded_pages_count": len(grounded_pages),
        "context_mode": ctx.get("mode"),
        "top_score": ctx.get("top_score"),
        "rewritten_query": ctx.get("query"),
        "status_events": status_events,
    }


def build_summary(results: list[dict], version_tag: str | None, session_id: str, base_url: str) -> dict:
    total_turns = len(TURNS_4_15)
    completed = [r for r in results if r.get("done")]
    total_sents = sum(r.get("sentence_stats", {}).get("total", 0) for r in results)
    total_cited = sum(r.get("sentence_stats", {}).get("with_citation", 0) for r in results)
    total_missing = sum(r.get("sentence_stats", {}).get("missing_citation", 0) for r in results)
    llm_failures = sum(1 for r in results if r.get("llm_generation_failed"))
    version_mismatches = sum(1 for r in results if r.get("version_mismatch"))
    answer_cite_total = sum(len(r.get("answer_citations", [])) for r in results)
    avg_elapsed = round(
        sum(r.get("elapsed_sec", 0) for r in results) / max(total_turns, 1), 2
    )

    return {
        "session_id": session_id,
        "version_tag": version_tag,
        "base_url": base_url,
        "executed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "turn_results": {
            "total": total_turns,
            "completed": len(completed),
            "avg_elapsed_sec": avg_elapsed,
            "llm_generation_failed": llm_failures,
            "version_mismatch": version_mismatches,
        },
        "citation_results": {
            "total_sentences": total_sents,
            "sentences_with_inline_citation": total_cited,
            "sentences_missing_citation": total_missing,
            "inline_coverage_pct": round(
                (total_cited / total_sents * 100) if total_sents else 0.0, 1
            ),
            "answer_citations_payload_total": answer_cite_total,
        },
        "spec_v301_checks": {
            "multiturn_10plus_maintained": len(completed) == total_turns,
            "every_sentence_has_inline_source_tag": total_missing == 0 and total_sents > 0,
            "version_tag_filter_works": version_mismatches == 0,
            "llm_generation_stable": llm_failures == 0,
        },
    }


def print_summary_table(summary: dict, results: list[dict]) -> None:
    print(f"\n{'#' * 78}")
    print("# 요약 (v.3.0.1 spec 검증)")
    print("#" * 78)
    turn = summary["turn_results"]
    cite = summary["citation_results"]
    checks = summary["spec_v301_checks"]
    print(f"턴 완주          : {turn['completed']}/{turn['total']}   (평균 {turn['avg_elapsed_sec']}s)")
    print(f"LLM 생성 실패    : {turn['llm_generation_failed']}")
    print(f"버전 불일치      : {turn['version_mismatch']}")
    print(f"문장 총계        : {cite['total_sentences']}")
    print(f"인라인 인용 커버 : {cite['sentences_with_inline_citation']}/{cite['total_sentences']} "
          f"({cite['inline_coverage_pct']}%)")
    print(f"payload 인용합계 : {cite['answer_citations_payload_total']}")
    print()
    print("[v.3.0.1 spec 체크]")
    for key, value in checks.items():
        mark = "PASS" if value else "FAIL"
        print(f"  [{mark}] {key}")
    print()
    print(f"{'turn':<5} {'문장':<5} {'인용':<5} {'누락':<5} {'payload':<8} {'sec':<7} {'LLM실패':<8} {'ver불일치':<10}")
    for r in results:
        if "error" in r:
            print(f"{r['turn']:<5} ERROR: {r['error']}")
            continue
        s = r["sentence_stats"]
        print(
            f"{r['turn']:<5} {s['total']:<5} {s['with_citation']:<5} {s['missing_citation']:<5} "
            f"{len(r['answer_citations']):<8} {r['elapsed_sec']:<7.1f} "
            f"{str(r['llm_generation_failed']):<8} {str(r['version_mismatch']):<10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--version", default="4.15", help="version_tag (기본: 4.15)")
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--output-dir",
        default="tests/results",
        help="결과 JSON 저장 디렉토리 (기본: tests/results)",
    )
    args = parser.parse_args()

    session_id = args.session_id or f"test-multiturn-{uuid.uuid4().hex[:8]}"
    print(f"[session_id] {session_id}")
    print(f"[version_tag] {args.version}")
    print(f"[base_url] {args.base_url}")

    try:
        with urllib.request.urlopen(f"{args.base_url}/api/library", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        startup = health.get("startup_indexing", {})
        print(
            f"[startup] status={startup.get('status')} "
            f"files={startup.get('completed_files')}/{startup.get('total_files')} "
            f"pct={startup.get('progress_pct')}"
        )
    except Exception as exc:
        print(f"[warn] health check 실패: {exc}")

    results: list[dict] = []
    for idx, question in enumerate(TURNS_4_15, start=1):
        try:
            results.append(
                run_turn(args.base_url, session_id, idx, question, args.version)
            )
        except Exception as exc:
            print(f"\n[ERROR] turn {idx} 실패: {exc}")
            results.append({"turn": idx, "question": question, "error": str(exc)})

    summary = build_summary(results, args.version, session_id, args.base_url)
    print_summary_table(summary, results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"multiturn_{session_id}.json"
    payload = {"summary": summary, "turns": results}
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[결과 저장] {output_path}")

    checks = summary["spec_v301_checks"]
    all_pass = all(checks.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
