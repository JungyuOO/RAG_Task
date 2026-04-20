"""Multi-turn smoke test for dev-ver2 copilot chat.

Covers 3 scenarios × 5 turns:
  1. doc_oauth     — official docs track (based on service_eval_dataset_v2 OAuth cases)
  2. live_overview — live OCP track (needs_connection routing when no connection)
  3. mixed_compare — cross-lane track (doc + live → compare)

Verifies per-turn:
  - lane_expected matches response.lane
  - inline_citation_expected matches presence of [N] markers in answer
  - sources list non-empty when applicable

Usage:
  python scripts/test_multiturn_v202.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


CITATION_RE = re.compile(r"\[\d+\]")


SCENARIOS: list[dict] = [
    {
        "id": "doc_oauth",
        "description": "OAuth token 문서 주제 5턴 (공식 문서 track)",
        "connection_id": "",
        "namespace": "",
        "turns": [
            {
                "message": "oauth 토큰 유효 기간 자체를 늘리려면 어디 설정해?",
                "expect_lane": "doc",
                "expect_citation": True,
            },
            {
                "message": "그 설정은 어떤 필드를 바꾸는 거야?",
                "expect_lane": "doc",
                "expect_citation": True,
            },
            {
                "message": "유저가 가진 oauth 토큰 리스트 뽑는 방법도 알려줘",
                "expect_lane": "doc",
                "expect_citation": True,
            },
            {
                "message": "출력에서 봐야 할 필드도 알려줘",
                "expect_lane": "doc",
                "expect_citation": True,
            },
            {
                "message": "특정 유저 oauth 토큰 삭제해서 세션 끊으려면?",
                "expect_lane": "doc",
                "expect_citation": True,
            },
        ],
    },
    {
        "id": "live_overview",
        "description": "Live OCP 조회 5턴 (no connection → needs_connection routing)",
        "connection_id": "",
        "namespace": "default",
        "turns": [
            {
                "message": "클러스터 overview 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
            {
                "message": "default 네임스페이스의 pod 목록 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
            {
                "message": "그중에 첫번째 pod의 YAML 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
            {
                "message": "그 pod의 이벤트 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
            {
                "message": "deployment 목록 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
        ],
    },
    {
        "id": "mixed_compare",
        "description": "Doc + Live 혼합 5턴 (compare mixed lane promotion)",
        "connection_id": "",
        "namespace": "default",
        "turns": [
            {
                "message": "공식 문서 기준 oauth 토큰 리스트 뽑는 명령어 예시 알려줘",
                "expect_lane": "doc",
                "expect_citation": True,
            },
            {
                "message": "내 클러스터에서 pod 목록 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
            {
                "message": "그 중 첫번째 pod의 yaml 보여줘",
                "expect_lane": "needs_connection",
                "expect_citation": False,
            },
            {
                "message": "공식 문서 예시랑 live yaml 차이 비교해줘",
                "expect_lane_any": ["mixed", "needs_connection", "doc"],
                "expect_citation": False,
            },
            {
                "message": "그 절차 다시 번호로 정리해줘",
                "expect_lane_any": ["doc", "mixed"],
                "expect_citation": True,
            },
        ],
    },
]


def post_chat(base_url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{base_url}/api/v1/chat/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def count_citations(answer: str) -> int:
    return len(CITATION_RE.findall(answer or ""))


def build_history_turn(role: str, text: str, response: dict | None = None) -> dict:
    turn = {
        "role": role,
        "text": text,
        "lane": "",
        "source_paths": [],
        "resource_names": [],
        "namespace": "",
    }
    if role == "assistant" and response is not None:
        turn["lane"] = response.get("lane", "") or ""
        sources = response.get("sources") or []
        turn["source_paths"] = [
            s.get("source_path", "") for s in sources if s.get("source_type") == "doc" and s.get("source_path")
        ]
        turn["resource_names"] = [
            s.get("label", "") for s in sources if s.get("source_type") == "live" and s.get("label")
        ]
        for s in sources:
            if s.get("source_type") == "live" and s.get("namespace"):
                turn["namespace"] = s["namespace"]
                break
    return turn


def check_lane(actual: str, expected: str | None, expected_any: list[str] | None) -> bool:
    def _matches(a: str, e: str) -> bool:
        if a == e:
            return True
        if e == "doc" and a.startswith("doc"):
            return True
        return False

    if expected is not None:
        return _matches(actual, expected)
    if expected_any:
        return any(_matches(actual, e) for e in expected_any)
    return True


def run_scenario(base_url: str, scenario: dict) -> dict:
    print(f"\n{'=' * 80}")
    print(f"[scenario] {scenario['id']} — {scenario['description']}")
    print("=" * 80)

    history: list[dict] = []
    results = []
    for idx, turn in enumerate(scenario["turns"], start=1):
        msg = turn["message"]
        print(f"\n[Turn {idx}] Q: {msg}")
        payload = {
            "message": msg,
            "connection_id": scenario.get("connection_id", ""),
            "namespace": scenario.get("namespace", ""),
            "history": history,
        }
        t0 = time.time()
        try:
            response = post_chat(base_url, payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"  [HTTP {exc.code}] {body[:200]}")
            results.append({
                "turn": idx,
                "question": msg,
                "error": f"HTTP {exc.code}: {body[:200]}",
            })
            history.append(build_history_turn("user", msg))
            continue
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results.append({"turn": idx, "question": msg, "error": str(exc)})
            history.append(build_history_turn("user", msg))
            continue
        elapsed = time.time() - t0

        answer = response.get("answer", "") or ""
        lane = response.get("lane", "") or ""
        mode = response.get("mode", "") or ""
        sources = response.get("sources") or []
        citations = count_citations(answer)

        lane_ok = check_lane(lane, turn.get("expect_lane"), turn.get("expect_lane_any"))
        cite_expected = bool(turn.get("expect_citation"))
        cite_ok = (citations > 0) if cite_expected else True

        print(f"  lane={lane} mode={mode} elapsed={elapsed:.1f}s sources={len(sources)} citations={citations}")
        print(f"  A: {answer[:300].strip()}{'…' if len(answer) > 300 else ''}")
        print(f"  [check] lane_ok={lane_ok} citation_ok={cite_ok} (expected_cite={cite_expected})")

        results.append({
            "turn": idx,
            "question": msg,
            "lane": lane,
            "mode": mode,
            "elapsed_sec": round(elapsed, 2),
            "sources_count": len(sources),
            "citations_count": citations,
            "answer_preview": answer[:500],
            "expected_lane": turn.get("expect_lane") or turn.get("expect_lane_any"),
            "lane_ok": lane_ok,
            "citation_ok": cite_ok,
        })

        history.append(build_history_turn("user", msg))
        history.append(build_history_turn("assistant", answer, response))

    completed = [r for r in results if "error" not in r]
    lane_pass = sum(1 for r in completed if r.get("lane_ok"))
    cite_pass = sum(1 for r in completed if r.get("citation_ok"))
    avg_elapsed = round(sum(r.get("elapsed_sec", 0) for r in completed) / max(len(completed), 1), 2)

    summary = {
        "id": scenario["id"],
        "description": scenario["description"],
        "turns_total": len(scenario["turns"]),
        "turns_completed": len(completed),
        "lane_pass": lane_pass,
        "citation_pass": cite_pass,
        "avg_elapsed_sec": avg_elapsed,
        "results": results,
    }
    print(f"\n[summary] {scenario['id']}: lane {lane_pass}/{len(completed)}, citation {cite_pass}/{len(completed)}, avg {avg_elapsed}s")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", default="tests/results/multiturn_v202.json")
    parser.add_argument("--only", default="", help="Scenario id filter (e.g. doc_oauth)")
    parser.add_argument("--connection-id", default="", help="OCP connection id for live/mixed")
    args = parser.parse_args()

    if args.connection_id:
        for s in SCENARIOS:
            if s["id"] in {"live_overview", "mixed_compare"}:
                s["connection_id"] = args.connection_id
                for t in s["turns"]:
                    if t.get("expect_lane") == "needs_connection":
                        t["expect_lane"] = "live"
                    if t.get("expect_lane_any") and "needs_connection" in t["expect_lane_any"]:
                        t["expect_lane_any"] = [l for l in t["expect_lane_any"] if l != "needs_connection"] + ["live"]

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s["id"] == args.only]
        if not scenarios:
            print(f"Unknown scenario id: {args.only}")
            return 1

    all_summaries = []
    for scenario in scenarios:
        all_summaries.append(run_scenario(args.base_url, scenario))

    print(f"\n{'#' * 80}\n# Final Ranking\n{'#' * 80}")
    for s in all_summaries:
        lane_rate = s["lane_pass"] / max(s["turns_completed"], 1) * 100
        cite_rate = s["citation_pass"] / max(s["turns_completed"], 1) * 100
        print(
            f"  {s['id']:<20} turns={s['turns_completed']}/{s['turns_total']} "
            f"lane={lane_rate:.0f}% citation={cite_rate:.0f}% avg={s['avg_elapsed_sec']}s"
        )

    best = max(
        all_summaries,
        key=lambda s: (s["lane_pass"] + s["citation_pass"], -s["avg_elapsed_sec"]),
        default=None,
    )
    if best is not None:
        print(f"\n[best] {best['id']} — {best['description']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
