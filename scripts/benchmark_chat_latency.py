from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid


def run_once(base_url: str, message: str) -> float:
    payload = json.dumps({"session_id": f"bench-{uuid.uuid4().hex[:8]}", "message": message}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Id": "latency-bench",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        response.read()
    return round(time.perf_counter() - started, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark /api/chat latency for selected questions.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    questions = [
        "pod 확인하는 명령어 뭐야?",
        "namespace 확인 명령어 뭐야?",
        "yaml 보려면 무슨 명령어 써?",
        "지금 내 pandas 보려면 어떤 명령어 쳐야 돼?",
        "현재 상태 확인 명령어랑 실제 결과 같이 알려줘",
    ]

    for question in questions:
        samples = [run_once(args.base_url, question) for _ in range(max(args.repeats, 1))]
        print(
            json.dumps(
                {
                    "question": question,
                    "samples_sec": samples,
                    "avg_sec": round(statistics.mean(samples), 2),
                    "max_sec": max(samples),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
