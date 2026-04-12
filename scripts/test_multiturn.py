"""
멀티턴 10개 대화 테스트 + 답변 정당성 체크
BM25 fix 이후 retrieval 품질 확인용
"""
from __future__ import annotations

import json
import uuid
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"
SESSION_ID = f"test-{uuid.uuid4().hex[:8]}"
VERSION_TAG = "4.15"

# OCP 문서에서 실제로 답할 수 있어야 하는 멀티턴 10개 질문
# 자연스러운 대화 흐름으로 구성 (이전 답변에 이어지는 질문 포함)
QUESTIONS = [
    "OpenShift에서 Pod란 무엇인가요?",
    "그럼 Pod를 생성하는 YAML 예시 보여줘",
    "방금 보여준 YAML에서 imagePullPolicy는 어떤 값들이 있어?",
    "Deployment와 Pod의 차이는 뭐야?",
    "Deployment에서 롤링 업데이트는 어떻게 설정해?",
    "PersistentVolumeClaim이 뭔지 설명해줘",
    "PVC를 Pod에 마운트하는 방법은?",
    "OpenShift에서 Route란 무엇이고 Service와 어떻게 다른가요?",
    "TLS 종료 방식에는 어떤 종류가 있어?",
    "지금까지 얘기한 것들 중에서 가장 중요한 개념 3가지만 정리해줘",
]

def send_chat(message: str, session_id: str, version_tag: str) -> dict:
    """SSE 스트림을 수신해서 answer와 context를 파싱한다."""
    payload = json.dumps({
        "message": message,
        "session_id": session_id,
        "version_tag": version_tag,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Client-Id": "test-client",
        },
        method="POST",
    )

    tokens = []
    context_event = None
    done_event = None

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            buf = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buf += chunk
                # 버퍼에서 완성된 라인 파싱
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "token":
                        tokens.append(event.get("text", ""))
                    elif etype == "context":
                        context_event = event
                    elif etype == "done":
                        done_event = event
    except urllib.error.URLError as e:
        return {"error": str(e), "answer": "", "sources": []}

    answer = "".join(tokens).strip()
    sources = []
    if context_event:
        # source_grounding 우선, 없으면 items 사용
        for sg in context_event.get("source_grounding", []):
            sources.append({"file": sg.get("file_name", ""), "score": sg.get("score", 0)})
        if not sources:
            for item in context_event.get("items", []):
                fname = item.get("source_path", "").split("/")[-1]
                sources.append({"file": fname, "score": item.get("rerank_score", 0)})

    return {
        "answer": answer,
        "sources": sources,
        "cached": done_event.get("cached", False) if done_event else False,
    }


def evaluate(q: str, result: dict, turn: int) -> dict:
    """간단한 정당성 체크."""
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    issues = []

    # 1. 답변이 있는가
    if len(answer) < 30:
        issues.append("답변이 너무 짧음")

    # 2. 소스가 검색됐는가
    if not sources:
        issues.append("소스 없음 (retrieval 실패 가능성)")

    # 3. 소스가 OCP 문서인가 (연관 없는 소스 탐지)
    for s in sources:
        fname = s.get("file", "").lower()
        if fname and "openshift" not in fname and "ocp" not in fname:
            issues.append(f"비OCP 소스 감지: {s['file']}")

    # 4. 답변에 에러 메시지가 없는가
    if "error" in result:
        issues.append(f"API 에러: {result['error']}")

    # 5. 이전 질문 참조 체크 (turn >= 3이고 "방금", "그", "이" 등 지시어 포함 시)
    pronoun_hints = ["방금", "그 yaml", "그럼", "위에서", "아까"]
    if turn >= 2 and any(h in q for h in pronoun_hints):
        if len(answer) < 50:
            issues.append("멀티턴 지시어 있는데 답변 부실")

    status = "PASS" if not issues else "FAIL"
    return {"status": status, "issues": issues}


def main():
    print(f"=== 멀티턴 테스트 | session={SESSION_ID} | version={VERSION_TAG} ===\n")

    results_summary = []

    for turn, question in enumerate(QUESTIONS, 1):
        print(f"[턴 {turn:02d}] Q: {question}")
        result = send_chat(question, SESSION_ID, VERSION_TAG)
        eval_result = evaluate(question, result, turn)

        answer_preview = result.get("answer", "")[:120].replace("\n", " ")
        source_names = [s["file"] for s in result.get("sources", [])][:3]

        print(f"       A: {answer_preview}{'...' if len(result.get('answer','')) > 120 else ''}")
        print(f"       Sources: {source_names}")
        print(f"       [{eval_result['status']}]", end="")
        if eval_result["issues"]:
            print(f" !! {', '.join(eval_result['issues'])}")
        else:
            print()
        print()

        results_summary.append({
            "turn": turn,
            "question": question,
            "status": eval_result["status"],
            "issues": eval_result["issues"],
            "answer_len": len(result.get("answer", "")),
            "source_count": len(result.get("sources", [])),
            "sources": source_names,
        })

    # 최종 요약
    total = len(results_summary)
    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    print("=" * 60)
    print(f"최종 결과: {passed}/{total} PASS")
    print()

    failed = [r for r in results_summary if r["status"] == "FAIL"]
    if failed:
        print("실패 턴:")
        for r in failed:
            print(f"  턴 {r['turn']}: {r['question'][:40]}...")
            for issue in r["issues"]:
                print(f"    - {issue}")
    else:
        print("모든 턴 PASS!")


if __name__ == "__main__":
    main()
