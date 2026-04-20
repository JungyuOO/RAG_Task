from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "data" / "chat_eval_dataset_v1.json"
RESULTS_DIR = ROOT / "tests" / "results" / "chat-eval"
API_BASE = "http://localhost:8000"
ENV_PATH = ROOT / ".env"


def api_get(path: str, timeout: float = 60, attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.get(f"{API_BASE}{path}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def api_post(path: str, payload: dict[str, Any], timeout: float = 120, attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def load_env() -> dict[str, str]:
    data: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def create_fresh_connection() -> str:
    env = load_env()
    payload = {
        "cluster_url": env["OCP_API_BASE_URL"],
        "auth_mode": "token",
        "verify_ssl": False,
        "default_namespace": env.get("OCP_DEFAULT_NAMESPACE", "demo"),
        "display_name": "chat-eval-run",
        "save_profile": False,
        "token": env["OCP_API_TOKEN"],
        "username": "",
        "password": None,
    }
    data = api_post("/api/v1/auth/ocp/connect", payload, timeout=60)
    connection_id = str(data["connection"]["connection_id"])
    api_post("/api/v1/auth/ocp/test", {"connection_id": connection_id}, timeout=120)
    return connection_id


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9가-힣_-]+", str(text or "").casefold())
        if len(token) >= 2
    }


def snippet_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / max(len(left_tokens), 1)


def fetch_preview_safe(source: dict[str, Any]) -> dict[str, Any] | None:
    if source.get("source_type") != "doc":
        return None
    source_path = source.get("source_path") or ""
    chunk_id = source.get("chunk_id") or ""
    if not source_path or not chunk_id:
        return None
    response = httpx.get(
        f"{API_BASE}/api/v1/docs-preview/snippet",
        params={"source_path": source_path, "chunk_id": chunk_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def evaluate_doc_turn(turn_spec: dict[str, Any], response: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {"citation_checks": []}
    answer = str(response.get("answer") or "")
    sources = response.get("sources") or []
    citation_map = response.get("citation_map") or []
    expect = turn_spec.get("expect") or {}

    required_keywords = [str(item) for item in expect.get("answer_keywords") or []]
    for keyword in required_keywords:
        if keyword.casefold() not in answer.casefold():
            reasons.append(f"missing_answer_keyword:{keyword}")

    any_keywords = [str(item) for item in expect.get("answer_any_keywords") or []]
    if any_keywords and not any(keyword.casefold() in answer.casefold() for keyword in any_keywords):
        reasons.append("missing_any_answer_keyword")

    min_citations = int(expect.get("min_doc_citations") or 0)
    doc_citations: list[dict[str, Any]] = []
    for item in citation_map:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("source_index")
        source_index = int(raw_index) if raw_index is not None else -1
        if 0 <= source_index < len(sources) and sources[source_index].get("source_type") == "doc":
            doc_citations.append(item)

    if min_citations and len(doc_citations) < min_citations:
        reasons.append(f"insufficient_citations:{len(doc_citations)}<{min_citations}")

    source_path_tokens = [str(item).casefold() for item in expect.get("source_path_tokens") or []]
    if source_path_tokens:
        doc_paths = " ".join(
            str(src.get("source_path") or "").casefold()
            for src in sources
            if src.get("source_type") == "doc"
        )
        if not any(token in doc_paths for token in source_path_tokens):
            reasons.append("source_path_token_miss")

    for citation in doc_citations:
        raw_index = citation.get("source_index")
        source_index = int(raw_index) if raw_index is not None else -1
        if source_index < 0 or source_index >= len(sources):
            reasons.append("citation_source_index_out_of_range")
            continue
        source = sources[source_index]
        preview = fetch_preview_safe(source)
        snippet = ""
        if preview is not None:
            snippet = str(preview.get("snippet") or "\n".join(preview.get("lines") or []))
        supporting_text = str(citation.get("supporting_text") or "")
        overlap = snippet_overlap(supporting_text, snippet)
        evidence["citation_checks"].append(
            {
                "citation_number": citation.get("citation_number"),
                "source_path": source.get("source_path"),
                "section_title": citation.get("section_title"),
                "supporting_text": supporting_text,
                "snippet_overlap": overlap,
            }
        )
        if supporting_text and overlap < 0.18:
            reasons.append(f"weak_citation_overlap:{citation.get('citation_number')}")

    return (not reasons), reasons, evidence


def evaluate_live_turn(turn_spec: dict[str, Any], response: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    answer = str(response.get("answer") or "")
    sources = response.get("sources") or []
    artifacts = response.get("artifacts") or []
    expect = turn_spec.get("expect") or {}

    expected_names = [str(item) for item in expect.get("live_resource_names") or []]
    if expected_names:
        live_labels = " ".join(str(src.get("label") or "") for src in sources if src.get("source_type") == "live")
        artifact_text = json.dumps(artifacts, ensure_ascii=False)
        if not any(name in live_labels or name in artifact_text or name in answer for name in expected_names):
            reasons.append("missing_expected_live_resource")

    answer_keywords = [str(item) for item in expect.get("answer_keywords") or []]
    for keyword in answer_keywords:
        if keyword.casefold() not in answer.casefold():
            reasons.append(f"missing_answer_keyword:{keyword}")

    any_keywords = [str(item) for item in expect.get("answer_any_keywords") or []]
    if any_keywords and not any(keyword.casefold() in answer.casefold() for keyword in any_keywords):
        reasons.append("missing_any_answer_keyword")

    artifact_types = [str(item) for item in expect.get("artifact_types") or []]
    if artifact_types:
        actual_types = {str(artifact.get("artifact_type") or "") for artifact in artifacts}
        evidence["artifact_types"] = sorted(actual_types)
        for artifact_type in artifact_types:
            if artifact_type not in actual_types:
                reasons.append(f"missing_artifact_type:{artifact_type}")

    if not any(src.get("source_type") == "live" for src in sources) and not artifacts:
        reasons.append("missing_live_evidence")

    return (not reasons), reasons, evidence


def run_yaml_apply_step(dataset: dict[str, Any], step: dict[str, Any], connection_id: str) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    namespace = str(step["target_namespace"])
    resource = str(step["target_resource"])
    name = str(step["target_name"])
    replicas_after = int(step["replicas_after"])

    try:
        detail = api_get(
            f"/api/v1/ocp/resource-detail/{connection_id}?resource={resource}&namespace={namespace}&name={name}"
        )
        manifest = detail["manifest_yaml"]
        current_replicas = int(detail.get("manifest_json", {}).get("spec", {}).get("replicas") or 0)

        preview = api_post(
            "/api/v1/actions/preview",
            {
                "connection_id": connection_id,
                "actor_id": "chat-eval-runner",
                "actor_roles": ["operator"],
                "action_type": "scale_deployment",
                "namespace": namespace,
                "resource_name": name,
                "replicas": replicas_after,
                "reason": "chat eval scenario replica update",
            },
        )
        if not preview.get("allowed"):
            reasons.append("scale_preview_blocked")
            return False, reasons, {"preview": preview, "current_replicas": current_replicas}

        request = api_post(
            "/api/v1/actions/requests",
            {
                "connection_id": connection_id,
                "actor_id": "chat-eval-runner",
                "actor_roles": ["operator"],
                "action_type": "scale_deployment",
                "namespace": namespace,
                "resource_name": name,
                "replicas": replicas_after,
                "reason": "chat eval scenario replica update",
            },
        )
        request_id = request["request_id"]
        if request["status"] != "approved":
            api_post(
                f"/api/v1/actions/requests/{request_id}/approve",
                {
                    "actor_id": "chat-eval-runner",
                    "actor_roles": ["operator"],
                    "decision_note": "chat eval approve",
                },
            )
        execution = api_post(
            f"/api/v1/actions/requests/{request_id}/execute",
            {
                "actor_id": "chat-eval-runner",
                "actor_roles": ["operator"],
                "execution_note": "chat eval execute",
                "force": False,
            },
        )
        if execution.get("status") != "succeeded":
            reasons.append("scale_execute_failed")

        updated = api_get(
            f"/api/v1/ocp/resource-detail/{connection_id}?resource={resource}&namespace={namespace}&name={name}"
        )
        replicas_now = int(updated.get("manifest_json", {}).get("spec", {}).get("replicas") or 0)
        if replicas_now != replicas_after:
            reasons.append(f"replica_not_updated:{replicas_now}!={replicas_after}")

        evidence = {
            "preview": preview,
            "request_id": request_id,
            "execution": execution,
            "replicas_before": current_replicas,
            "replicas_after": replicas_after,
            "replicas_now": replicas_now,
            "before_manifest": manifest[:4000],
            "restore_target": current_replicas,
            "resource": resource,
            "namespace": namespace,
            "name": name,
        }
        return (not reasons), reasons, evidence
    except Exception as exc:  # noqa: BLE001
        return False, [f"yaml_apply_exception:{exc}"], {}


def restore_scaled_resource(connection_id: str, evidence: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    namespace = str(evidence.get("namespace") or "")
    resource_name = str(evidence.get("name") or "")
    replicas_target = int(evidence.get("restore_target") or 0)
    try:
        preview = api_post(
            "/api/v1/actions/preview",
            {
                "connection_id": connection_id,
                "actor_id": "chat-eval-runner",
                "actor_roles": ["operator"],
                "action_type": "scale_deployment",
                "namespace": namespace,
                "resource_name": resource_name,
                "replicas": replicas_target,
                "reason": "chat eval restore original replica count",
            },
        )
        if not preview.get("allowed"):
            return False, {"preview": preview, "reason": "restore_preview_blocked"}

        request = api_post(
            "/api/v1/actions/requests",
            {
                "connection_id": connection_id,
                "actor_id": "chat-eval-runner",
                "actor_roles": ["operator"],
                "action_type": "scale_deployment",
                "namespace": namespace,
                "resource_name": resource_name,
                "replicas": replicas_target,
                "reason": "chat eval restore original replica count",
            },
        )
        request_id = request["request_id"]
        if request["status"] != "approved":
            api_post(
                f"/api/v1/actions/requests/{request_id}/approve",
                {
                    "actor_id": "chat-eval-runner",
                    "actor_roles": ["operator"],
                    "decision_note": "chat eval restore approve",
                },
            )
        execution = api_post(
            f"/api/v1/actions/requests/{request_id}/execute",
            {
                "actor_id": "chat-eval-runner",
                "actor_roles": ["operator"],
                "execution_note": "chat eval restore execute",
                "force": False,
            },
        )
        detail = api_get(
            f"/api/v1/ocp/resource-detail/{connection_id}?resource=deployments&namespace={namespace}&name={resource_name}"
        )
        replicas_now = int(detail.get("manifest_json", {}).get("spec", {}).get("replicas") or 0)
        return replicas_now == replicas_target, {
            "preview": preview,
            "request_id": request_id,
            "execution": execution,
            "replicas_target": replicas_target,
            "replicas_now": replicas_now,
        }
    except Exception as exc:  # noqa: BLE001
        return False, {"reason": f"restore_exception:{exc}"}


def evaluate_lane(expect: dict[str, Any], response: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    lane = str(response.get("lane") or "")
    lane_any = [str(item) for item in expect.get("lane_any") or []]
    if lane_any and not any(lane == expected or lane.startswith(expected) for expected in lane_any):
        reasons.append(f"lane_mismatch:{lane}")
    return (not reasons), reasons


def evaluate_mixed_turn(turn_spec: dict[str, Any], response: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    expect = turn_spec.get("expect") or {}
    reasons: list[str] = []
    evidence: dict[str, Any] = {}

    needs_doc = bool(expect.get("min_doc_citations") or expect.get("source_path_tokens"))
    needs_live = bool(expect.get("artifact_types") or expect.get("live_resource_names"))

    doc_ok, doc_reasons, doc_evidence = evaluate_doc_turn(turn_spec, response)
    live_ok, live_reasons, live_evidence = evaluate_live_turn(turn_spec, response)

    if needs_doc and not doc_ok:
        reasons.extend(doc_reasons)
    if needs_live and not live_ok:
        reasons.extend(live_reasons)
    if not needs_doc and not needs_live:
        if not (doc_ok or live_ok):
            reasons.extend(doc_reasons or live_reasons)

    evidence["doc"] = doc_evidence
    evidence["live"] = live_evidence
    return (not reasons), reasons, evidence


def run_dataset(dataset: dict[str, Any], *, only_ids: set[str] | None = None) -> dict[str, Any]:
    connection_id = create_fresh_connection()
    results: list[dict[str, Any]] = []

    scenarios = dataset["scenarios"]
    if only_ids:
        scenarios = [scenario for scenario in scenarios if scenario["id"] in only_ids]

    for scenario in scenarios:
        history: list[dict[str, Any]] = []
        restore_candidates: list[dict[str, Any]] = []
        scenario_result = {
            "id": scenario["id"],
            "category": scenario["category"],
            "description": scenario["description"],
            "steps": [],
            "passed": True,
        }

        for index, step in enumerate(scenario["turns"], start=1):
            started = time.time()
            if step["step_type"] == "chat":
                payload = {
                    "message": step["message"],
                    "connection_id": connection_id,
                    "namespace": dataset["live_selection"]["namespace"],
                    "history": history,
                }
                try:
                    response = api_post("/api/v1/chat/query", payload, timeout=180)
                    lane_ok, lane_reasons = evaluate_lane(step["expect"], response)
                    if scenario["category"] == "doc":
                        eval_ok, eval_reasons, evidence = evaluate_doc_turn(step, response)
                    elif scenario["category"] == "live":
                        eval_ok, eval_reasons, evidence = evaluate_live_turn(step, response)
                    else:
                        eval_ok, eval_reasons, evidence = evaluate_mixed_turn(step, response)
                    passed = lane_ok and eval_ok
                    reasons = [*lane_reasons, *eval_reasons]
                except Exception as exc:  # noqa: BLE001
                    response = {"error": str(exc)}
                    passed = False
                    reasons = [f"exception:{exc}"]
                    evidence = {}

                scenario_result["steps"].append(
                    {
                        "index": index,
                        "type": "chat",
                        "message": step["message"],
                        "passed": passed,
                        "reasons": reasons,
                        "elapsed_sec": round(time.time() - started, 2),
                        "response": response,
                        "evidence": evidence,
                    }
                )
                scenario_result["passed"] = scenario_result["passed"] and passed

                if "error" not in response:
                    history.append(
                        {
                            "role": "user",
                            "text": step["message"],
                            "lane": "",
                            "source_paths": [],
                            "resource_names": [],
                            "namespace": "",
                        }
                    )
                    history.append(
                        {
                            "role": "assistant",
                            "text": response.get("answer", ""),
                            "lane": response.get("lane", ""),
                            "source_paths": [
                                s.get("source_path", "")
                                for s in response.get("sources", [])
                                if s.get("source_type") == "doc"
                            ],
                            "resource_names": [
                                s.get("label", "")
                                for s in response.get("sources", [])
                                if s.get("source_type") == "live"
                            ],
                            "namespace": next(
                                (s.get("namespace", "") for s in response.get("sources", []) if s.get("namespace")),
                                "",
                            ),
                        }
                    )

            elif step["step_type"] == "yaml_apply":
                passed, reasons, evidence = run_yaml_apply_step(dataset, step, connection_id)
                if evidence and passed:
                    restore_candidates.append(evidence)
                scenario_result["steps"].append(
                    {
                        "index": index,
                        "type": "yaml_apply",
                        "target": {
                            "resource": step["target_resource"],
                            "namespace": step["target_namespace"],
                            "name": step["target_name"],
                        },
                        "passed": passed,
                        "reasons": reasons,
                        "elapsed_sec": round(time.time() - started, 2),
                        "evidence": evidence,
                    }
                )
                scenario_result["passed"] = scenario_result["passed"] and passed

        if restore_candidates:
            restore_evidence: list[dict[str, Any]] = []
            restore_failed = False
            for candidate in reversed(restore_candidates):
                ok, restore_info = restore_scaled_resource(connection_id, candidate)
                restore_evidence.append({"ok": ok, **restore_info})
                if not ok:
                    restore_failed = True
            scenario_result["post_restore"] = restore_evidence
            if restore_failed:
                scenario_result["passed"] = False

        results.append(scenario_result)

    passed = [item for item in results if item["passed"]]
    failed = [item for item in results if not item["passed"]]

    return {
        "dataset_version": dataset["version"],
        "generated_by": dataset.get("generated_by"),
        "connection_id": connection_id,
        "scenario_total": len(results),
        "scenario_passed": len(passed),
        "scenario_failed": len(failed),
        "results": results,
    }


def main() -> int:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    only_raw = str(os.environ.get("CHAT_EVAL_ONLY") or "").strip()
    only_ids = {item.strip() for item in only_raw.split(",") if item.strip()} or None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_dataset(dataset, only_ids=only_ids)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = RESULTS_DIR / f"chat-eval-{timestamp}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = RESULTS_DIR / "latest.json"
    latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {output_path}")
    print(
        json.dumps(
            {
                "scenario_total": result["scenario_total"],
                "scenario_passed": result["scenario_passed"],
                "scenario_failed": result["scenario_failed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
