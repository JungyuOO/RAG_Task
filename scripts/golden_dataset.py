from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ALLOWED_GROUPS = {"official_ocp", "customer_generated", "mixed"}
ALLOWED_SHAPES = {"summary", "checklist", "comparison", "procedure", "status_analysis"}


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    question: str
    version_tag: str
    expected_group: str
    expected_answer_shape: str
    expected_source_tokens: tuple[str, ...]
    required_keywords: tuple[str, ...]
    forbidden_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldenDataset:
    version: str
    title: str
    cases: tuple[GoldenCase, ...]


def _require_non_empty_string(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _require_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if not items:
        raise ValueError(f"{field_name} must contain at least one item")
    return items


def _parse_case(payload: dict) -> GoldenCase:
    expected_group = _require_non_empty_string(payload.get("expected_group"), "expected_group")
    if expected_group not in ALLOWED_GROUPS:
        raise ValueError(f"unsupported expected_group: {expected_group}")

    expected_answer_shape = _require_non_empty_string(payload.get("expected_answer_shape"), "expected_answer_shape")
    if expected_answer_shape not in ALLOWED_SHAPES:
        raise ValueError(f"unsupported expected_answer_shape: {expected_answer_shape}")

    return GoldenCase(
        id=_require_non_empty_string(payload.get("id"), "id"),
        question=_require_non_empty_string(payload.get("question"), "question"),
        version_tag=_require_non_empty_string(payload.get("version_tag"), "version_tag"),
        expected_group=expected_group,
        expected_answer_shape=expected_answer_shape,
        expected_source_tokens=_require_string_list(payload.get("expected_source_tokens"), "expected_source_tokens"),
        required_keywords=_require_string_list(payload.get("required_keywords"), "required_keywords"),
        forbidden_markers=_require_string_list(payload.get("forbidden_markers"), "forbidden_markers"),
    )


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    version = _require_non_empty_string(payload.get("version"), "version")
    title = _require_non_empty_string(payload.get("title"), "title")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must contain at least one case")

    cases = tuple(_parse_case(case_payload) for case_payload in raw_cases)
    return GoldenDataset(version=version, title=title, cases=cases)


def summarize_golden_dataset(dataset: GoldenDataset) -> dict[str, object]:
    groups = sorted({case.expected_group for case in dataset.cases})
    shapes = sorted({case.expected_answer_shape for case in dataset.cases})
    return {
        "version": dataset.version,
        "title": dataset.title,
        "case_count": len(dataset.cases),
        "groups": groups,
        "answer_shapes": shapes,
    }
