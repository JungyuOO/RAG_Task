from __future__ import annotations

import difflib
import hashlib
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_./+-]+")
_KO_CHAR_RANGE = re.compile(r"[가-힣]")
_KO_SUFFIXES = sorted(
    [
        "으로부터",
        "에서부터",
        "에게서",
        "까지는",
        "까지도",
        "으로는",
        "에서는",
        "과의",
        "으로",
        "로서",
        "로써",
        "로는",
        "로도",
        "에게",
        "에서",
        "처럼",
        "만큼",
        "보다",
        "까지",
        "부터",
        "에는",
        "에도",
        "께서",
        "과는",
        "과도",
        "라고",
        "라는",
        "이라",
        "이고",
        "이며",
        "이랑",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "과",
        "와",
        "로",
        "도",
        "만",
        "에",
        "의",
    ],
    key=len,
    reverse=True,
)
_TECH_TOKEN_WHITELIST = {"pv", "pvc", "rbac", "scc", "api", "cli", "yaml", "json", "oc", "ocp", "k8s"}
DOMAIN_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "Pod": ("pod", "파드", "팟", "포드"),
    "Deployment": ("deployment", "deploy", "디플로이먼트", "디플로이"),
    "Service": ("service", "서비스"),
    "Service Mesh": ("service mesh", "service-mesh", "servicemesh", "서비스 메시", "서비스메시", "서비스 메쉬", "서비스메쉬", "서비스 매쉬", "서비스매쉬"),
    "Ingress": ("ingress", "인그레스"),
    "Route": ("route", "라우트"),
    "Node": ("node", "노드"),
    "ConfigMap": ("configmap", "config map", "컨피그맵", "설정맵"),
    "Secret": ("secret", "시크릿", "시크렛"),
    "StatefulSet": ("statefulset", "stateful set", "스테이트풀셋", "스테이트풀 셋"),
    "DaemonSet": ("daemonset", "daemon set", "데몬셋", "데몬 셋"),
    "PV": ("pv", "persistent volume", "퍼시스턴트볼륨", "퍼시스턴트 볼륨", "피브이"),
    "PVC": ("pvc", "persistent volume claim", "퍼시스턴트볼륨클레임", "퍼시스턴트 볼륨 클레임", "피브이씨"),
    "OCP": ("ocp", "openshift", "open shift", "오픈시프트", "오씨피"),
    "Kubernetes": ("kubernetes", "k8s", "쿠버네티스", "케이8에스", "케이에잇에스"),
}


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _compact_domain_token(value: str) -> str:
    return re.sub(r"[\s_\-./]", "", value.casefold())


def _replace_exact_domain_aliases(text: str) -> str:
    normalized = text
    alias_pairs = [
        (alias, canonical)
        for canonical, aliases in DOMAIN_ALIAS_GROUPS.items()
        for alias in aliases
    ]
    alias_pairs.sort(key=lambda item: len(item[0]), reverse=True)
    for alias, canonical in alias_pairs:
        normalized = re.sub(re.escape(alias), canonical, normalized, flags=re.IGNORECASE)
    return normalized


def _replace_fuzzy_domain_aliases(text: str) -> str:
    words = text.split()
    if not words:
        return text

    alias_lookup = {
        _compact_domain_token(alias): canonical
        for canonical, aliases in DOMAIN_ALIAS_GROUPS.items()
        for alias in aliases
    }
    result: list[str] = []
    i = 0
    while i < len(words):
        matched = False
        for width in (3, 2, 1):
            if i + width > len(words):
                continue
            phrase = " ".join(words[i : i + width])
            compact = _compact_domain_token(phrase)
            if len(compact) < 4:
                continue
            best_alias = None
            best_score = 0.0
            for alias_compact in alias_lookup:
                score = difflib.SequenceMatcher(None, compact, alias_compact).ratio()
                if score > best_score:
                    best_score = score
                    best_alias = alias_compact
            threshold = 0.90 if width == 1 else 0.82
            if best_alias is not None and best_score >= threshold:
                result.append(alias_lookup[best_alias])
                i += width
                matched = True
                break
        if not matched:
            result.append(words[i])
            i += 1
    return " ".join(result)


def normalize_domain_terms(text: str) -> str:
    normalized = normalize_text(text)
    normalized = _replace_exact_domain_aliases(normalized)
    return _replace_fuzzy_domain_aliases(normalized)


def strip_korean_suffix(token: str) -> str:
    if not _KO_CHAR_RANGE.search(token):
        return token
    for suffix in _KO_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            stripped = token[: -len(suffix)]
            if len(stripped) >= 2:
                return stripped
    return token


def tokenize(text: str) -> list[str]:
    raw_tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [strip_korean_suffix(token) for token in raw_tokens]


def normalize_query_keywords(text: str, keywords: list[str] | None = None) -> list[str]:
    candidates: list[str] = []
    if keywords:
        for keyword in keywords:
            candidates.extend(TOKEN_PATTERN.findall(normalize_text(str(keyword))))
    if text:
        candidates.extend(TOKEN_PATTERN.findall(normalize_text(text)))

    normalized_keywords: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        lowered = strip_korean_suffix(token.lower())
        if not lowered:
            continue
        if len(lowered) < 2 and lowered not in _TECH_TOKEN_WHITELIST:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized_keywords.append(lowered)
    return normalized_keywords


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def keyword_overlap_score(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_counter = Counter(query_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap = sum(min(query_counter[token], candidate_counter[token]) for token in query_counter)
    return overlap / max(len(query_tokens), 1)


def extracted_markdown_file_name(source_path: Path) -> str:
    return f"{source_path.stem}-{stable_hash(str(source_path))[:8]}.md"


def extracted_markdown_path(extract_dir: Path, source_path: Path) -> Path:
    return extract_dir / extracted_markdown_file_name(source_path)


def extracted_markdown_candidates(extract_dir: Path, source_path: Path) -> list[Path]:
    stem = source_path.stem
    glob_matches = list(extract_dir.glob(f"{stem}-????????.md"))
    if glob_matches:
        return glob_matches

    candidates: list[Path] = []
    for candidate_source in (source_path, source_path.resolve()):
        candidate_path = extracted_markdown_path(extract_dir, candidate_source)
        if candidate_path not in candidates:
            candidates.append(candidate_path)
    return candidates
