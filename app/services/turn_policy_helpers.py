from __future__ import annotations


def contains_any_marker(normalized: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized for marker in markers)


def mentions_selected_source(normalized: str, topic_state: dict) -> bool:
    for source in topic_state.get("selected_sources", []):
        source_name = str(source).lower()
        if source_name.endswith(".pdf") and source_name in normalized:
            return True
    return False


def has_uppercase_document_token(normalized: str) -> bool:
    for token in normalized.split():
        if len(token) >= 2 and token.isupper() and token.isalpha():
            return True
    return False


def cited_pages(topic_state: dict) -> list[int]:
    pages: list[int] = []
    for item in topic_state.get("last_answer_citations", []) or []:
        value = item.get("page_number")
        if value is None:
            continue
        if str(value).isdigit():
            page = int(value)
            if page not in pages:
                pages.append(page)
    return pages


def competing_topics(topic_state: dict, normalize) -> list[str]:
    topics: list[str] = []
    for item in topic_state.get("recent_user_topics", []):
        normalized = normalize(str(item))
        if normalized and normalized not in topics:
            topics.append(normalized)
    return topics


def build_clarification_scope_hint(topic_state: dict, summary_topic: str, active_topic: str, normalize) -> str:
    active = active_topic or summary_topic
    topics = competing_topics(topic_state, normalize)
    pages = cited_pages(topic_state)
    page_hint = ", ".join(f"p.{page}" for page in pages[:3])
    topic_hint = ", ".join(topics[:3])
    return topic_hint or page_hint or (active or "recent topic")
