"""Session state defaults and pure derivation helpers."""

from __future__ import annotations

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text

DEFAULT_SUMMARY = {
    "topic": "",
    "user_goal": "",
    "recent_documents": [],
    "recent_pages": [],
    "unresolved_questions": [],
    "last_user_message": "",
}

DEFAULT_TOPIC_STATE = {
    "active_topic": "",
    "active_document_group": "auto",
    "active_entities": [],
    "selected_sources": [],
    "selected_versions": [],
    "selected_pages": [],
    "last_retrieval_mode": "",
    "last_answer_citations": [],
    "last_user_focus": "",
    "recent_user_topics": [],
    "last_explicit_resource": "",
    "last_explicit_resources": [],
    "last_intent": "",
    "last_response_shape": "",
    "last_answer_route": "",
    "last_format_constraints": [],
    "last_code_resource_kind": "",
    "last_grounded_chunk_ids": [],
    "last_grounded_section_paths": [],
    "last_example_source_pages": [],
    "last_example_anchor": {},
    "last_doc_type": "",
    "last_document_group_preference": "auto",
    "procedure_state": {},
}

DEFAULT_TOPIC_THREAD_SUMMARY = {
    "topic_label": "",
    "summary": "",
    "entities": [],
    "sources": [],
    "selected_versions": [],
    "important_pages": [],
    "open_questions": [],
    "resolved_facts": [],
    "last_user_focus": "",
    "last_retrieval_mode": "",
    "last_explicit_resource": "",
    "last_explicit_resources": [],
    "last_intent": "",
    "last_response_shape": "",
    "last_answer_route": "",
    "last_format_constraints": [],
    "last_code_resource_kind": "",
    "last_grounded_chunk_ids": [],
    "last_grounded_section_paths": [],
    "last_example_source_pages": [],
    "last_example_anchor": {},
    "last_doc_type": "",
    "last_document_group_preference": "auto",
    "turn_count": 0,
}

GENERIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "may", "might", "shall", "should",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "my", "your", "his", "its", "our", "their",
    "what", "when", "where", "which", "who", "whom", "how", "why",
    "that", "this", "these", "those", "there", "here",
    "and", "or", "but", "if", "then", "so", "because", "as", "than",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "not", "no", "yes", "all", "some", "any", "each", "every",
    "more", "most", "very", "too", "also", "just", "only",
    "again", "above", "below", "up", "down", "out", "off",
    "explain", "show", "tell", "give", "get", "make", "let",
}


def extract_entities(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = [token.strip(".,:;()[]{}!?") for token in normalized.split(" ")]
    entities: list[str] = []

    for token in tokens:
        if len(token) < 2:
            continue
        lowered = token.lower()
        if lowered in GENERIC_STOPWORDS:
            continue
        if lowered.endswith(".pdf"):
            entities.append(token)
            continue
        if any(char.isdigit() for char in token):
            entities.append(token)
            continue
        if token[0].isupper():
            entities.append(token)
            continue
        if any(char.isalpha() for char in token) and any(char.isupper() for char in token[1:]):
            entities.append(token)
            continue
        if any("\uac00" <= char <= "\ud7a3" for char in token):
            entities.append(token)
            continue
    return entities[:8]


def extract_focus_phrase(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    entities = extract_entities(normalized)
    if entities:
        return normalize_text(" ".join(entities[:3]))
    words = [
        word
        for word in normalized.split(" ")
        if len(word) >= 2 and word.lower() not in GENERIC_STOPWORDS
    ]
    return normalize_text(" ".join(words[:4]))


def extract_topic_from_turns(user_turns: list[ChatTurn]) -> str:
    for turn in reversed(user_turns):
        entities = extract_entities(turn.content)
        if entities:
            return entities[0]
        words = normalize_text(turn.content).split(" ")
        if words:
            return " ".join(words[:5])
    return ""


def build_structured_summary(turns: list[ChatTurn]) -> dict:
    recent_turns = turns[-10:]
    user_turns = [turn for turn in recent_turns if turn.role == "user"]
    assistant_turns = [turn for turn in recent_turns if turn.role == "assistant"]

    recent_documents: list[str] = []
    recent_pages: list[int] = []
    unresolved_questions: list[str] = []
    for turn in assistant_turns[-3:]:
        metadata = turn.metadata or {}
        for source in metadata.get("source_grounding", []):
            file_name = normalize_text(str(source.get("file_name") or ""))
            if file_name and file_name not in recent_documents:
                recent_documents.append(file_name)
        for page in metadata.get("grounded_pages", []):
            page_number = int(page.get("page_number") or 0)
            if page_number and page_number not in recent_pages:
                recent_pages.append(page_number)
        if metadata.get("mode") == "general" and turn.content:
            unresolved_questions.append(normalize_text(turn.content[:120]))

    last_user_message = normalize_text(user_turns[-1].content) if user_turns else ""
    user_goal = normalize_text(user_turns[-1].content) if user_turns else ""
    topic = recent_documents[0] if recent_documents else extract_topic_from_turns(user_turns)

    return {
        "topic": topic,
        "user_goal": user_goal,
        "recent_documents": recent_documents[:3],
        "recent_pages": recent_pages[:5],
        "unresolved_questions": unresolved_questions[:2],
        "last_user_message": last_user_message,
    }


def build_topic_state(turns: list[ChatTurn]) -> dict:
    recent_turns = turns[-8:]
    selected_sources: list[str] = []
    selected_pages: list[int] = []
    last_answer_citations: list[dict] = []
    last_retrieval_mode = ""
    active_entities: list[str] = []
    recent_user_topics: list[str] = []
    last_user_focus = ""
    procedure_state: dict = {}
    last_explicit_resource = ""
    last_explicit_resources: list[str] = []
    last_intent = ""
    last_response_shape = ""
    last_answer_route = ""
    last_format_constraints: list[str] = []
    last_code_resource_kind = ""
    last_grounded_chunk_ids: list[str] = []
    last_grounded_section_paths: list[str] = []
    last_example_source_pages: list[int] = []
    last_example_anchor: dict = {}
    last_doc_type = ""
    selected_versions: list[str] = []
    last_document_group_preference = "auto"

    for turn in recent_turns:
        metadata = turn.metadata or {}
        if turn.role == "assistant":
            last_retrieval_mode = str(metadata.get("mode") or last_retrieval_mode)
            query_interpretation = metadata.get("query_interpretation") or {}
            turn_versions = [
                normalize_text(str(value))
                for value in query_interpretation.get("target_versions", []) or []
                if value
            ]
            if turn_versions:
                selected_versions = []
                for version in turn_versions:
                    if version and version not in selected_versions:
                        selected_versions.append(version)
            group_preference = str(query_interpretation.get("document_group_preference") or "").strip()
            if group_preference:
                last_document_group_preference = group_preference
            resources = [
                normalize_text(str(value)).lower()
                for value in query_interpretation.get("resources", []) or []
                if value
            ]
            if resources:
                last_explicit_resources = []
                for resource in resources:
                    if resource and resource not in last_explicit_resources:
                        last_explicit_resources.append(resource)
                last_explicit_resource = last_explicit_resources[0]
            last_intent = str(query_interpretation.get("intent") or last_intent)
            last_response_shape = str(query_interpretation.get("response_shape") or last_response_shape)
            turn_doc_type = str(metadata.get("doc_type") or "")
            if turn_doc_type:
                last_doc_type = turn_doc_type
            last_answer_route = str(metadata.get("answer_route") or last_answer_route)
            formats = [
                normalize_text(str(value)).lower()
                for value in query_interpretation.get("format_constraints", []) or []
                if value
            ]
            if formats:
                last_format_constraints = []
                for fmt in formats:
                    if fmt and fmt not in last_format_constraints:
                        last_format_constraints.append(fmt)
            for item in metadata.get("source_grounding", []):
                file_name = normalize_text(str(item.get("file_name") or ""))
                if file_name and file_name not in selected_sources:
                    selected_sources.append(file_name)
            for item in metadata.get("preview_pages", []):
                page_number = int(item.get("page_number") or 0)
                if page_number and page_number not in selected_pages:
                    selected_pages.append(page_number)
                if page_number and page_number not in last_example_source_pages:
                    last_example_source_pages.append(page_number)
            if metadata.get("answer_citations"):
                last_answer_citations = metadata["answer_citations"][:4]
            if isinstance(metadata.get("procedure_state"), dict) and metadata.get("procedure_state", {}).get("steps"):
                procedure_state = metadata["procedure_state"]
            if isinstance(metadata.get("last_example_anchor"), dict) and metadata.get("last_example_anchor"):
                last_example_anchor = metadata["last_example_anchor"]
            track_code_resource_kind = (
                last_answer_route == "extractive_code"
                or last_response_shape.casefold() == "code"
                or bool(last_example_anchor)
            )
            for item in metadata.get("items", []) or []:
                chunk_id = normalize_text(str(item.get("chunk_id") or ""))
                section_path = normalize_text(str(item.get("section_path") or ""))
                page_number = int(item.get("page_number") or item.get("page_start") or 0)
                if chunk_id and chunk_id not in last_grounded_chunk_ids:
                    last_grounded_chunk_ids.append(chunk_id)
                if section_path and section_path not in last_grounded_section_paths:
                    last_grounded_section_paths.append(section_path)
                if page_number and page_number not in last_example_source_pages:
                    last_example_source_pages.append(page_number)
                if (
                    track_code_resource_kind
                    and not last_code_resource_kind
                    and str(item.get("code_subtype") or "") == "k8s_manifest"
                ):
                    code_signals = [
                        normalize_text(str(value)).lower()
                        for value in item.get("code_signals", []) or []
                        if value
                    ]
                    for signal in code_signals:
                        if signal in {
                            "configmap", "secret", "pod", "deployment", "service",
                            "persistentvolume", "persistentvolumeclaim", "route", "ingress", "storageclass",
                        }:
                            last_code_resource_kind = signal
                            break
            if not last_code_resource_kind and len(last_explicit_resources) == 1 and last_answer_route == "extractive_code":
                last_code_resource_kind = last_explicit_resources[0]
        else:
            active_entities.extend(extract_entities(turn.content))
            focus = extract_focus_phrase(turn.content)
            if focus:
                last_user_focus = focus
                if focus not in recent_user_topics:
                    recent_user_topics.append(focus)

    active_topic = selected_sources[0] if selected_sources else extract_topic_from_turns(
        [turn for turn in recent_turns if turn.role == "user"]
    )
    deduped_entities: list[str] = []
    for entity in active_entities:
        if entity and entity not in deduped_entities:
            deduped_entities.append(entity)

    return {
        "active_topic": active_topic,
        "active_document_group": last_document_group_preference or "auto",
        "active_entities": deduped_entities[:6],
        "selected_sources": selected_sources[:3],
        "selected_versions": selected_versions[:3],
        "selected_pages": selected_pages[:5],
        "last_retrieval_mode": last_retrieval_mode,
        "last_answer_citations": last_answer_citations,
        "last_user_focus": last_user_focus,
        "recent_user_topics": recent_user_topics[-4:],
        "last_explicit_resource": last_explicit_resource,
        "last_explicit_resources": last_explicit_resources[:4],
        "last_intent": last_intent,
        "last_response_shape": last_response_shape,
        "last_answer_route": last_answer_route,
        "last_format_constraints": last_format_constraints[:4],
        "last_code_resource_kind": last_code_resource_kind,
        "last_grounded_chunk_ids": last_grounded_chunk_ids[:6],
        "last_grounded_section_paths": last_grounded_section_paths[:4],
        "last_example_source_pages": last_example_source_pages[:6],
        "last_example_anchor": last_example_anchor,
        "last_doc_type": last_doc_type,
        "last_document_group_preference": last_document_group_preference or "auto",
        "procedure_state": procedure_state,
    }


def stringify_summary(summary_json: dict, topic_state: dict) -> str:
    parts: list[str] = []
    if summary_json.get("topic"):
        parts.append(f"Topic: {summary_json['topic']}")
    if summary_json.get("user_goal"):
        parts.append(f"Goal: {normalize_text(str(summary_json['user_goal']))[:160]}")
    if summary_json.get("recent_documents"):
        parts.append("Docs: " + ", ".join(summary_json["recent_documents"][:3]))
    if summary_json.get("recent_pages"):
        parts.append("Pages: " + ", ".join(str(page) for page in summary_json["recent_pages"][:5]))
    if topic_state.get("active_entities"):
        parts.append("Entities: " + ", ".join(topic_state["active_entities"][:4]))
    if summary_json.get("unresolved_questions"):
        parts.append("Open: " + " | ".join(summary_json["unresolved_questions"][:2]))
    return normalize_text(" ; ".join(parts)[:700])


def build_topic_thread_summary(topic_label: str, turns: list[ChatTurn]) -> dict:
    structured_summary = build_structured_summary(turns)
    topic_state = build_topic_state(turns)
    resolved_facts: list[str] = []
    for turn in turns:
        if turn.role != "assistant":
            continue
        normalized = normalize_text(turn.content)
        if normalized and normalized not in resolved_facts:
            resolved_facts.append(normalized[:180])
    summary_text = stringify_summary(structured_summary, topic_state)
    return {
        "topic_label": topic_label or structured_summary.get("topic") or "",
        "summary": summary_text,
        "entities": topic_state.get("active_entities", [])[:6],
        "sources": structured_summary.get("recent_documents", [])[:3],
        "selected_versions": topic_state.get("selected_versions", [])[:3],
        "important_pages": structured_summary.get("recent_pages", [])[:5],
        "open_questions": structured_summary.get("unresolved_questions", [])[:3],
        "resolved_facts": resolved_facts[:5],
        "last_user_focus": topic_state.get("last_user_focus", ""),
        "last_retrieval_mode": topic_state.get("last_retrieval_mode", ""),
        "last_explicit_resource": topic_state.get("last_explicit_resource", ""),
        "last_explicit_resources": topic_state.get("last_explicit_resources", [])[:4],
        "last_intent": topic_state.get("last_intent", ""),
        "last_response_shape": topic_state.get("last_response_shape", ""),
        "last_answer_route": topic_state.get("last_answer_route", ""),
        "last_format_constraints": topic_state.get("last_format_constraints", [])[:4],
        "last_code_resource_kind": topic_state.get("last_code_resource_kind", ""),
        "last_grounded_chunk_ids": topic_state.get("last_grounded_chunk_ids", [])[:6],
        "last_grounded_section_paths": topic_state.get("last_grounded_section_paths", [])[:4],
        "last_example_source_pages": topic_state.get("last_example_source_pages", [])[:6],
        "last_example_anchor": topic_state.get("last_example_anchor", {}),
        "last_doc_type": topic_state.get("last_doc_type", ""),
        "last_document_group_preference": topic_state.get("last_document_group_preference", "auto"),
        "turn_count": len(turns),
    }


def merge_defaults(default: dict, loaded: dict) -> dict:
    merged = default.copy()
    merged.update(loaded or {})
    return merged


def deserialize_topic_row(row: dict) -> dict:
    summary = merge_defaults(
        DEFAULT_TOPIC_THREAD_SUMMARY,
        __import__("json").loads(row["summary_json"] or "{}"),
    )
    source_state = __import__("json").loads(row["source_state_json"] or "{}")
    entity_state = __import__("json").loads(row["entity_state_json"] or "{}")
    open_questions = __import__("json").loads(row["open_questions_json"] or "[]")
    resolved_facts = __import__("json").loads(row["resolved_facts_json"] or "[]")
    if source_state.get("sources"):
        summary["sources"] = [str(source) for source in source_state["sources"] if source][:3]
    if entity_state.get("entities"):
        summary["entities"] = [str(entity) for entity in entity_state["entities"] if entity][:6]
    summary["open_questions"] = [str(item) for item in open_questions if item][:3]
    summary["resolved_facts"] = [str(item) for item in resolved_facts if item][:5]
    summary["selected_versions"] = [str(item) for item in summary.get("selected_versions", []) if item][:3]
    summary["last_document_group_preference"] = str(summary.get("last_document_group_preference") or "auto")
    summary["last_user_focus"] = str(row["last_user_focus"] or summary.get("last_user_focus") or "")
    summary["last_retrieval_mode"] = str(row["last_retrieval_mode"] or summary.get("last_retrieval_mode") or "")
    summary["turn_count"] = int(row["turn_count"] or summary.get("turn_count") or 0)
    return {
        "topic_id": row["topic_id"],
        "session_id": row["session_id"],
        "topic_label": row["topic_label"],
        "status": row["status"],
        "summary": summary,
        "sources": summary.get("sources", []),
        "entities": summary.get("entities", []),
        "open_questions": summary.get("open_questions", []),
        "resolved_facts": summary.get("resolved_facts", []),
        "last_user_focus": summary.get("last_user_focus", ""),
        "last_retrieval_mode": summary.get("last_retrieval_mode", ""),
        "turn_count": summary.get("turn_count", 0),
        "last_active_turn_id": row["last_active_turn_id"],
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def build_summary_bundle(turns: list[ChatTurn]) -> tuple[dict, dict, str]:
    summary_json = build_structured_summary(turns)
    topic_state = build_topic_state(turns)
    summary_text = stringify_summary(summary_json, topic_state)
    return summary_json, topic_state, summary_text


def build_rewrite_context_payload(recent: list[ChatTurn], summary: dict, topic_state: dict) -> dict | None:
    if not recent:
        return None

    conversation_history: list[dict] = []
    last_assistant_turn: ChatTurn | None = None
    for turn in recent[-4:]:
        entry: dict = {"role": turn.role, "content": turn.content[:200]}
        if turn.role == "assistant" and turn.metadata:
            sources = [
                source.get("file_name", "")
                for source in turn.metadata.get("source_grounding", [])[:2]
                if source.get("file_name")
            ]
            if sources:
                entry["sources"] = sources
            last_assistant_turn = turn
        conversation_history.append(entry)

    last_response_shape = ""
    last_response_intent = ""
    if last_assistant_turn and last_assistant_turn.metadata:
        query_interpretation = last_assistant_turn.metadata.get("query_interpretation") or {}
        last_response_shape = str(query_interpretation.get("response_shape") or "")
        last_response_intent = str(query_interpretation.get("intent") or "")

    return {
        "conversation_history": conversation_history,
        "active_topic": str(topic_state.get("active_topic") or summary.get("topic") or ""),
        "active_entities": [str(entity) for entity in topic_state.get("active_entities", []) if entity][:6],
        "selected_sources": [str(source) for source in topic_state.get("selected_sources", []) if source][:3],
        "selected_versions": [str(v) for v in topic_state.get("selected_versions", []) if v][:3],
        "selected_pages": topic_state.get("selected_pages", [])[:5],
        "last_retrieval_mode": str(topic_state.get("last_retrieval_mode") or ""),
        "last_response_shape": last_response_shape,
        "last_response_intent": last_response_intent,
        "last_explicit_resources": [str(value) for value in topic_state.get("last_explicit_resources", []) if value][:4],
        "last_code_resource_kind": str(topic_state.get("last_code_resource_kind") or ""),
        "last_example_anchor": topic_state.get("last_example_anchor", {}),
        "last_document_group_preference": topic_state.get("last_document_group_preference", "auto"),
        "last_doc_type": str(topic_state.get("last_doc_type") or ""),
    }
