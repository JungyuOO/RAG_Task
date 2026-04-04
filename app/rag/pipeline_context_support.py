from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from app.rag.context import TurnContextResolver
from app.rag.types import TurnPolicyDecision
from app.rag.utils import normalize_text, tokenize

logger = logging.getLogger("rag.pipeline")


class PipelineContextMixin:
    async def _resolve_turn_context(self, session_id: str, user_message: str) -> dict:
        recent_turns = self.session_repository.recent_turns(session_id)
        structured_summary = self.session_repository.structured_summary(session_id)
        session_topic_state = self.session_repository.topic_state(session_id)
        session_topics = self.session_repository.list_topics(session_id)
        current_topic_id = str(session_topic_state.get("last_active_topic_id") or "")
        resolver = getattr(self, "turn_context_resolver", TurnContextResolver())
        resolution = resolver.resolve(
            user_message=user_message,
            session_topics=session_topics,
            recent_turns=recent_turns,
            current_topic_id=current_topic_id or None,
        )
        resolved_topic = self.session_repository.get_topic(resolution.topic_id) if resolution.topic_id else None
        resolved_topic_state = self._topic_to_topic_state(resolved_topic)
        topic_state = resolved_topic_state or session_topic_state
        scoped_recent_turns = self.session_repository.recent_topic_turns(session_id, resolution.topic_id) if resolution.topic_id else recent_turns
        intent_result = await self.intent_agent.classify(
            user_message,
            context={
                "active_topic": topic_state.get("active_topic"),
                "selected_sources": topic_state.get("selected_sources", []),
                "procedure_state": topic_state.get("procedure_state", {}),
                "recent_turn_count": len(scoped_recent_turns),
                "summary_topic": structured_summary.get("topic", ""),
            },
        )
        policy = self._policy_from_intent(intent_result, resolution.resolution_type, bool(topic_state.get("active_topic") or topic_state.get("selected_sources")))
        if resolution.needs_clarification and resolution.clarification_prompt:
            policy = TurnPolicyDecision(
                turn_type="clarification",
                response_mode="clarification",
                use_retrieval=False,
                use_memory_rewrite=False,
                allow_preview=False,
                allow_citations=False,
                needs_clarification=True,
                clarification_reason="resolver_ambiguous_topic",
                clarification_prompt=resolution.clarification_prompt,
            )
        return {
            "recent_turns": recent_turns,
            "structured_summary": structured_summary,
            "session_topic_state": session_topic_state,
            "session_topics": session_topics,
            "resolution": resolution,
            "resolved_topic": resolved_topic,
            "topic_state": topic_state,
            "scoped_recent_turns": scoped_recent_turns,
            "policy": policy,
            "intent_result": intent_result,
        }

    def _policy_from_intent(self, intent_result: dict, resolution_type: str, has_prior_context: bool) -> TurnPolicyDecision:
        intent = str(intent_result.get("intent", "general") or "general").casefold()
        if intent == "greeting":
            return TurnPolicyDecision("greeting", "general", False, False, False, False)
        if intent == "unsupported_language":
            return TurnPolicyDecision("general_chat", "general", False, False, False, False)
        if intent == "step_navigation":
            return TurnPolicyDecision("conversational_ack", "conversational", False, False, False, False)
        if intent in {"rag", "clarification"}:
            use_memory_rewrite = resolution_type == "continue" or has_prior_context
            turn_type = "document_followup" if resolution_type == "continue" else "document_query"
            return TurnPolicyDecision(turn_type, "rag", True, use_memory_rewrite, True, True)
        return TurnPolicyDecision("general_chat", "general", False, False, False, False)

    def _build_non_retrieval_state(self, user_message: str, turn_context: dict) -> dict:
        policy: TurnPolicyDecision = turn_context["policy"]
        resolution = turn_context["resolution"]
        return {
            "rewritten_query": user_message.strip(),
            "top_score": 0.0,
            "use_retrieved_context": False,
            "grounded_pages": [],
            "ordered_context_items": [],
            "selected_context_items": [],
            "preferred_preview_source": None,
            "preview_pages": [],
            "response_mode": policy.response_mode,
            "turn_policy": policy.to_dict(),
            "turn_resolution": resolution.to_dict(),
            "resolved_topic_id": resolution.topic_id,
        }

    def _domain_guard_state(self, user_message: str, turn_context: dict) -> dict | None:
        policy: TurnPolicyDecision = turn_context["policy"]
        if policy.use_retrieval:
            return None
        logger.info("[InputGuard] type=%s mode=%s use_retrieval=%s", policy.turn_type, policy.response_mode, policy.use_retrieval)
        return self._build_non_retrieval_state(user_message, turn_context)

    def _should_skip_procedure_shortcut(self, user_message: str, session_topics: list[dict], current_topic_id: str | None) -> bool:
        normalized = normalize_text(user_message).lower()
        if not normalized:
            return False
        explicit_switch_markers = ("back to", "switch to", "다시", "아까", "이전", "말고")
        if not any(marker in normalized for marker in explicit_switch_markers):
            return False
        for topic in session_topics:
            topic_id = str(topic.get("topic_id") or "")
            if current_topic_id and topic_id == current_topic_id:
                continue
            label = normalize_text(str(topic.get("topic_label") or "")).lower()
            sources = [normalize_text(str(value)).lower() for value in topic.get("sources", []) if value]
            entities = [normalize_text(str(value)).lower() for value in topic.get("entities", []) if value]
            if label and label in normalized:
                return True
            if any(source and source in normalized for source in sources[:3]):
                return True
            if any(entity and entity in normalized for entity in entities[:6]):
                return True
        return False

    def _detect_procedure_followup(self, user_message: str, procedure_state: dict) -> dict | None:
        steps = procedure_state.get("steps", [])
        if not steps:
            return None
        normalized = normalize_text(user_message).lower()
        if not normalized:
            return None
        step_match = re.search(r"(\d+)\s*단계", normalized)
        if step_match:
            return {"type": "jump", "step_number": int(step_match.group(1))}
        step_match = re.search(r"\bstep\s*(\d+)\b", normalized)
        if step_match:
            return {"type": "jump", "step_number": int(step_match.group(1))}
        if any(marker in normalized for marker in ("단계별", "step by step", "순서대로", "절차")):
            return {"type": "outline"}
        if any(marker in normalized for marker in ("다음", "계속", "next", "continue")):
            return {"type": "next"}
        if any(marker in normalized for marker in ("처음부터", "1단계부터", "first step")):
            return {"type": "jump", "step_number": 1}
        return None

    def _looks_like_step_navigation_without_state(self, user_message: str, procedure_state: dict) -> bool:
        if procedure_state.get("steps"):
            return False
        normalized = normalize_text(user_message).lower()
        if not normalized:
            return False
        if re.search(r"(\d+)\s*단계", normalized):
            return True
        if re.search(r"\bstep\s*(\d+)\b", normalized):
            return True
        return any(marker in normalized for marker in ("다음 단계", "next step"))

    def _resolve_requested_resource_kinds(self, query_interpretation: dict | None) -> set[str]:
        query_interpretation = query_interpretation or {}
        requested_kinds: set[str] = set()
        for resource in query_interpretation.get("resources", []) or []:
            normalized = str(resource).casefold().strip()
            requested_kinds.update(self.RESOURCE_KIND_ALIASES.get(normalized, {normalized}))
        return requested_kinds

    def _infer_item_resource_kinds(self, item: dict) -> set[str]:
        metadata = item["chunk"].get("metadata", {})
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        explicit_kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$", lowered_text)
        inferred_kinds: set[str] = set()
        if explicit_kind_match:
            inferred_kinds.add(explicit_kind_match.group(1).casefold())
        for signal in metadata.get("code_signals", []) or []:
            normalized = str(signal).casefold().strip()
            for alias_set in self.RESOURCE_KIND_ALIASES.values():
                if normalized in alias_set:
                    inferred_kinds.update(alias_set)
        return inferred_kinds

    def _expand_query_with_resource_aliases(self, query: str, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        lowered_query = query.casefold()
        extra_tokens: list[str] = []
        for resource in query_interpretation.get("resources", []) or []:
            normalized = str(resource).casefold().strip()
            phrase_aliases = self.RESOURCE_QUERY_PHRASES.get(normalized, ())
            if any(phrase in lowered_query for phrase in phrase_aliases):
                continue
            for alias in sorted(self.RESOURCE_KIND_ALIASES.get(normalized, {normalized})):
                if alias and alias not in lowered_query:
                    extra_tokens.append(alias)
        if not extra_tokens:
            return query
        return f"{query} {' '.join(extra_tokens)}".strip()

    def _build_procedure_followup_answer(self, followup: dict, procedure_state: dict) -> tuple[str, dict] | None:
        steps = procedure_state.get("steps", [])
        if not steps:
            return None
        current_step = int(procedure_state.get("current_step") or 1)
        total_steps = int(procedure_state.get("total_steps") or len(steps))
        updated_state = {**procedure_state, "steps": steps, "total_steps": total_steps}
        if followup["type"] == "outline":
            lines = ["이전 답변 기준 단계별 정리입니다."]
            for step in steps:
                lines.append(f"{step['step_number']}. {step['title']}")
            updated_state["current_step"] = current_step
            return "\n".join(lines).strip(), updated_state
        requested_step = min(current_step + 1, total_steps) if followup["type"] == "next" else int(followup.get("step_number") or 1)
        matched = next((step for step in steps if int(step["step_number"]) == requested_step), None)
        if matched is None:
            return f"이전 답변 기준으로는 {requested_step}단계가 없습니다. 현재 정리된 단계는 1단계부터 {total_steps}단계까지입니다.", updated_state
        updated_state["current_step"] = requested_step
        parts = [f"{requested_step}단계: {matched['title']}"]
        if matched.get("body"):
            parts.append(matched["body"])
        return "\n\n".join(parts).strip(), updated_state

    def _prefer_block_type_items(self, items: list[dict], *, block_type: str, limit: int | None = None) -> list[dict]:
        if not items:
            return []
        preferred = [item for item in items if block_type in str(item["chunk"].get("metadata", {}).get("block_types", "")).split(",")]
        if not preferred:
            return []
        return preferred[:limit] if limit is not None else preferred

    def _heading_overlap_score(self, user_message: str, metadata: dict) -> float:
        query_tokens = {token for token in tokenize(user_message) if len(token) >= 2 and token not in {"yaml", "manifest", "code", "example", "sample", "demo"}}
        if not query_tokens:
            return 0.0
        section_title_tokens = set(tokenize(str(metadata.get("section_title", ""))))
        section_path_tokens = set(tokenize(str(metadata.get("section_path", ""))))
        parent_heading_tokens: set[str] = set()
        for heading in metadata.get("parent_headings", []) or []:
            parent_heading_tokens.update(tokenize(str(heading)))
        score = 0.0
        score += 0.2 * len(query_tokens & parent_heading_tokens)
        score += 0.5 * len(query_tokens & section_path_tokens)
        score += 0.8 * len(query_tokens & section_title_tokens)
        return score
