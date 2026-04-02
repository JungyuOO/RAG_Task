from __future__ import annotations

import json
import re

MISSING_CONTEXT_RESPONSE = (
    "업로드하신 문서에서 관련 내용을 찾을 수 없습니다. "
    "다른 질문을 해주시거나 관련 문서를 업로드해 주세요."
)

KOREAN_ONLY_INSTRUCTION = (
    "반드시 한국어로 답변하라. "
    "사용자가 어떤 언어로 질문하더라도 항상 한국어로만 답변하라."
)

CODE_EXAMPLE_INSTRUCTION = (
    " The user is asking for code/YAML examples. "
    "ONLY include code blocks that are directly relevant to the user's specific question. "
    "Do NOT include unrelated code from the same page or nearby sections. "
    "preserve resource names, field names, values, and command syntax exactly as written in the source. "
    "Do not invent alternate example names, values, or commands. "
    "Format code blocks with proper ```yaml or ```bash fencing. "
    "Briefly explain what each code block does before showing it."
)


class PromptComposer:
    def __init__(self, session_repository, settings) -> None:
        self.session_repository = session_repository
        self.settings = settings

    @staticmethod
    def topic_to_topic_state(topic: dict | None) -> dict:
        if not topic:
            return {}
        summary = topic.get("summary", {})
        return {
            "active_topic": topic.get("topic_label") or summary.get("topic_label") or "",
            "active_entities": topic.get("entities", [])[:6],
            "selected_sources": topic.get("sources", [])[:3],
            "selected_pages": summary.get("important_pages", [])[:5],
            "last_retrieval_mode": topic.get("last_retrieval_mode", ""),
            "last_answer_citations": [],
            "last_user_focus": topic.get("last_user_focus", ""),
            "recent_user_topics": [topic.get("topic_label") or summary.get("topic_label") or ""],
            "last_explicit_resource": summary.get("last_explicit_resource", ""),
            "last_explicit_resources": summary.get("last_explicit_resources", [])[:4],
            "last_intent": summary.get("last_intent", ""),
            "last_response_shape": summary.get("last_response_shape", ""),
            "last_answer_route": summary.get("last_answer_route", ""),
            "last_format_constraints": summary.get("last_format_constraints", [])[:4],
            "last_code_resource_kind": summary.get("last_code_resource_kind", ""),
            "last_grounded_chunk_ids": summary.get("last_grounded_chunk_ids", [])[:6],
            "last_grounded_section_paths": summary.get("last_grounded_section_paths", [])[:4],
            "last_example_source_pages": summary.get("last_example_source_pages", [])[:6],
            "last_example_anchor": summary.get("last_example_anchor", {}),
        }

    def build_rewrite_context_from_topic(self, topic: dict | None, topic_turns: list) -> dict | None:
        if not topic:
            return None
        topic_state = self.topic_to_topic_state(topic)
        conversation_history: list[dict] = []
        last_assistant_turn = None
        for turn in topic_turns[-4:]:
            entry = {"role": turn.role, "content": str(turn.content)[:200]}
            if turn.role == "assistant" and turn.metadata:
                sources = [
                    str(item.get("file_name", ""))
                    for item in turn.metadata.get("source_grounding", [])[:2]
                    if item.get("file_name")
                ]
                if sources:
                    entry["sources"] = sources
                last_assistant_turn = turn
            conversation_history.append(entry)

        last_response_shape = ""
        last_response_intent = ""
        if last_assistant_turn and last_assistant_turn.metadata:
            qi = last_assistant_turn.metadata.get("query_interpretation") or {}
            last_response_shape = str(qi.get("response_shape") or "")
            last_response_intent = str(qi.get("intent") or "")

        return {
            "conversation_history": conversation_history,
            "active_topic": str(topic_state.get("active_topic") or ""),
            "active_entities": topic_state.get("active_entities", [])[:6],
            "selected_sources": topic_state.get("selected_sources", [])[:3],
            "selected_pages": topic_state.get("selected_pages", [])[:5],
            "last_retrieval_mode": str(topic_state.get("last_retrieval_mode") or ""),
            "last_response_shape": last_response_shape,
            "last_response_intent": last_response_intent,
            "last_explicit_resources": topic_state.get("last_explicit_resources", [])[:4],
            "last_code_resource_kind": str(topic_state.get("last_code_resource_kind") or ""),
            "last_example_anchor": topic_state.get("last_example_anchor", {}),
        }

    def build_prompt_memory_snapshot(self, session_id: str, topic_id: str | None = None) -> dict:
        snapshot = self.session_repository.memory_snapshot(session_id)
        summary = snapshot.get("session_summary", {})
        topic_state = snapshot.get("topic_state", {})
        recent_turns = snapshot.get("recent_turns", [])
        if topic_id:
            topic = self.session_repository.get_topic(topic_id)
            if topic is not None:
                topic_summary = self.session_repository.topic_memory_snapshot(session_id, topic_id)
                topic_state = self.topic_to_topic_state(topic)
                summary = {
                    "topic": topic_summary.get("topic_label", ""),
                    "user_goal": topic_summary.get("last_user_focus", ""),
                    "recent_documents": topic_summary.get("sources", [])[:3],
                    "recent_pages": topic_summary.get("important_pages", [])[:4],
                }
                recent_turns = [turn.to_dict() for turn in self.session_repository.recent_topic_turns(session_id, topic_id)]
        prompt_recent_turns = max(int(self.settings.llm_prompt_recent_turns), 1)
        compact_recent_turns = [
            {
                "role": turn.get("role", ""),
                "content": str(turn.get("content", ""))[:180],
            }
            for turn in recent_turns[-prompt_recent_turns:]
        ]
        return {
            "topic": summary.get("topic", ""),
            "user_goal": str(summary.get("user_goal", ""))[:180],
            "recent_documents": summary.get("recent_documents", [])[:3],
            "recent_pages": summary.get("recent_pages", [])[:4],
            "active_topic": topic_state.get("active_topic", ""),
            "selected_sources": topic_state.get("selected_sources", [])[:3],
            "selected_pages": topic_state.get("selected_pages", [])[:4],
            "last_retrieval_mode": topic_state.get("last_retrieval_mode", ""),
            "last_explicit_resource": topic_state.get("last_explicit_resource", ""),
            "last_explicit_resources": topic_state.get("last_explicit_resources", [])[:4],
            "last_intent": topic_state.get("last_intent", ""),
            "last_response_shape": topic_state.get("last_response_shape", ""),
            "last_answer_route": topic_state.get("last_answer_route", ""),
            "last_format_constraints": topic_state.get("last_format_constraints", [])[:4],
            "last_code_resource_kind": topic_state.get("last_code_resource_kind", ""),
            "recent_turns": compact_recent_turns,
        }

    def build_prompt_recent_turns(self, session_id: str, topic_id: str | None = None) -> list[dict]:
        prompt_recent_turns = max(int(self.settings.llm_prompt_recent_turns), 1)
        if topic_id:
            recent_turns = self.session_repository.recent_topic_turns(session_id, topic_id)[-prompt_recent_turns:]
        else:
            recent_turns = self.session_repository.recent_turns(session_id)[-prompt_recent_turns:]
        return [
            {
                "role": turn.role,
                "content": str(turn.content)[:500],
            }
            for turn in recent_turns
        ]

    def build_prompt_recent_turns_clean(self, session_id: str, topic_id: str | None = None) -> list[dict]:
        citation_pattern = re.compile(r"\[[^\]]+\.(?:pdf|PDF)[^\]]*\][^\n]*")
        turns = self.build_prompt_recent_turns(session_id, topic_id=topic_id)
        cleaned = []
        for turn in turns:
            if turn["role"] == "assistant":
                content = citation_pattern.sub("", turn["content"]).strip()
                cleaned.append({**turn, "content": content})
            else:
                cleaned.append(turn)
        return cleaned

    def build_prompt_context_text(self, context_blocks: list[str]) -> str:
        max_items = max(int(self.settings.llm_prompt_context_items), 1)
        char_limit = max(int(self.settings.llm_prompt_context_char_limit), 600)
        selected_blocks = context_blocks[:max_items]
        parts: list[str] = []
        used = 0
        for block in selected_blocks:
            remaining = char_limit - used
            if remaining <= 0:
                break
            trimmed = block[:remaining]
            parts.append(trimmed)
            used += len(trimmed)
        return "\n\n".join(parts) if parts else "No reliable retrieved context."

    def build_system_prompt(
        self,
        code_example_request: bool,
        query_interpretation: dict | None = None,
    ) -> str:
        system_prompt = (
            "You are a document-grounded RAG assistant. "
            "You ONLY answer questions based on the retrieved document context provided below. "
            "If retrieved context is provided, answer from that context and mention source file names and page numbers when possible. "
            "If retrieved context is weak, missing, or irrelevant to the user's question, do NOT answer the question. "
            f"Instead, respond with: '{MISSING_CONTEXT_RESPONSE}' "
            "Do NOT answer general knowledge questions, trivia, or anything not grounded in the retrieved context. "
            "If the user is simply reacting, acknowledging, or thanking you after a document-grounded answer, respond conversationally without reusing document citations. "
            "Do not mention unrelated prior questions or prior document topics unless the current user message explicitly asks for them. "
            "When retrieved context is used, end the answer with a short source line such as '[file.pdf] p.5' or '[file.pdf] p.5-6'. "
            "Keep answers concise but grounded. "
            f"{KOREAN_ONLY_INSTRUCTION}"
        )
        if code_example_request:
            system_prompt += CODE_EXAMPLE_INSTRUCTION

        # single-resource explain 시 focus 지시
        qi = query_interpretation or {}
        resources = qi.get("resources", [])
        intent = qi.get("intent", "")
        response_shape = qi.get("response_shape", "")
        if len(resources) == 1 and (intent == "explain" or response_shape == "text"):
            target = resources[0]
            system_prompt += (
                f" [Focus Constraint] The user is asking specifically about '{target}'. "
                f"Answer ONLY about '{target}'. "
                "Even if the retrieved context contains information about other related resources, "
                "do NOT include comparisons or explanations of other resources unless the user explicitly asked for them. "
                "Stay focused on the requested resource."
            )

        return system_prompt

    def build_llm_messages(
        self,
        session_id: str,
        user_message: str,
        code_example_request: bool,
        response_mode: str,
        turn_policy: dict,
        top_score: float,
        context_blocks: list[str],
        is_new_topic: bool = False,
        topic_id: str | None = None,
        query_interpretation: dict | None = None,
    ) -> list[dict]:
        system_prompt = self.build_system_prompt(code_example_request, query_interpretation=query_interpretation)
        summary = self.session_repository.summary(session_id)
        prompt_memory = self.build_prompt_memory_snapshot(session_id, topic_id=topic_id)
        recent_turns = (
            self.build_prompt_recent_turns_clean(session_id, topic_id=topic_id)
            if is_new_topic
            else self.build_prompt_recent_turns(session_id, topic_id=topic_id)
        )
        context_text = self.build_prompt_context_text(context_blocks)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"Conversation summary:\n{summary or 'No summary yet.'}\n\n"
                    f"Session memory:\n{json.dumps(prompt_memory, ensure_ascii=False)}\n\n"
                    f"Retrieval mode: {response_mode}\n"
                    f"Turn policy: {json.dumps(turn_policy, ensure_ascii=False)}\n"
                    f"Top retrieval score: {top_score:.4f}\n\n"
                    f"Retrieved context:\n{context_text}"
                ),
            },
            *recent_turns,
            {"role": "user", "content": user_message},
        ]
