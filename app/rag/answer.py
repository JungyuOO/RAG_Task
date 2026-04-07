from __future__ import annotations

import re
import logging
import time

logger = logging.getLogger("rag.answer")

from app.rag.answer_citation import AnswerCitationMixin
from app.rag.answer_format import AnswerFormatMixin
from app.rag.answer_inline_citation import InlineCitationMixin
from app.rag.retrieval_service import RetrievalService


class AnswerGenerator(AnswerFormatMixin, AnswerCitationMixin, InlineCitationMixin):
    """답변 포매팅, 인용 구성, preview payload 생성을 조합하는 서비스."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    @staticmethod
    def public_context_payload(payload: dict) -> dict:
        public_payload = dict(payload or {})
        public_payload.pop("_stored_procedure_state", None)
        return public_payload

    @staticmethod
    def strip_code_blocks_for_non_code_route(answer: str, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return answer
        return re.sub(r"```(?:[\w+-]+)?\n.*?```", "", answer, flags=re.DOTALL).strip()

    @staticmethod
    def build_example_anchor(context_items: list[dict], query_interpretation: dict | None) -> dict:
        query_interpretation = query_interpretation or {}
        fields: list[str] = []
        context_ids: list[str] = []
        page_numbers: list[int] = []
        section_paths: list[str] = []
        source_path = ""
        resources = [str(value) for value in query_interpretation.get("resources", []) if value]
        resource_kind = resources[0] if resources else ""
        for item in context_items[:3]:
            chunk = item["chunk"]
            if not source_path:
                source_path = str(chunk.get("source_path") or "")
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id not in context_ids:
                context_ids.append(chunk_id)
            page_number = int(chunk.get("page_number") or chunk.get("metadata", {}).get("page_start") or 0)
            if page_number and page_number not in page_numbers:
                page_numbers.append(page_number)
            section_path = str(chunk.get("metadata", {}).get("section_path", "") or "")
            if section_path and section_path not in section_paths:
                section_paths.append(section_path)
            for field in re.findall(r"(?im)^\s*([a-z][a-z0-9_-]*)\s*:", str(chunk.get("text", "") or "")):
                lowered = field.casefold()
                if lowered not in fields:
                    fields.append(lowered)
        return {
            "resource_kind": resource_kind,
            "context_ids": context_ids[:6],
            "source_path": source_path,
            "page_numbers": page_numbers[:6],
            "section_paths": section_paths[:4],
            "fields": fields[:12],
        }

    @staticmethod
    def extract_procedure_state(answer: str) -> dict:
        if not answer:
            return {}

        steps: list[dict] = []
        current_step: dict | None = None
        for raw_line in answer.replace("\r\n", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            normalized_line = re.sub(r"^\*+|\*+$", "", line).strip()
            match = re.match(r"^(?:\*\*)?(\d+)(?:\.\s+|\s*단계[:\s]+)(.+?)(?:\*\*)?$", normalized_line)
            if match:
                current_step = {"step_number": int(match.group(1)), "title": match.group(2).strip(), "body_lines": []}
                steps.append(current_step)
                continue
            if current_step is not None:
                current_step["body_lines"].append(line)

        if len(steps) < 2:
            return {}

        normalized_steps = []
        for step in steps:
            normalized_steps.append(
                {
                    "step_number": step["step_number"],
                    "title": step["title"],
                    "body": "\n".join(step["body_lines"]).strip(),
                }
            )
        return {"mode": "procedure", "steps": normalized_steps, "current_step": 1, "total_steps": len(normalized_steps)}

    @staticmethod
    def should_capture_procedure_state(query_interpretation, answer_route, policy_decision) -> bool:
        if answer_route == "procedure_state_followup":
            return True
        query_interpretation = query_interpretation or {}
        intent = str(query_interpretation.get("intent", "") or "").casefold()
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
        if intent == "procedure_followup" or response_shape == "procedure":
            return True
        return policy_decision.response_mode == "conversational" and policy_decision.turn_type != "conversational_ack"

    def finalize_answer(
        self,
        answer: str,
        rewritten_query: str,
        use_retrieved_context: bool,
        top_score: float,
        selected_context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
        response_mode: str,
        policy_decision,
        query_interpretation: dict | None,
        answer_route: str,
        retrieval_min_score: float,  # noqa: ARG002
        doc_type: str = "",
    ) -> tuple[str, list[dict], dict]:
        t_total = time.perf_counter()

        answer = self.strip_code_blocks_for_non_code_route(answer, answer_route)
        answer = self.sanitize_answer(answer, use_retrieved_context)

        t_citation = time.perf_counter()
        if use_retrieved_context and policy_decision.allow_citations and not self.should_suppress_citations(answer):
            answer = self.inject_inline_citations(answer, selected_context_items)
        answer_citations = (
            self.build_answer_citation_payload(answer, selected_context_items, grounded_pages, preferred_preview_source)
            if policy_decision.allow_citations
            else []
        )
        logger.info(
            "[Timing][Answer.finalize_answer] citation_phase=%.3fs citations=%d selected_context_items=%d grounded_pages=%d",
            time.perf_counter() - t_citation,
            len(answer_citations),
            len(selected_context_items),
            len(grounded_pages),
        )
        final_answer = self.ensure_answer_source_line(answer, answer_citations, use_retrieved_context)

        show_preview = policy_decision.allow_preview and use_retrieved_context and bool(answer_citations or grounded_pages)
        if show_preview:
            t_preview = time.perf_counter()
            final_source, final_preview_pages = self.build_answer_aligned_preview_pages(
                answer_citations,
                selected_context_items,
                preferred_preview_source,
                grounded_pages,
            )
            logger.info(
                "[Timing][Answer.finalize_answer] preview_phase=%.3fs preview_pages=%d source=%s",
                time.perf_counter() - t_preview,
                len(final_preview_pages),
                final_source,
            )
        else:
            final_source, final_preview_pages = None, []

        logger.info(
            "[Timing][Answer.finalize_answer] total=%.3fs answer_chars=%d",
            time.perf_counter() - t_total,
            len(final_answer),
        )

        final_payload = self.build_context_payload(
            rewritten_query,
            response_mode,
            top_score,
            final_source,
            final_preview_pages,
            selected_context_items,
            grounded_pages,
            answer_citations,
            preview_finalized=True,
        )
        final_payload["query_interpretation"] = query_interpretation or {}
        final_payload["answer_route"] = answer_route
        if doc_type:
            final_payload["doc_type"] = doc_type

        if answer_route == "extractive_code":
            final_payload["last_example_anchor"] = self.build_example_anchor(selected_context_items, query_interpretation)

        if self.should_capture_procedure_state(query_interpretation, answer_route, policy_decision):
            procedure_state = self.extract_procedure_state(final_answer)
        else:
            procedure_state = {}

        if procedure_state:
            if answer_route == "procedure_state_followup":
                final_payload["procedure_state"] = procedure_state
            else:
                final_payload["_stored_procedure_state"] = procedure_state

        return final_answer, answer_citations, final_payload
