from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from app.config import Settings, get_settings
from app.ocp_chat import OcpChatService
from app.ocp_client import OcpApiClient
from app.rag.pipeline import RagPipeline
from app.rag.types import TurnPolicyDecision
from app.rag.indexing import IndexingService
from app.session.repository import SessionRepository
from app.storage import TaskRepository


@dataclass(slots=True)
class AppContainer:
    """Shared runtime container exposed to FastAPI routes."""

    settings: Settings
    pipeline: RagPipeline
    chat_service: ChatService
    session_repository: SessionRepository
    indexing_service: IndexingService
    task_repository: TaskRepository
    ocp_api_client: OcpApiClient
    ocp_chat_service: OcpChatService


def build_container(settings: Settings | None = None) -> AppContainer:
    """Build the runtime container from settings."""
    resolved_settings = settings or get_settings()
    pipeline = RagPipeline(resolved_settings)
    session_repository = SessionRepository(pipeline.session_store)
    task_repository = TaskRepository(resolved_settings.db_dsn)
    ocp_api_client = OcpApiClient(
        base_url=resolved_settings.ocp_api_base_url,
        token=resolved_settings.ocp_api_token,
        verify_ssl=resolved_settings.ocp_api_verify_ssl,
        default_namespace=resolved_settings.ocp_default_namespace,
    )
    ocp_chat_service = OcpChatService(
        ocp_api_client=ocp_api_client,
        session_repository=session_repository,
        answer_service=pipeline.answer_service,
        llm_client=pipeline.llm,
    )
    return AppContainer(
        settings=resolved_settings,
        pipeline=pipeline,
        chat_service=ChatService(pipeline, session_repository, ocp_chat_service),
        session_repository=session_repository,
        indexing_service=pipeline.indexing_service,
        task_repository=task_repository,
        ocp_api_client=ocp_api_client,
        ocp_chat_service=ocp_chat_service,
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


class ChatService:
    """Public entrypoint for chat and retry flows."""

    def __init__(self, pipeline: Any, session_repository: Any, ocp_chat_service: Any) -> None:
        self.pipeline = pipeline
        self.session_repository = session_repository
        self.ocp_chat_service = ocp_chat_service

    def stream(self, request: Any):
        query_mode = self.ocp_chat_service.detect_query_mode(request.session_id, request.message)
        if query_mode == "mixed":
            return self._stream_mixed(request)
        if query_mode == "ocp":
            return self.ocp_chat_service.stream(
                session_id=request.session_id,
                user_message=request.message,
                append_user_turn=getattr(request, "append_user_turn", True),
            )
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=request.message,
            allowed_source_paths=request.allowed_source_paths,
            uploaded_source_paths=getattr(request, "uploaded_source_paths", None),
            append_user_turn=request.append_user_turn,
            version_tag=getattr(request, "version_tag", None),
        )

    def retry(self, request: Any):
        requested_message = (request.message or "").strip()
        pending_message = self.session_repository.pending_user_message(
            request.session_id,
            owner_id=request.owner_id,
        )
        user_message = pending_message or requested_message
        if not user_message:
            raise LookupError("No pending user message found for retry.")

        append_user_turn = request.append_user_turn
        if pending_message and (not requested_message or pending_message == requested_message):
            append_user_turn = False
        query_mode = self.ocp_chat_service.detect_query_mode(request.session_id, user_message)
        if query_mode == "mixed":
            mixed_request = type(
                "MixedRetryRequest",
                (),
                {
                    "session_id": request.session_id,
                    "message": user_message,
                    "allowed_source_paths": request.allowed_source_paths,
                    "uploaded_source_paths": getattr(request, "uploaded_source_paths", None),
                    "append_user_turn": append_user_turn,
                    "version_tag": getattr(request, "version_tag", None),
                },
            )()
            return self._stream_mixed(mixed_request)
        if query_mode == "ocp":
            return self.ocp_chat_service.stream(
                session_id=request.session_id,
                user_message=user_message,
                append_user_turn=append_user_turn,
            )
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=user_message,
            allowed_source_paths=request.allowed_source_paths,
            uploaded_source_paths=getattr(request, "uploaded_source_paths", None),
            append_user_turn=append_user_turn,
        )

    async def _stream_mixed(self, request: Any):
        session_id = request.session_id
        user_message = request.message
        append_user_turn = getattr(request, "append_user_turn", True)

        plan = await self.ocp_chat_service._build_plan(session_id, user_message)
        if plan is None:
            async for event in self.pipeline.stream_chat(
                session_id=session_id,
                user_message=user_message,
                allowed_source_paths=request.allowed_source_paths,
                uploaded_source_paths=getattr(request, "uploaded_source_paths", None),
                append_user_turn=append_user_turn,
                version_tag=getattr(request, "version_tag", None),
            ):
                yield event
            return

        if append_user_turn:
            self.session_repository.add_turn(session_id, "user", user_message)

        query_interpretation: dict = {}
        doc_payload: dict | None = None
        yield {"type": "status", "stage": "searching_documents", "message": "공식 문서에서 관련 명령어를 찾는 중입니다."}
        doc_query = self._build_mixed_document_query(user_message, plan)
        doc_state = await self.pipeline._prepare_retrieval_state(
            session_id,
            doc_query,
            request.allowed_source_paths,
            uploaded_source_paths=getattr(request, "uploaded_source_paths", None),
            version_tag=getattr(request, "version_tag", None),
        )
        doc_answer, doc_payload, query_interpretation = self._build_mixed_document_answer(user_message, doc_state, plan, doc_query=doc_query)
        if self._needs_mixed_yaml_retry(plan, doc_answer):
            retry_query = self._build_mixed_document_retry_query(doc_query)
            retry_state = await self.pipeline._prepare_retrieval_state(
                session_id,
                retry_query,
                request.allowed_source_paths,
                uploaded_source_paths=getattr(request, "uploaded_source_paths", None),
                version_tag=getattr(request, "version_tag", None),
            )
            retry_answer, retry_payload, retry_qi = self._build_mixed_document_answer(user_message, retry_state, plan, doc_query=retry_query)
            if self._document_answer_has_yaml_signal(retry_answer):
                doc_answer, doc_payload, query_interpretation = retry_answer, retry_payload, retry_qi

        yield {"type": "status", "stage": "querying_ocp", "message": "현재 OpenShift 상태를 조회하는 중입니다."}
        ocp_result = await self.ocp_chat_service._execute_plan(plan)

        final_answer = self._compose_mixed_answer(
            user_message=user_message,
            document_answer=doc_answer,
            ocp_answer=ocp_result["answer"],
            ocp_answer_route=ocp_result["answer_route"],
        )
        final_payload = doc_payload or self.pipeline.answer_service.build_context_payload(
            user_message,
            "mixed",
            max(float(doc_state.get("top_score") or 0.0), float(getattr(plan, "confidence", 0.0) or 0.0)),
            None,
            [],
            [],
            [],
            [],
            preview_finalized=True,
        )
        final_payload["mode"] = "mixed"
        final_payload["answer_route"] = "mixed_doc_ocp"
        final_payload["query_interpretation"] = {
            **(query_interpretation or {}),
            "mixed_with_ocp": True,
        }
        final_payload["ocp_context"] = ocp_result["ocp_context"]
        final_payload["ocp_answer_route"] = ocp_result["answer_route"]
        if isinstance(final_payload.get("items"), list):
            final_payload["items"] = final_payload["items"][:3]

        yield {"type": "context", **self.pipeline.answer_service.public_context_payload(final_payload)}
        yield {"type": "token", "content": final_answer, "cached": False}
        self.session_repository.add_turn(session_id, "assistant", final_answer, metadata=final_payload)
        yield {"type": "done", "cached": False}

    def _build_mixed_document_answer(self, user_message: str, state: dict, plan: Any | None = None, doc_query: str | None = None) -> tuple[str, dict | None, dict]:
        query_interpretation = dict(state.get("query_interpretation") or {})
        if plan is not None and not query_interpretation.get("resources"):
            inferred_resources: list[str] = []
            for resource in list(getattr(plan, "resources", []) or []):
                normalized = str(resource or "").strip().casefold()
                if normalized.endswith("s"):
                    normalized = normalized[:-1]
                if normalized and normalized not in inferred_resources:
                    inferred_resources.append(normalized)
            if inferred_resources:
                query_interpretation["resources"] = inferred_resources
        if self._is_compare_request(user_message) and str(getattr(plan, "mode", "") or "").casefold() == "yaml":
            query_interpretation["generic_command_query"] = True
            format_constraints = [str(value).casefold().strip() for value in query_interpretation.get("format_constraints", []) if value]
            if "yaml" not in format_constraints:
                format_constraints.append("yaml")
            query_interpretation["format_constraints"] = format_constraints
        selected_context_items = list(state.get("selected_context_items") or [])
        ordered_context_items = list(state.get("ordered_context_items") or selected_context_items)
        if not selected_context_items:
            return "", None, query_interpretation

        answer_route = self.pipeline._resolve_answer_route(query_interpretation)
        answer = ""
        final_context_items = selected_context_items
        answer_query = str(doc_query or user_message)
        if answer_route == "extractive_code":
            final_context_items = self.pipeline._select_code_example_context_items(
                answer_query,
                query_interpretation,
                ordered_context_items,
                selected_context_items,
            )
            answer = self.pipeline.answer_service.build_extractive_code_answer(
                final_context_items,
                requested_resource_kinds=self.pipeline._resolve_requested_resource_kinds(query_interpretation),
                user_message=answer_query,
                query_interpretation=query_interpretation,
            ) or ""
        elif answer_route == "extractive_table":
            answer = self.pipeline.answer_service.build_extractive_table_answer(selected_context_items) or ""
        else:
            answer = self.pipeline.answer_service.build_extractive_text_answer(selected_context_items) or ""

        if not answer:
            return "", None, query_interpretation

        turn_policy = state.get("turn_policy") or {}
        policy_decision = (
            TurnPolicyDecision(**turn_policy)
            if turn_policy
            else TurnPolicyDecision(
                turn_type="document_query",
                response_mode="rag",
                use_retrieval=True,
                use_memory_rewrite=False,
                allow_preview=True,
                allow_citations=True,
            )
        )
        _final_answer, _citations, payload = self.pipeline._finalize_answer(
            answer=answer,
            rewritten_query=state.get("rewritten_query") or user_message,
            use_retrieved_context=bool(final_context_items),
            top_score=float(state.get("top_score") or 0.0),
            selected_context_items=final_context_items,
            grounded_pages=list(state.get("grounded_pages") or []),
            preferred_preview_source=state.get("preferred_preview_source"),
            response_mode=state.get("response_mode", "rag"),
            policy_decision=policy_decision,
            query_interpretation=query_interpretation,
            answer_route=answer_route,
            doc_type=state.get("doc_type", ""),
        )
        return answer.strip(), payload, query_interpretation

    @staticmethod
    def _is_compare_request(user_message: str) -> bool:
        lowered = str(user_message or "").casefold()
        return any(marker in lowered for marker in ("차이", "비교", "다르", "달라", "difference", "compare"))

    def _compose_mixed_answer(
        self,
        *,
        user_message: str,
        document_answer: str,
        ocp_answer: str,
        ocp_answer_route: str,
    ) -> str:
        sections: list[str] = []
        if self._is_compare_request(user_message):
            if document_answer.startswith("공식 문서 기준\n"):
                return document_answer.strip()
            if document_answer:
                sections.append("공식 문서 기준\n" + document_answer.strip())
            live_heading = "현재 OCP 기준"
            live_body = str(ocp_answer or "").strip()
            if ocp_answer_route == "ocp_yaml" and "먼저 알려 주세요" in live_body:
                live_body = (
                    "실제 YAML은 비교 대상 Pod를 먼저 특정해야 확인할 수 있습니다.\n"
                    + live_body
                )
            sections.append(live_heading + "\n" + live_body)
            sections.append("비교 가이드\n공식 문서 예시와 실제 YAML을 나란히 보려면 비교할 Pod 이름을 먼저 지정해 주세요.")
            return "\n\n".join(section for section in sections if section.strip()).strip()

        if document_answer:
            sections.append("문서 기준 명령어\n" + document_answer.strip())
        sections.append("현재 OCP 결과\n" + str(ocp_answer or "").strip())
        return "\n\n".join(section for section in sections if section.strip()).strip()

    def _build_mixed_document_query(self, user_message: str, plan: Any | None) -> str:
        if plan is None:
            return user_message

        lowered = str(user_message or "").casefold()
        resources = [str(resource).strip().casefold() for resource in list(getattr(plan, "resources", []) or []) if resource]
        singular_resources = [resource[:-1] if resource.endswith("s") else resource for resource in resources]
        mode = str(getattr(plan, "mode", "") or "").casefold()
        status_check = bool(getattr(plan, "status_check", False))
        warning_only = bool(getattr(plan, "warning_only", False))
        explicit_resource_mentioned = any(resource in lowered or singular in lowered for resource, singular in zip(resources, singular_resources, strict=False))

        if mode == "yaml":
            resource = singular_resources[0] if singular_resources else "resource"
            if self._is_compare_request(user_message):
                return f"{resource} yaml -o yaml oc describe {resource} official example"
            return f"{resource} yaml -o yaml 명령어"
        if warning_only:
            return "warning 이벤트 확인 명령어"
        if status_check and (not singular_resources or not explicit_resource_mentioned):
            return "현재 상태 확인 명령어"
        if singular_resources:
            resource = singular_resources[0]
            if status_check:
                return f"{resource} 상태 확인 명령어"
            return f"{resource} 확인 명령어"
        return user_message

    @staticmethod
    def _build_mixed_document_retry_query(doc_query: str) -> str:
        lowered = str(doc_query or "").casefold()
        if "yaml" in lowered or "-o yaml" in lowered:
            return f"{doc_query} describe"
        return f"{doc_query} oc"

    @staticmethod
    def _document_answer_has_yaml_signal(answer: str) -> bool:
        lowered = str(answer or "").casefold()
        return any(marker in lowered for marker in ("-o yaml", "```yaml", "yaml", "oc describe"))

    def _needs_mixed_yaml_retry(self, plan: Any | None, doc_answer: str) -> bool:
        if plan is None:
            return False
        if str(getattr(plan, "mode", "") or "").casefold() != "yaml":
            return False
        return not self._document_answer_has_yaml_signal(doc_answer)
