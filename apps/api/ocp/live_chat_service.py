from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from apps.api.schemas.chat import CopilotChatHistoryTurn, CopilotChatStage, OcpLiveChatResponse
from apps.api.ocp.auth import OcpConnectionBroker
from apps.api.ocp.live_answer_composer import LiveAnswerComposer
from apps.api.ocp.live_question_planner import LiveQuestionPlanner
from apps.api.ocp.live_service import ConnectedOcpService
from apps.api.ocp.live_tool_executor import LiveToolExecutor
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient
from apps.api.rag.retrieval.document_retriever import DocumentRetriever


class LiveOcpChatService:
    """Tool-driven live OCP chat with planner, deterministic executor, and answer composer."""

    def __init__(
        self,
        live_service: ConnectedOcpService,
        *,
        document_retriever: DocumentRetriever | None = None,
        llm_client: OpenAiCompatibleLlmClient | None = None,
        question_planner: LiveQuestionPlanner | None = None,
        tool_executor: LiveToolExecutor | None = None,
        answer_composer: LiveAnswerComposer | None = None,
    ) -> None:
        self.live_service = live_service
        self.document_retriever = document_retriever
        self.question_planner = question_planner or LiveQuestionPlanner(llm_client=llm_client)
        self.tool_executor = tool_executor or LiveToolExecutor(
            live_service=live_service,
            document_retriever=document_retriever,
        )
        self.answer_composer = answer_composer or LiveAnswerComposer()

    async def answer(
        self,
        *,
        connection_id: str,
        message: str,
        namespace: str,
        broker: OcpConnectionBroker,
        progress: Callable[[CopilotChatStage], Awaitable[None] | None] | None = None,
        recent_turns: list[CopilotChatHistoryTurn] | None = None,
    ) -> OcpLiveChatResponse:
        await self._emit_progress(
            progress,
            key="verify_connection",
            label="클러스터 연결 확인 중",
            detail="선택한 연결 정보와 기본 namespace를 확인하고 있습니다.",
        )
        profile = broker.get_profile(connection_id)
        if profile is None:
            raise LookupError(f"Unknown connection_id={connection_id}")

        await self._emit_progress(
            progress,
            key="plan_live_tools",
            label="Live tool plan 생성 중",
            detail="질문을 해석해서 overview/list/yaml/doc 비교 도구 호출 계획을 만들고 있습니다.",
        )
        plan = await self.question_planner.plan(
            message=message,
            recent_turns=recent_turns or [],
            last_namespace=namespace or profile.default_namespace,
        )
        if not plan.namespace and plan.intent != "namespace_list":
            plan.namespace = namespace or profile.default_namespace

        execution = await self.tool_executor.execute(
            connection_id=connection_id,
            cluster_url=profile.cluster_url,
            plan=plan,
            broker=broker,
            progress=progress,
        )
        return self.answer_composer.compose(result=execution)

    @staticmethod
    async def _emit_progress(
        callback: Callable[[CopilotChatStage], Awaitable[None] | None] | None,
        *,
        key: str,
        label: str,
        detail: str,
    ) -> None:
        if callback is None:
            return
        outcome = callback(CopilotChatStage(key=key, label=label, detail=detail, status="running"))
        if inspect.isawaitable(outcome):
            await outcome


