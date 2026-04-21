from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from apps.api.schemas.chat import CopilotChatResponse, CopilotChatStage
from apps.api.schemas.ocp_live import (
    OcpLiveNamespaceListResponse,
    OcpLiveResourceDetailResponse,
    OcpLiveResourceListResponse,
    OcpLiveResourceSummary,
    OcpOverviewResponse,
)
from apps.api.ocp.auth import OcpConnectionBroker
from apps.api.ocp.live_question_planner import LiveQuestionPlan
from apps.api.ocp.live_service import ConnectedOcpService
from apps.api.rag.retrieval.document_retriever import DocumentRetriever


@dataclass(slots=True)
class LiveComparisonSummary:
    live_summary_lines: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LiveExecutionResult:
    connection_id: str
    cluster_url: str
    plan: LiveQuestionPlan
    namespace_list: OcpLiveNamespaceListResponse | None = None
    overview: OcpOverviewResponse | None = None
    resource_list: OcpLiveResourceListResponse | None = None
    resource_detail: OcpLiveResourceDetailResponse | None = None
    selected_item: OcpLiveResourceSummary | None = None
    doc_response: CopilotChatResponse | None = None
    comparison: LiveComparisonSummary | None = None
    relations: dict[str, object] | None = None
    health_summary: dict[str, object] | None = None
    executed_operations: list[str] = field(default_factory=list)


class LiveToolExecutor:
    def __init__(
        self,
        *,
        live_service: ConnectedOcpService,
        document_retriever: DocumentRetriever | None = None,
    ) -> None:
        self.live_service = live_service
        self.document_retriever = document_retriever

    async def execute(
        self,
        *,
        connection_id: str,
        cluster_url: str,
        plan: LiveQuestionPlan,
        broker: OcpConnectionBroker,
        progress: Callable[[CopilotChatStage], Awaitable[None] | None] | None = None,
    ) -> LiveExecutionResult:
        result = LiveExecutionResult(connection_id=connection_id, cluster_url=cluster_url, plan=plan)
        for operation in plan.operations:
            if operation == "list_namespaces":
                await self._emit_progress(
                    progress,
                    key="tool_list_namespaces",
                    label="tool: list_namespaces",
                    detail="접근 가능한 namespace 목록을 조회하고 있습니다.",
                )
                result.namespace_list = await self.live_service.list_namespaces(connection_id, broker)
            elif operation == "get_overview":
                await self._emit_progress(
                    progress,
                    key="tool_get_overview",
                    label="tool: get_overview",
                    detail="클러스터 overview와 주요 리소스 분포를 조회하고 있습니다.",
                )
                result.overview = await self.live_service.get_overview(connection_id, broker)
            elif operation == "list_resources":
                await self._emit_progress(
                    progress,
                    key="tool_list_resources",
                    label="tool: list_resources",
                    detail=f"{plan.namespace or 'default'} namespace의 {plan.resource or 'pods'} 목록을 조회하고 있습니다.",
                )
                result.resource_list = await self.live_service.list_resources(
                    connection_id,
                    resource=plan.resource or "pods",
                    namespace=plan.namespace,
                    broker=broker,
                )
            elif operation == "get_resource_yaml":
                if plan.resource_name:
                    await self._emit_progress(
                        progress,
                        key="tool_get_resource_yaml",
                        label="tool: get_resource_yaml",
                        detail="선택한 리소스의 live YAML/manifest를 조회하고 있습니다.",
                    )
                    result.resource_detail = await self.live_service.resolve_resource_detail_by_name(
                        connection_id,
                        namespace=plan.namespace,
                        name=plan.resource_name,
                        broker=broker,
                        preferred_resource=plan.resource,
                    )
                    result.selected_item = await self._load_selected_item(
                        connection_id=connection_id,
                        detail=result.resource_detail,
                        broker=broker,
                    )
                elif plan.resource:
                    result.resource_list = await self.live_service.list_resources(
                        connection_id,
                        resource=plan.resource or "pods",
                        namespace=plan.namespace,
                        broker=broker,
                    )
            elif operation == "search_official_docs":
                if self.document_retriever is not None and result.resource_detail is not None:
                    await self._emit_progress(
                        progress,
                        key="tool_search_official_docs",
                        label="tool: search_official_docs",
                        detail="관련 공식 문서를 검색하고 있습니다.",
                    )
                    query = plan.doc_topic or self._default_doc_topic(plan=plan, detail=result.resource_detail)
                    result.doc_response = await self.document_retriever.answer(message=query)
            elif operation == "compare_live_with_docs":
                result.comparison = self._build_comparison_summary(detail=result.resource_detail, doc_response=result.doc_response)
            elif operation == "resolve_resource_relations":
                await self._emit_progress(
                    progress,
                    key="tool_resolve_resource_relations",
                    label="tool: resolve_resource_relations",
                    detail="리소스 간 연결 관계를 분석하고 있습니다.",
                )
                result.relations = await self.live_service.resolve_resource_relations(
                    connection_id,
                    namespace=plan.namespace,
                    broker=broker,
                    resource=plan.resource,
                    name=plan.resource_name,
                )
            elif operation == "get_resource_health_summary":
                if plan.resource_name:
                    await self._emit_progress(
                        progress,
                        key="tool_get_resource_health_summary",
                        label="tool: get_resource_health_summary",
                        detail="리소스 상태와 관련 이벤트를 기반으로 진단 정보를 만들고 있습니다.",
                    )
                    result.health_summary = await self.live_service.get_resource_health_summary(
                        connection_id,
                        namespace=plan.namespace,
                        name=plan.resource_name,
                        broker=broker,
                        preferred_resource=plan.resource,
                    )
                    if result.resource_detail is None and isinstance(result.health_summary.get("manifest_yaml"), str):
                        pass
                elif plan.resource:
                    result.resource_list = await self.live_service.list_resources(
                        connection_id,
                        resource=plan.resource or "deployments",
                        namespace=plan.namespace,
                        broker=broker,
                    )
            result.executed_operations.append(operation)
        return result

    async def _load_selected_item(
        self,
        *,
        connection_id: str,
        detail: OcpLiveResourceDetailResponse,
        broker: OcpConnectionBroker,
    ) -> OcpLiveResourceSummary | None:
        summary = await self.live_service.list_resources(
            connection_id,
            resource=detail.resource,
            namespace=detail.namespace,
            broker=broker,
        )
        for item in summary.items:
            if item.name.casefold() == detail.name.casefold():
                return item
        return None

    @staticmethod
    def _default_doc_topic(*, plan: LiveQuestionPlan, detail: OcpLiveResourceDetailResponse) -> str:
        return f"{plan.resource or detail.resource} {detail.kind} {detail.name} yaml manifest configuration"

    @staticmethod
    def _build_comparison_summary(
        *,
        detail: OcpLiveResourceDetailResponse | None,
        doc_response: CopilotChatResponse | None,
    ) -> LiveComparisonSummary:
        summary = LiveComparisonSummary()
        if detail is not None:
            summary.live_summary_lines.extend(
                [
                    f"- namespace: {detail.namespace}",
                    f"- kind: {detail.kind}",
                    f"- name: {detail.name}",
                ]
            )
            spec = detail.manifest_json.get("spec") if isinstance(detail.manifest_json, dict) else {}
            if isinstance(spec, dict) and "replicas" in spec:
                summary.live_summary_lines.append(f"- replicas: {spec.get('replicas')}")
            template_spec = ((spec or {}).get("template") or {}).get("spec") if isinstance(spec, dict) else {}
            containers = (template_spec or {}).get("containers") if isinstance(template_spec, dict) else None
            if isinstance(containers, list) and containers:
                names = [str(item.get("name") or "") for item in containers if isinstance(item, dict)]
                images = [str(item.get("image") or "") for item in containers if isinstance(item, dict)]
                if any(names):
                    summary.live_summary_lines.append(f"- containers: {', '.join(filter(None, names))}")
                if any(images):
                    summary.live_summary_lines.append(f"- images: {', '.join(filter(None, images[:3]))}")
        if doc_response is None or (not doc_response.sources and not doc_response.answer):
            summary.limitations.append("- 직접 매칭되는 공식 문서 근거를 찾지 못했습니다.")
        summary.limitations.extend(
            [
                "- live YAML은 현재 클러스터 상태입니다.",
                "- 공식 문서 비교는 설명/권장 구성 기준의 요약 비교이며, 1:1 구조 diff는 아닐 수 있습니다.",
            ]
        )
        return summary

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


