from __future__ import annotations

from apps.api.api.schemas.chat import CopilotChatArtifact, CopilotChatArtifactItem, CopilotChatSourceItem
from apps.api.api.schemas.ocp_chat import OcpLiveChatResponse
from apps.api.api.schemas.ocp_live import OcpLiveResourceDetailResponse, OcpLiveResourceSummary
from apps.api.integrations.ocp.live_tool_executor import LiveExecutionResult


class LiveAnswerComposer:
    def compose(self, *, result: LiveExecutionResult) -> OcpLiveChatResponse:
        plan = result.plan
        if plan.intent == "namespace_list":
            items = result.namespace_list.items if result.namespace_list is not None else []
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:namespaces",
                answer=self._format_namespace_answer(items),
                items=[],
                sources=[],
                artifacts=[],
            )

        if plan.intent == "overview":
            overview = result.overview
            resource_counts = overview.resource_counts if overview is not None else {}
            default_namespace = overview.default_namespace if overview is not None else ""
            namespace_count = overview.namespace_count if overview is not None else 0
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:overview",
                namespace=default_namespace,
                answer=self._format_overview_answer(resource_counts, default_namespace, namespace_count),
                items=[],
                sources=[],
                artifacts=[],
            )

        if plan.intent == "resource_relations":
            relations = result.relations or {}
            relation_items = []
            for relation in list(relations.get("relations") or [])[:8]:
                pod_name = str((relation or {}).get("pod") or "")
                services = [str(item) for item in ((relation or {}).get("services") or []) if str(item)]
                relation_items.append(
                    CopilotChatArtifactItem(
                        name=pod_name,
                        kind="Pod",
                        namespace=str(relations.get("namespace") or plan.namespace or ""),
                        resource="pods",
                        metadata={"services": services},
                        action_target={
                            "resource": "pods",
                            "namespace": str(relations.get("namespace") or plan.namespace or ""),
                            "name": pod_name,
                            "kind": "Pod",
                        },
                    )
                )
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:resource_relations",
                resource=plan.resource,
                namespace=str(relations.get("namespace") or plan.namespace or ""),
                answer=self._format_relations_answer(relations),
                items=[],
                sources=[],
                artifacts=[
                    CopilotChatArtifact(
                        artifact_type="resource_list",
                        title="resource relations",
                        description="resolved live resource relations",
                        resource=plan.resource,
                        namespace=str(relations.get("namespace") or plan.namespace or ""),
                        items=relation_items,
                    )
                ],
            )

        if plan.intent == "resource_health":
            if result.health_summary is None and result.resource_list is not None and result.resource_list.items:
                return OcpLiveChatResponse(
                    connection_id=result.connection_id,
                    cluster_url=result.cluster_url,
                    mode="tool:resource_candidate_list",
                    resource=result.resource_list.resource,
                    namespace=result.resource_list.namespace,
                    answer=self._format_candidate_answer("resource_health", result.resource_list.resource, result.resource_list.namespace, result.resource_list.items),
                    items=result.resource_list.items[:8],
                    sources=[self._map_live_summary_item(item) for item in result.resource_list.items[:8]],
                    artifacts=[self._build_resource_list_artifact(resource=result.resource_list.resource, namespace=result.resource_list.namespace, items=result.resource_list.items[:8])],
                )
            health = result.health_summary or {}
            resource_name = str(health.get("name") or plan.resource_name or "")
            namespace = str(health.get("namespace") or plan.namespace or "")
            resource = str(health.get("resource") or plan.resource or "deployments")
            editor_artifact = None
            manifest_yaml = str(health.get("manifest_yaml") or "")
            if manifest_yaml and resource_name and namespace:
                detail = OcpLiveResourceDetailResponse(
                    connection_id=result.connection_id,
                    cluster_url=result.cluster_url,
                    resource=resource,
                    namespace=namespace,
                    name=resource_name,
                    kind=str(health.get("kind") or ""),
                    manifest_yaml=manifest_yaml,
                    manifest_json=(health.get("manifest_json") or {}),
                )
                editor_artifact = self._build_resource_editor_artifact(detail, None)
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:resource_health",
                resource=resource,
                namespace=namespace,
                answer=self._format_health_answer(health, result.doc_response.answer if result.doc_response is not None else ""),
                items=[],
                sources=list((result.doc_response.sources if result.doc_response is not None else [])[:3]),
                artifacts=[editor_artifact] if editor_artifact is not None else [],
            )

        if plan.intent in {"resource_yaml", "doc_compare"} and result.resource_detail is None:
            if result.resource_list is not None and result.resource_list.items:
                return OcpLiveChatResponse(
                    connection_id=result.connection_id,
                    cluster_url=result.cluster_url,
                    mode="tool:resource_candidate_list",
                    resource=result.resource_list.resource,
                    namespace=result.resource_list.namespace,
                    answer=self._format_candidate_answer(plan.intent, result.resource_list.resource, result.resource_list.namespace, result.resource_list.items),
                    items=result.resource_list.items[:8],
                    sources=[self._map_live_summary_item(item) for item in result.resource_list.items[:8]],
                    artifacts=[self._build_resource_list_artifact(resource=result.resource_list.resource, namespace=result.resource_list.namespace, items=result.resource_list.items[:8])],
                )
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:resource_detail" if plan.intent == "resource_yaml" else "tool:doc_compare",
                resource=plan.resource,
                namespace=plan.namespace,
                answer=self._missing_name_answer(plan.intent),
                items=[],
                sources=[],
                artifacts=[],
            )

        if plan.intent == "resource_yaml" and result.resource_detail is not None:
            detail = result.resource_detail
            selected_items = [result.selected_item] if result.selected_item is not None else [self._summary_from_detail(detail)]
            live_source = self._map_live_detail_source(detail, selected_items[0] if selected_items else None)
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:resource_detail",
                resource=detail.resource,
                namespace=detail.namespace,
                answer=self._format_resource_detail_answer(detail.kind, detail.name, detail.namespace, detail.manifest_yaml),
                items=[item for item in selected_items if item is not None],
                sources=[live_source],
                artifacts=[self._build_resource_editor_artifact(detail, selected_items[0] if selected_items else None)],
            )

        if plan.intent == "doc_compare" and result.resource_detail is not None:
            detail = result.resource_detail
            live_source = self._map_live_detail_source(detail, result.selected_item)
            doc_sources = list((result.doc_response.sources if result.doc_response is not None else [])[:3])
            return OcpLiveChatResponse(
                connection_id=result.connection_id,
                cluster_url=result.cluster_url,
                mode="tool:doc_compare",
                resource=detail.resource,
                namespace=detail.namespace,
                answer=self._format_doc_compare_answer(
                    detail=detail,
                    doc_sources=doc_sources,
                    doc_answer=result.doc_response.answer if result.doc_response is not None else "",
                    comparison=result.comparison.live_summary_lines if result.comparison is not None else [],
                    limitations=result.comparison.limitations if result.comparison is not None else [],
                ),
                items=[],
                sources=[live_source, *doc_sources],
                artifacts=[self._build_resource_editor_artifact(detail, result.selected_item)],
            )

        resource_list = result.resource_list
        resource = resource_list.resource if resource_list is not None else (plan.resource or "pods")
        namespace = resource_list.namespace if resource_list is not None else plan.namespace
        count = resource_list.count if resource_list is not None else 0
        items = resource_list.items if resource_list is not None else []
        return OcpLiveChatResponse(
            connection_id=result.connection_id,
            cluster_url=result.cluster_url,
            mode="tool:resource_list",
            resource=resource,
            namespace=namespace,
            answer=self._format_resource_answer(resource, namespace, count, items),
            items=items,
            sources=[self._map_live_summary_item(item) for item in items[:8]],
            artifacts=[self._build_resource_list_artifact(resource=resource, namespace=namespace, items=items)],
        )

    @staticmethod
    def _missing_name_answer(intent: str) -> str:
        if intent == "doc_compare":
            return "공식 문서와 비교하려면 정확한 리소스 이름이 필요합니다. 예: deployment web 공식 문서랑 live yaml 차이 알려줘"
        return "리소스 YAML을 보려면 정확한 리소스 이름이 필요합니다. 예: pod my-app-123 yaml 보여줘"

    @staticmethod
    def _format_namespace_answer(items: list[str]) -> str:
        if not items:
            return "조회 가능한 namespace를 찾지 못했습니다."
        preview = ", ".join(items[:8])
        return f"조회 가능한 namespace는 총 {len(items)}개입니다. 예시: {preview}"

    @staticmethod
    def _format_overview_answer(resource_counts: dict[str, int], default_namespace: str, namespace_count: int) -> str:
        if not resource_counts:
            return "클러스터 overview를 불러왔지만 기본 namespace 기준 리소스 통계가 비어 있습니다."
        parts = [f"{name} {count if count >= 0 else 'N/A'}개" for name, count in resource_counts.items()]
        scope = default_namespace or "기본 namespace 미설정"
        return f"현재 연결은 namespace {scope} 기준이며, 전체 namespace는 {namespace_count}개입니다. 리소스 요약: {', '.join(parts)}."

    @staticmethod
    def _format_resource_answer(
        resource: str,
        namespace: str,
        count: int,
        items: list[OcpLiveResourceSummary],
    ) -> str:
        if count == 0:
            return f"{namespace} namespace에서 {resource}를 찾지 못했습니다."
        preview_names = ", ".join(item.name for item in items[:5])
        return f"{namespace} namespace에서 {resource}는 총 {count}개입니다. 예시: {preview_names}"

    @staticmethod
    def _format_resource_detail_answer(kind: str, name: str, namespace: str, manifest_yaml: str) -> str:
        return f"{namespace} namespace의 {kind} {name} manifest입니다.\n```yaml\n{manifest_yaml}\n```"

    @staticmethod
    def _format_candidate_answer(intent: str, resource: str, namespace: str, items: list[OcpLiveResourceSummary]) -> str:
        preview = ", ".join(item.name for item in items[:5])
        if intent == "doc_compare":
            return f"{namespace} namespace에서 비교할 {resource} 후보를 먼저 골라야 합니다. 예시: {preview}"
        if intent == "resource_health":
            return f"{namespace} namespace에서 진단할 {resource} 후보를 먼저 골라야 합니다. 예시: {preview}"
        return f"{namespace} namespace에서 볼 {resource} 후보를 먼저 골라야 합니다. 예시: {preview}"

    @staticmethod
    def _format_doc_compare_answer(
        *,
        detail: OcpLiveResourceDetailResponse,
        doc_sources: list[CopilotChatSourceItem],
        doc_answer: str,
        comparison: list[str],
        limitations: list[str],
    ) -> str:
        lines = [
            f"현재 live {detail.kind} {detail.name}의 상태를 기준으로 관련 공식 문서와 비교했습니다.",
            "",
            "현재 live YAML 요약",
            *(comparison or [f"- namespace: {detail.namespace}", f"- kind: {detail.kind}", f"- name: {detail.name}"]),
        ]
        if doc_sources:
            lines.extend(["", "관련 공식 문서 근거"])
            for index, source in enumerate(doc_sources, start=1):
                section = str(source.metadata.get("section_title") or source.label or "document").strip()
                preview = str(source.metadata.get("preview_text") or source.metadata.get("synthesis_text") or "").strip()
                lines.append(f"- [{index}] {section}: {preview[:220]}")
        elif doc_answer:
            lines.extend(["", "문서 검색 결과", doc_answer])
        if limitations:
            lines.extend(["", "비교 안내", *limitations])
        return "\n".join(lines).strip()

    @staticmethod
    def _format_relations_answer(relations: dict[str, object]) -> str:
        namespace = str(relations.get("namespace") or "-")
        resolved = list(relations.get("relations") or [])
        if not resolved:
            return f"{namespace} namespace에서 명확한 pod → service 연결 관계를 찾지 못했습니다."
        lines = [f"{namespace} namespace의 pod → service 연결 관계입니다."]
        for relation in resolved[:8]:
            pod_name = str((relation or {}).get("pod") or "")
            services = [str(item) for item in ((relation or {}).get("services") or []) if str(item)]
            if pod_name and services:
                lines.append(f"- {pod_name} -> {', '.join(services)}")
        return "\n".join(lines)

    @staticmethod
    def _format_health_answer(health: dict[str, object], doc_answer: str) -> str:
        resource = str(health.get("resource") or "resource")
        name = str(health.get("name") or "")
        namespace = str(health.get("namespace") or "")
        issues = [str(item) for item in (health.get("issues") or []) if str(item)]
        lines = [f"{namespace} namespace의 {resource} {name} 진단 결과입니다."]
        if issues:
            lines.extend([f"- {issue}" for issue in issues])
        else:
            lines.append("- 현재 live 상태에서 즉시 보이는 문제는 없습니다.")
        if doc_answer.strip():
            lines.extend(["", "관련 문서 기준", doc_answer.strip()])
        return "\n".join(lines).strip()

    @staticmethod
    def _map_live_summary_item(item: OcpLiveResourceSummary) -> CopilotChatSourceItem:
        return CopilotChatSourceItem(
            source_type="live",
            label=item.name,
            namespace=item.namespace,
            kind=item.kind,
            provenance=["live", "tool:list_resources"],
            metadata=item.model_dump(),
        )

    @staticmethod
    def _map_live_detail_source(
        detail: OcpLiveResourceDetailResponse,
        summary: OcpLiveResourceSummary | None = None,
    ) -> CopilotChatSourceItem:
        metadata = {
            "manifest_yaml": detail.manifest_yaml,
            "resource": detail.resource,
            "namespace": detail.namespace,
            "kind": detail.kind,
            "name": detail.name,
            **(summary.model_dump() if summary is not None else {}),
        }
        return CopilotChatSourceItem(
            source_type="live",
            label=detail.name,
            namespace=detail.namespace,
            kind=detail.kind,
            provenance=["live", "tool:get_resource_yaml"],
            metadata=metadata,
        )

    @staticmethod
    def _summary_from_detail(detail: OcpLiveResourceDetailResponse) -> OcpLiveResourceSummary:
        return OcpLiveResourceSummary(name=detail.name, namespace=detail.namespace, kind=detail.kind)

    @staticmethod
    def _build_resource_list_artifact(
        *,
        resource: str,
        namespace: str,
        items: list[OcpLiveResourceSummary],
    ) -> CopilotChatArtifact:
        return CopilotChatArtifact(
            artifact_type="resource_list",
            title=f"{resource} list",
            description=f"{namespace} namespace resource list",
            resource=resource,
            namespace=namespace,
            payload={"count": len(items)},
            items=[
                CopilotChatArtifactItem(
                    name=item.name,
                    kind=item.kind,
                    namespace=item.namespace,
                    resource=resource,
                    phase=item.phase,
                    type=item.type,
                    host=item.host,
                    to=item.to,
                    node_name=item.node_name,
                    cluster_ip=item.cluster_ip,
                    action_target={
                        "resource": resource,
                        "namespace": item.namespace or namespace,
                        "name": item.name,
                        "kind": item.kind,
                    },
                    metadata=item.model_dump(),
                )
                for item in items
            ],
        )

    @staticmethod
    def _build_resource_editor_artifact(
        detail: OcpLiveResourceDetailResponse,
        summary: OcpLiveResourceSummary | None,
    ) -> CopilotChatArtifact:
        return CopilotChatArtifact(
            artifact_type="resource_editor",
            title=f"{detail.name} YAML",
            description="Open live YAML editor",
            resource=detail.resource,
            namespace=detail.namespace,
            resource_name=detail.name,
            payload={
                "kind": detail.kind,
                "manifest_yaml": detail.manifest_yaml,
                "manifest_json": detail.manifest_json,
            },
            items=[
                CopilotChatArtifactItem(
                    name=detail.name,
                    kind=detail.kind,
                    namespace=detail.namespace,
                    resource=detail.resource,
                    phase=summary.phase if summary is not None else "",
                    type=summary.type if summary is not None else "",
                    host=summary.host if summary is not None else "",
                    to=summary.to if summary is not None else "",
                    node_name=summary.node_name if summary is not None else "",
                    cluster_ip=summary.cluster_ip if summary is not None else "",
                    action_target={
                        "resource": detail.resource,
                        "namespace": detail.namespace,
                        "name": detail.name,
                        "kind": detail.kind,
                    },
                    metadata=(summary.model_dump() if summary is not None else {}),
                )
            ],
        )
