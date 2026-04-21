from __future__ import annotations

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.auth import router as auth_router, _broker
from apps.api.routes.chat import router as chat_router
from apps.api.routes.ocp import router as ocp_router
from apps.api.schemas.copilot_chat import CopilotChatResponse, CopilotChatSourceItem
from apps.api.runtime import connected_ocp_service, live_ocp_chat_service


class _FakeDocumentRetriever:
    async def answer(self, *, message: str, allowed_source_paths=None):  # noqa: ANN001
        del allowed_source_paths
        return CopilotChatResponse(
            lane="doc_new",
            mode="keyword_retrieval",
            answer=f"doc answer for {message}",
            sources=[
                CopilotChatSourceItem(
                    source_type="doc",
                    label="official docs",
                    source_path="data/corpus/pdfs/official/en/deployments.md",
                    relative_source_path="data/corpus/pdfs/official/en/deployments.md",
                    provenance=["doc_new"],
                    metadata={
                        "section_title": "Deployment configuration",
                        "preview_text": "Deployments manage replicated pods and rollout behavior.",
                    },
                )
            ],
        )


class LiveOcpChatRouteTests(unittest.TestCase):
    LIVE_POD_NAME = "el-kugnus-bot-listener-c9ffbb846-fpcts"

    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(auth_router, prefix="/api/v1/auth")
        self.app.include_router(chat_router, prefix="/api/v1")
        self.app.include_router(ocp_router, prefix="/api/v1")
        self.client = TestClient(self.app)
        _broker.profile_store.clear()
        _broker.secret_store.clear()
        connected_ocp_service.transport = None
        live_ocp_chat_service.document_retriever = None

    def test_live_chat_answers_pod_question(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/api/v1/namespaces/demo/pods"):
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "metadata": {"name": self.LIVE_POD_NAME, "namespace": "demo"},
                                "status": {"phase": "Running"},
                                "spec": {"nodeName": "node-1"},
                            }
                        ]
                    },
                )
            if path.endswith(f"/api/v1/namespaces/demo/pods/{self.LIVE_POD_NAME}"):
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "v1",
                        "kind": "Pod",
                        "metadata": {"name": self.LIVE_POD_NAME, "namespace": "demo"},
                        "spec": {"nodeName": "node-1"},
                        "status": {"phase": "Running"},
                    },
                )
            if path.endswith("/api/v1/namespaces"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "demo"}}]})
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)

        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": "show pods",
                "namespace": "demo",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resource"], "pods")
        self.assertEqual(payload["namespace"], "demo")
        self.assertIn(self.LIVE_POD_NAME, payload["answer"])
        self.assertEqual(payload["items"][0]["name"], self.LIVE_POD_NAME)
        self.assertEqual(payload["artifacts"][0]["artifact_type"], "resource_list")
        self.assertEqual(payload["artifacts"][0]["items"][0]["name"], self.LIVE_POD_NAME)

        yaml_response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": f"pod {self.LIVE_POD_NAME} yaml 보여줘",
                "namespace": "demo",
            },
        )
        self.assertEqual(yaml_response.status_code, 200)
        yaml_payload = yaml_response.json()
        self.assertEqual(yaml_payload["mode"], "tool:resource_detail")
        self.assertIn("```yaml", yaml_payload["answer"])
        self.assertEqual(yaml_payload["items"][0]["name"], self.LIVE_POD_NAME)
        self.assertEqual(yaml_payload["artifacts"][0]["artifact_type"], "resource_editor")

        implicit_yaml_response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": f"{self.LIVE_POD_NAME} yaml 보여줘",
                "namespace": "demo",
            },
        )
        self.assertEqual(implicit_yaml_response.status_code, 200)
        implicit_yaml_payload = implicit_yaml_response.json()
        self.assertEqual(implicit_yaml_payload["mode"], "tool:resource_detail")
        self.assertIn("```yaml", implicit_yaml_payload["answer"])
        self.assertEqual(implicit_yaml_payload["items"][0]["name"], self.LIVE_POD_NAME)

    def test_live_chat_followup_uses_recent_live_resource_memory(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/api/v1/namespaces/demo/pods"):
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "metadata": {"name": self.LIVE_POD_NAME, "namespace": "demo"},
                                "status": {"phase": "Running"},
                                "spec": {"nodeName": "node-1"},
                            }
                        ]
                    },
                )
            if path.endswith(f"/api/v1/namespaces/demo/pods/{self.LIVE_POD_NAME}"):
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "v1",
                        "kind": "Pod",
                        "metadata": {"name": self.LIVE_POD_NAME, "namespace": "demo"},
                        "spec": {"nodeName": "node-1"},
                        "status": {"phase": "Running"},
                    },
                )
            if path.endswith("/api/v1/namespaces"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "demo"}}]})
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)

        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": "그거 yaml 보여줘",
                "namespace": "",
                "history": [
                    {
                        "role": "assistant",
                        "text": f"demo namespace에서 pods 총 1개입니다. 예시: {self.LIVE_POD_NAME}",
                        "lane": "live",
                        "resourceNames": [self.LIVE_POD_NAME],
                        "namespace": "demo",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "tool:resource_detail")
        self.assertEqual(payload["namespace"], "demo")
        self.assertIn("```yaml", payload["answer"])
        self.assertEqual(payload["items"][0]["name"], self.LIVE_POD_NAME)

    def test_live_chat_routes_pod_event_question_to_events(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/api/v1/namespaces/demo/events"):
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "metadata": {"name": "event-a", "namespace": "demo"},
                                "type": "Warning",
                                "reason": "BackOff",
                                "involvedObject": {"name": self.LIVE_POD_NAME, "kind": "Pod"},
                            }
                        ]
                    },
                )
            if path.endswith("/api/v1/namespaces"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "demo"}}]})
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)

        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": "pod의 이벤트 보여줘",
                "namespace": "demo",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "tool:resource_list")
        self.assertEqual(payload["resource"], "events")
        self.assertIn("event", payload["answer"].lower())

    def test_live_chat_answers_namespace_question(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/v1/namespaces"):
                return httpx.Response(
                    200,
                    json={"items": [{"metadata": {"name": "demo"}}, {"metadata": {"name": "openshift-monitoring"}}]},
                )
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)

        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": "namespaces 알려줘",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "tool:namespaces")
        self.assertIn("namespace", payload["answer"])

    def test_live_chat_intent_switch_ignores_previous_live_resource_memory(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/v1/namespaces"):
                return httpx.Response(
                    200,
                    json={"items": [{"metadata": {"name": "demo"}}, {"metadata": {"name": "openshift-monitoring"}}]},
                )
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)

        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": "namespaces 알려줘",
                "namespace": "",
                "history": [
                    {
                        "role": "assistant",
                        "text": f"demo namespace에서 pods 총 1개입니다. 예시: {self.LIVE_POD_NAME}",
                        "lane": "live",
                        "resourceNames": [self.LIVE_POD_NAME],
                        "namespace": "demo",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "tool:namespaces")
        self.assertIn("namespace", payload["answer"])

    def test_live_chat_compares_live_yaml_with_official_docs(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/apis/apps/v1/namespaces/demo/deployments/web"):
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": "web", "namespace": "demo"},
                        "spec": {
                            "replicas": 2,
                            "template": {
                                "spec": {
                                    "containers": [{"name": "web", "image": "nginx:1.27"}],
                                }
                            },
                        },
                    },
                )
            if path.endswith("/apis/apps/v1/namespaces/demo/deployments"):
                return httpx.Response(
                    200,
                    json={"items": [{"metadata": {"name": "web", "namespace": "demo"}, "spec": {"replicas": 2}, "status": {}}]},
                )
            if path.endswith("/api/v1/namespaces"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "demo"}}]})
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)
        live_ocp_chat_service.document_retriever = _FakeDocumentRetriever()

        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        response = self.client.post(
            "/api/v1/chat/live",
            json={
                "connection_id": connection_id,
                "message": "deployment web 공식 문서랑 지금 live yaml 차이 알려줘",
                "namespace": "demo",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "tool:doc_compare")
        self.assertIn("관련 공식 문서 근거", payload["answer"])
        self.assertGreaterEqual(len(payload["sources"]), 2)


if __name__ == "__main__":
    unittest.main()


