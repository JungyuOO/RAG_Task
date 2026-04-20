from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class AlertDispatchResult:
    delivered: bool
    message: str = ""
    status_code: int = 0


class WebhookAlertNotifier:
    def __init__(
        self,
        *,
        webhook_url: str,
        token: str = "",
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.token = token
        self.transport = transport
        self.timeout = timeout

    async def send(self, payload: dict) -> AlertDispatchResult:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(self.webhook_url, headers=headers, json=payload)
            if response.status_code >= 400:
                return AlertDispatchResult(
                    delivered=False,
                    message=f"Webhook returned HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            return AlertDispatchResult(
                delivered=True,
                message="Webhook alert delivered.",
                status_code=response.status_code,
            )
        except httpx.HTTPError as exc:
            return AlertDispatchResult(delivered=False, message=str(exc))


def build_default_alert_notifier() -> WebhookAlertNotifier | None:
    webhook_url = str(os.environ.get("RAG_TASK_ALERT_WEBHOOK_URL") or "").strip()
    if not webhook_url:
        return None
    token = str(os.environ.get("RAG_TASK_ALERT_WEBHOOK_TOKEN") or "").strip()
    return WebhookAlertNotifier(webhook_url=webhook_url, token=token)
