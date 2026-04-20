from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from apps.api.api.schemas.auth import OcpConnectionProfile, OcpLeaseSchedulerStatusResponse
from apps.api.integrations.ocp.auth.alert_notifier import WebhookAlertNotifier, build_default_alert_notifier
from apps.api.integrations.ocp.auth.broker import OcpConnectionBroker


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class OcpLeaseSchedulerService:
    def __init__(
        self,
        *,
        broker: OcpConnectionBroker,
        interval_seconds: int = 300,
        max_backoff_seconds: int = 3600,
        alert_notifier: WebhookAlertNotifier | None = None,
        alert_cooldown_seconds: int = 900,
        enabled: bool = False,
    ) -> None:
        self.broker = broker
        self.interval_seconds = max(int(interval_seconds), 30)
        self.max_backoff_seconds = max(int(max_backoff_seconds), self.interval_seconds)
        self.alert_notifier = alert_notifier
        self.alert_cooldown_seconds = max(int(alert_cooldown_seconds), 60)
        self.enabled = bool(enabled)
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._last_run_at = ""
        self._last_success_at = ""
        self._last_failure_at = ""
        self._last_error = ""
        self._consecutive_failures = 0
        self._next_run_delay_seconds = self.interval_seconds
        self._profiles_checked = 0
        self._renewals_applied = 0
        self._recent_failures: list[str] = []
        self._alert_dispatch_count = 0
        self._last_alert_at = ""
        self._last_alert_level = "none"
        self._last_alert_status = ""
        self._last_alert_error = ""

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name="ocp-lease-scheduler")

    async def stop(self) -> None:
        if self._task is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop_event = None

    async def run_once(self) -> OcpLeaseSchedulerStatusResponse:
        profiles = self._list_profiles()
        renewals_applied = 0
        self._profiles_checked = len(profiles)
        errors: list[str] = []

        for profile in profiles:
            try:
                before = dict(profile.metadata)
                secret_status = self.broker.describe_secret(profile, refresh=True, auto_renew=True)
                if secret_status:
                    updated_profile = profile.model_copy(
                        update={
                            "metadata": {
                                **profile.metadata,
                                "secret_backend": str(secret_status.get("secret_backend") or ""),
                                "secret_version": str(secret_status.get("secret_version") or ""),
                                "secret_created_at": str(secret_status.get("secret_created_at") or ""),
                                "secret_lease_renewable": bool(secret_status.get("lease_renewable") or False),
                                "secret_lease_ttl_seconds": int(secret_status.get("lease_ttl_seconds") or 0),
                                "secret_lease_expires_at": str(secret_status.get("lease_expires_at") or ""),
                                "secret_rotation_supported": bool(secret_status.get("rotation_supported") or False),
                                "secret_auto_renew_applied": bool(secret_status.get("auto_renew_applied") or False),
                                "secret_auto_renew_threshold_seconds": int(secret_status.get("auto_renew_threshold_seconds") or 0),
                                "secret_renew_message": str(secret_status.get("renew_message") or ""),
                            }
                        }
                    )
                    self.broker.profile_store.put(updated_profile)
                    if secret_status.get("auto_renew_applied") or before.get("secret_lease_ttl_seconds") != updated_profile.metadata.get("secret_lease_ttl_seconds"):
                        renewals_applied += 1
            except Exception as exc:
                now = _utc_now_iso()
                error_message = f"{profile.connection_id}: {exc}"
                errors.append(error_message)
                updated_profile = profile.model_copy(
                    update={
                        "metadata": {
                            **profile.metadata,
                            "secret_last_renew_error": str(exc),
                            "secret_last_renew_failed_at": now,
                        }
                    }
                )
                self.broker.profile_store.put(updated_profile)

        self._renewals_applied = renewals_applied
        self._last_run_at = _utc_now_iso()
        if errors:
            self._consecutive_failures += 1
            self._last_failure_at = self._last_run_at
            self._last_error = errors[0]
            self._recent_failures = [*self._recent_failures, *errors][-8:]
            multiplier = 2 ** max(self._consecutive_failures - 1, 0)
            self._next_run_delay_seconds = min(self.interval_seconds * multiplier, self.max_backoff_seconds)
        else:
            self._consecutive_failures = 0
            self._last_success_at = self._last_run_at
            self._last_error = ""
            self._next_run_delay_seconds = self.interval_seconds
        await self._dispatch_alert_if_needed()
        return self.status()

    def status(self) -> OcpLeaseSchedulerStatusResponse:
        return OcpLeaseSchedulerStatusResponse(
            enabled=self.enabled,
            running=self._task is not None and not self._task.done(),
            interval_seconds=self.interval_seconds,
            last_run_at=self._last_run_at,
            last_success_at=self._last_success_at,
            last_failure_at=self._last_failure_at,
            last_error=self._last_error,
            consecutive_failures=self._consecutive_failures,
            next_run_delay_seconds=self._next_run_delay_seconds,
            alert_level=self._alert_level(),
            profiles_checked=self._profiles_checked,
            renewals_applied=self._renewals_applied,
            recent_failures=list(self._recent_failures),
            alert_delivery_enabled=self.alert_notifier is not None,
            alert_target=self._alert_target(),
            alert_dispatch_count=self._alert_dispatch_count,
            last_alert_at=self._last_alert_at,
            last_alert_level=self._last_alert_level,
            last_alert_status=self._last_alert_status,
            last_alert_error=self._last_alert_error,
        )

    async def _run_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._next_run_delay_seconds)
            except asyncio.TimeoutError:
                continue

    def _list_profiles(self) -> list[OcpConnectionProfile]:
        if hasattr(self.broker.profile_store, "list_profiles"):
            return list(self.broker.profile_store.list_profiles())
        return []

    def _alert_level(self) -> str:
        if self._consecutive_failures >= 3:
            return "critical"
        if self._consecutive_failures >= 1:
            return "warning"
        return "none"

    def _alert_target(self) -> str:
        if self.alert_notifier is None:
            return ""
        return self.alert_notifier.webhook_url

    async def _dispatch_alert_if_needed(self) -> None:
        current_level = self._alert_level()
        if self.alert_notifier is None or current_level == "none":
            return
        if not self._should_dispatch_alert(current_level):
            return
        payload = {
            "event": "lease_scheduler_alert",
            "alert_level": current_level,
            "last_error": self._last_error,
            "consecutive_failures": self._consecutive_failures,
            "next_run_delay_seconds": self._next_run_delay_seconds,
            "recent_failures": self._recent_failures,
            "last_run_at": self._last_run_at,
        }
        result = await self.alert_notifier.send(payload)
        self._last_alert_at = _utc_now_iso()
        self._last_alert_level = current_level
        self._last_alert_status = "delivered" if result.delivered else "failed"
        self._last_alert_error = "" if result.delivered else result.message
        if result.delivered:
            self._alert_dispatch_count += 1

    def _should_dispatch_alert(self, current_level: str) -> bool:
        if self._last_alert_level != current_level:
            return True
        if not self._last_alert_at:
            return True
        try:
            last_alert = datetime.fromisoformat(self._last_alert_at)
        except ValueError:
            return True
        now = datetime.now(timezone.utc).replace(microsecond=0)
        elapsed = int((now - last_alert).total_seconds())
        return elapsed >= self.alert_cooldown_seconds


def build_default_lease_scheduler(*, broker: OcpConnectionBroker) -> OcpLeaseSchedulerService:
    backend = str(os.environ.get("RAG_TASK_SECRET_BACKEND") or "").strip().casefold()
    enabled = backend in {"vault_hashicorp", "vault_http"} and str(os.environ.get("RAG_TASK_VAULT_BACKGROUND_RENEW", "1")).strip().casefold() not in {"0", "false", "no", "off"}
    interval_seconds = int(os.environ.get("RAG_TASK_VAULT_BACKGROUND_RENEW_INTERVAL_SECONDS") or 300)
    max_backoff_seconds = int(os.environ.get("RAG_TASK_VAULT_BACKGROUND_RENEW_MAX_BACKOFF_SECONDS") or 3600)
    alert_cooldown_seconds = int(os.environ.get("RAG_TASK_ALERT_WEBHOOK_COOLDOWN_SECONDS") or 900)
    return OcpLeaseSchedulerService(
        broker=broker,
        interval_seconds=interval_seconds,
        max_backoff_seconds=max_backoff_seconds,
        alert_notifier=build_default_alert_notifier(),
        alert_cooldown_seconds=alert_cooldown_seconds,
        enabled=enabled,
    )

