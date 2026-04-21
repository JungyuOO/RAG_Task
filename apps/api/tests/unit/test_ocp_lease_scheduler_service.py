from __future__ import annotations

import unittest

from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionProfile
from apps.api.ocp.auth.alert_notifier import AlertDispatchResult
from apps.api.ocp.auth.lease_scheduler import OcpLeaseSchedulerService


class _FakeSecretStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def describe(
        self,
        secret_ref: str,
        *,
        refresh: bool = False,
        renew: bool = False,
        auto_renew: bool = False,
        renew_threshold_seconds: int = 0,
        renew_increment: str = "",
    ) -> dict:
        self.calls.append(
            {
                "secret_ref": secret_ref,
                "refresh": refresh,
                "renew": renew,
                "auto_renew": auto_renew,
                "renew_threshold_seconds": renew_threshold_seconds,
                "renew_increment": renew_increment,
            }
        )
        return {
            "secret_backend": "vault_hashicorp",
            "lease_renewable": True,
            "lease_ttl_seconds": 300,
            "lease_expires_at": "2026-04-14T12:00:00Z",
            "auto_renew_applied": True,
            "auto_renew_threshold_seconds": renew_threshold_seconds,
            "renew_message": "auto renewed",
        }


class _FakeProfileStore:
    def __init__(self, profiles: list[OcpConnectionProfile]) -> None:
        self._profiles = {profile.connection_id: profile for profile in profiles}

    def list_profiles(self) -> list[OcpConnectionProfile]:
        return list(self._profiles.values())

    def put(self, profile: OcpConnectionProfile) -> None:
        self._profiles[profile.connection_id] = profile


class _FakeBroker:
    def __init__(self, profiles: list[OcpConnectionProfile]) -> None:
        self.secret_store = _FakeSecretStore()
        self.profile_store = _FakeProfileStore(profiles)

    def describe_secret(self, profile: OcpConnectionProfile, *, refresh: bool = False, renew: bool = False, auto_renew: bool = False):
        return self.secret_store.describe(
            profile.secret_ref,
            refresh=refresh,
            renew=renew,
            auto_renew=auto_renew,
            renew_threshold_seconds=600,
            renew_increment="",
        )


class _FailingBroker(_FakeBroker):
    def describe_secret(self, profile: OcpConnectionProfile, *, refresh: bool = False, renew: bool = False, auto_renew: bool = False):
        raise RuntimeError(f"lease refresh failed for {profile.connection_id}")


class _FakeAlertNotifier:
    def __init__(self) -> None:
        self.webhook_url = "https://alerts.example.com/webhook"
        self.payloads: list[dict] = []

    async def send(self, payload: dict) -> AlertDispatchResult:
        self.payloads.append(payload)
        return AlertDispatchResult(delivered=True, message="ok", status_code=200)


class OcpLeaseSchedulerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_updates_profile_metadata_and_status(self) -> None:
        profile = OcpConnectionProfile(
            connection_id="ocp-conn-1",
            display_name="demo",
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            verify_ssl=True,
            default_namespace="demo",
            username_hint="alice",
            secret_ref="ocp-secret-1",
            metadata={},
        )
        broker = _FakeBroker([profile])
        service = OcpLeaseSchedulerService(broker=broker, interval_seconds=300, enabled=True)

        result = await service.run_once()

        self.assertTrue(result.enabled)
        self.assertEqual(result.profiles_checked, 1)
        self.assertEqual(result.renewals_applied, 1)
        updated = broker.profile_store.list_profiles()[0]
        self.assertTrue(updated.metadata["secret_auto_renew_applied"])
        self.assertEqual(updated.metadata["secret_lease_ttl_seconds"], 300)

    async def test_run_once_tracks_failures_and_backoff(self) -> None:
        profile = OcpConnectionProfile(
            connection_id="ocp-conn-fail",
            display_name="demo",
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            verify_ssl=True,
            default_namespace="demo",
            username_hint="alice",
            secret_ref="ocp-secret-fail",
            metadata={},
        )
        broker = _FailingBroker([profile])
        service = OcpLeaseSchedulerService(broker=broker, interval_seconds=300, max_backoff_seconds=1200, enabled=True)

        first = await service.run_once()
        self.assertEqual(first.alert_level, "warning")
        self.assertEqual(first.consecutive_failures, 1)
        self.assertEqual(first.next_run_delay_seconds, 300)
        self.assertEqual(first.recent_failures[-1], "ocp-conn-fail: lease refresh failed for ocp-conn-fail")

        second = await service.run_once()
        self.assertEqual(second.consecutive_failures, 2)
        self.assertEqual(second.next_run_delay_seconds, 600)
        self.assertEqual(second.alert_level, "warning")

        third = await service.run_once()
        self.assertEqual(third.consecutive_failures, 3)
        self.assertEqual(third.next_run_delay_seconds, 1200)
        self.assertEqual(third.alert_level, "critical")

    async def test_run_once_dispatches_webhook_alert_on_failure(self) -> None:
        profile = OcpConnectionProfile(
            connection_id="ocp-conn-alert",
            display_name="demo",
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            verify_ssl=True,
            default_namespace="demo",
            username_hint="alice",
            secret_ref="ocp-secret-alert",
            metadata={},
        )
        broker = _FailingBroker([profile])
        notifier = _FakeAlertNotifier()
        service = OcpLeaseSchedulerService(
            broker=broker,
            interval_seconds=300,
            max_backoff_seconds=1200,
            alert_notifier=notifier,
            alert_cooldown_seconds=60,
            enabled=True,
        )

        result = await service.run_once()

        self.assertEqual(result.alert_level, "warning")
        self.assertTrue(result.alert_delivery_enabled)
        self.assertEqual(result.alert_dispatch_count, 1)
        self.assertEqual(result.last_alert_status, "delivered")
        self.assertEqual(len(notifier.payloads), 1)

    async def test_run_once_tracks_alert_delivery_failure(self) -> None:
        class _FailingAlertNotifier(_FakeAlertNotifier):
            async def send(self, payload: dict) -> AlertDispatchResult:
                self.payloads.append(payload)
                return AlertDispatchResult(delivered=False, message="webhook 500", status_code=500)

        profile = OcpConnectionProfile(
            connection_id="ocp-conn-alert-fail",
            display_name="demo",
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            verify_ssl=True,
            default_namespace="demo",
            username_hint="alice",
            secret_ref="ocp-secret-alert-fail",
            metadata={},
        )
        broker = _FailingBroker([profile])
        notifier = _FailingAlertNotifier()
        service = OcpLeaseSchedulerService(
            broker=broker,
            interval_seconds=300,
            max_backoff_seconds=1200,
            alert_notifier=notifier,
            alert_cooldown_seconds=60,
            enabled=True,
        )

        result = await service.run_once()

        self.assertEqual(result.last_alert_status, "failed")
        self.assertEqual(result.last_alert_error, "webhook 500")


if __name__ == "__main__":
    unittest.main()





