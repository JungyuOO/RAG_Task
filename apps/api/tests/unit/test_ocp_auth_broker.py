from __future__ import annotations

import unittest

from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest
from apps.api.ocp.auth import InMemoryConnectionSecretStore, OcpConnectionBroker


class OcpAuthBrokerTests(unittest.TestCase):
    def test_token_request_normalizes_url_and_requires_token(self) -> None:
        request = OcpConnectionRequest(
            cluster_url="https://api.cluster.example.com/",
            auth_mode=OcpAuthMode.TOKEN,
            token="sha256~abc",
        )
        self.assertEqual(request.cluster_url, "https://api.cluster.example.com")

    def test_password_request_requires_username_and_password(self) -> None:
        request = OcpConnectionRequest(
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.PASSWORD,
            username="admin",
            password="secret",
        )
        self.assertEqual(request.username, "admin")

    def test_broker_profile_hides_secrets_and_builds_runtime_config(self) -> None:
        store = InMemoryConnectionSecretStore()
        broker = OcpConnectionBroker(secret_store=store)
        request = OcpConnectionRequest(
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            token="sha256~abc",
            default_namespace="demo",
            display_name="demo-cluster",
        )

        profile = broker.create_profile(request)
        runtime = broker.build_runtime_config(profile)
        secret_status = broker.describe_secret(profile)

        self.assertEqual(profile.display_name, "demo-cluster")
        self.assertNotIn("token", profile.model_dump())
        self.assertEqual(runtime["token"], "sha256~abc")
        self.assertFalse(runtime["exchange_required"])
        self.assertEqual(secret_status["secret_backend"], "protected_file")

    def test_password_mode_marks_exchange_required(self) -> None:
        broker = OcpConnectionBroker()
        request = OcpConnectionRequest(
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.PASSWORD,
            username="developer",
            password="secret",
        )

        profile = broker.create_profile(request)
        runtime = broker.build_runtime_config(profile)

        self.assertTrue(runtime["exchange_required"])
        self.assertEqual(runtime["username"], "developer")
        self.assertEqual(runtime["password"], "secret")

    def test_disconnect_removes_secret(self) -> None:
        store = InMemoryConnectionSecretStore()
        broker = OcpConnectionBroker(secret_store=store)
        request = OcpConnectionRequest(
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            token="sha256~abc",
        )
        profile = broker.create_profile(request)

        self.assertIsNotNone(store.get(profile.secret_ref))
        broker.disconnect(profile)
        self.assertIsNone(store.get(profile.secret_ref))

    def test_broker_can_lookup_profile_by_connection_id(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
            )
        )

        looked_up = broker.get_profile(profile.connection_id)
        self.assertIsNotNone(looked_up)
        assert looked_up is not None
        self.assertEqual(looked_up.connection_id, profile.connection_id)


if __name__ == "__main__":
    unittest.main()




