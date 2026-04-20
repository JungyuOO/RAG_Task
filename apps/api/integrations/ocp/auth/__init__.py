"""Authentication and connection helpers for OCP access."""

from apps.api.integrations.ocp.auth.alert_notifier import AlertDispatchResult, WebhookAlertNotifier, build_default_alert_notifier
from apps.api.integrations.ocp.auth.broker import (
    InMemoryConnectionProfileStore,
    InMemoryConnectionSecretStore,
    OcpConnectionBroker,
    build_default_connection_secret_store,
)
from apps.api.integrations.ocp.auth.lease_scheduler import OcpLeaseSchedulerService, build_default_lease_scheduler
from apps.api.integrations.ocp.auth.secret_store_types import StoredConnectionSecret
from apps.api.integrations.ocp.auth.verifier import OcpConnectionVerifier
from apps.api.integrations.ocp.auth.vault_store import (
    HashiCorpVaultKvV2SecretClient,
    VaultHttpConnectionSecretStore,
    VaultHttpSecretClient,
)

__all__ = [
    "AlertDispatchResult",
    "HashiCorpVaultKvV2SecretClient",
    "InMemoryConnectionProfileStore",
    "InMemoryConnectionSecretStore",
    "OcpConnectionBroker",
    "OcpConnectionVerifier",
    "OcpLeaseSchedulerService",
    "StoredConnectionSecret",
    "VaultHttpConnectionSecretStore",
    "VaultHttpSecretClient",
    "WebhookAlertNotifier",
    "build_default_alert_notifier",
    "build_default_connection_secret_store",
    "build_default_lease_scheduler",
]
