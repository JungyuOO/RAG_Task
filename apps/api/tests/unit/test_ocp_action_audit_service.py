from __future__ import annotations

import unittest

from apps.api.schemas.ocp_action_audit import OcpActionAuditEventType
from apps.api.schemas.ocp_actions import OcpActionType
from apps.api.ocp.action_audit_service import OcpActionAuditService


class OcpActionAuditServiceTests(unittest.TestCase):
    def test_log_and_list_recent(self) -> None:
        service = OcpActionAuditService()
        record = service.log(
            event_type=OcpActionAuditEventType.REQUEST_CREATED,
            actor_id="ui-local",
            request_id="req-1",
            execution_id="",
            action_type=OcpActionType.SCALE_DEPLOYMENT,
            namespace="demo",
            resource_name="demo-app",
            risk_level="medium",
            details={"summary": "created"},
        )
        self.assertEqual(record.actor_id, "ui-local")

        listed = service.list_recent(limit=10)
        self.assertGreaterEqual(len(listed.items), 1)


if __name__ == "__main__":
    unittest.main()




