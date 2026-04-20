from __future__ import annotations

from apps.api.app_factory import create_app
from apps.api.core.logging import configure_logging

configure_logging()

app = create_app()

