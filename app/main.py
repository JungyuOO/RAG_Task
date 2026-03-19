from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.dependencies import build_container


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.state.container = build_container(settings)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="app/web"), name="static")
    app.mount("/resources", StaticFiles(directory="app/resources"), name="resources")
    return app


app = create_app()
