from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.include_router(router)
app.mount("/static", StaticFiles(directory="app/web"), name="static")
app.mount("/resources", StaticFiles(directory="app/resources"), name="resources")
