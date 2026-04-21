from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from apps.api.runtime import scm_connection_repository, scm_oauth_service, workspace_repository

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.post("/{provider}/start")
async def start_scm_oauth(provider: str, request: Request, workspace_id: str = Query(...)) -> dict[str, str]:
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider not in {"github", "gitlab"}:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider.")
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    callback_url = str(request.url_for("handle_scm_oauth_callback", provider=normalized_provider))
    try:
        authorize_url, state = scm_oauth_service.build_authorize_url(
            provider=normalized_provider,  # type: ignore[arg-type]
            workspace_id=workspace_id,
            callback_url=callback_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"provider": normalized_provider, "authorize_url": authorize_url, "state": state}


@router.get("/{provider}/callback", name="handle_scm_oauth_callback")
async def handle_scm_oauth_callback(
    provider: str,
    request: Request,
    code: str = Query(""),
    state: str = Query(""),
) -> RedirectResponse:
    normalized_provider = str(provider or "").strip().lower()
    frontend_base = f"{request.base_url}scm"
    if normalized_provider not in {"github", "gitlab"}:
        return RedirectResponse(url=f"{frontend_base}?oauth_status=error&message=unsupported-provider", status_code=303)
    if not code or not state:
        return RedirectResponse(url=f"{frontend_base}?oauth_status=error&message=missing-code-or-state", status_code=303)
    callback_url = str(request.url_for("handle_scm_oauth_callback", provider=normalized_provider))
    try:
        connection = scm_oauth_service.complete_callback(
            provider=normalized_provider,  # type: ignore[arg-type]
            code=code,
            state=state,
            callback_url=callback_url,
            connection_repository=scm_connection_repository,
        )
        return RedirectResponse(
            url=f"{frontend_base}?oauth_status=connected&provider={normalized_provider}&connection_id={connection.scm_connection_id}",
            status_code=303,
        )
    except Exception as exc:
        return RedirectResponse(url=f"{frontend_base}?oauth_status=error&message={str(exc)}", status_code=303)
