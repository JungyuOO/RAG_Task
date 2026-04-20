from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from apps.api.api.schemas.chat import CopilotChatRequest, CopilotChatResponse, CopilotChatStage, OcpLiveChatRequest, OcpLiveChatResponse
from apps.api.runtime import connection_broker, live_ocp_chat_service, unified_copilot_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=CopilotChatResponse)
async def query_copilot(request: CopilotChatRequest) -> CopilotChatResponse:
    try:
        return await unified_copilot_service.answer(
            message=request.message,
            connection_id=request.connection_id,
            namespace=request.namespace,
            recent_turns=request.history,
            broker=connection_broker,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/query/stream")
async def query_copilot_stream(request: CopilotChatRequest) -> StreamingResponse:
    queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

    async def on_progress(stage: CopilotChatStage) -> None:
        await queue.put({"type": "stage", "stage": stage.model_dump(mode="json")})

    async def on_answer_delta(delta: str) -> None:
        await queue.put({"type": "answer_delta", "delta": delta})

    async def run_answer() -> None:
        try:
            response = await unified_copilot_service.answer(
                message=request.message,
                connection_id=request.connection_id,
                namespace=request.namespace,
                recent_turns=request.history,
                broker=connection_broker,
                progress=on_progress,
                answer_delta=on_answer_delta,
            )
            await queue.put({"type": "result", "response": response.model_dump(mode="json")})
        except LookupError as exc:
            await queue.put({"type": "error", "status_code": 404, "message": str(exc)})
        except ValueError as exc:
            await queue.put({"type": "error", "status_code": 400, "message": str(exc)})
        except Exception as exc:
            await queue.put({"type": "error", "status_code": 502, "message": str(exc)})
        finally:
            await queue.put(None)

    async def stream():
        task = asyncio.create_task(run_answer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"{json.dumps(item, ensure_ascii=False)}\n"
        finally:
            await task

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/live", response_model=OcpLiveChatResponse)
async def live_ocp_chat(request: OcpLiveChatRequest) -> OcpLiveChatResponse:
    try:
        return await live_ocp_chat_service.answer(
            connection_id=request.connection_id,
            message=request.message,
            namespace=request.namespace,
            broker=connection_broker,
            recent_turns=request.history,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
