from __future__ import annotations

from dataclasses import dataclass, field

from apps.api.api.schemas.chat import CopilotChatHistoryTurn


@dataclass(slots=True)
class ChatMemoryState:
    last_lane: str = ""
    last_doc_source_paths: list[str] = field(default_factory=list)
    last_live_namespace: str = ""
    last_live_resource_names: list[str] = field(default_factory=list)
    last_user_message: str = ""
    last_assistant_message: str = ""


def build_chat_memory(recent_turns: list[CopilotChatHistoryTurn]) -> ChatMemoryState:
    state = ChatMemoryState()
    for turn in reversed(recent_turns):
        if not state.last_user_message and turn.role == "user" and str(turn.text or "").strip():
            state.last_user_message = str(turn.text or "").strip()
        if turn.role != "assistant":
            continue
        if not state.last_assistant_message and str(turn.text or "").strip():
            state.last_assistant_message = str(turn.text or "").strip()
        if not state.last_lane and str(turn.lane or "").strip():
            state.last_lane = str(turn.lane or "").strip()
        if not state.last_doc_source_paths and getattr(turn, "source_paths", None):
            state.last_doc_source_paths = [path for path in turn.source_paths if str(path or "").strip()]
        if not state.last_live_namespace and str(getattr(turn, "namespace", "") or "").strip():
            state.last_live_namespace = str(turn.namespace or "").strip()
        if not state.last_live_resource_names and getattr(turn, "resource_names", None):
            state.last_live_resource_names = [name for name in turn.resource_names if str(name or "").strip()]
        if (
            state.last_lane
            and state.last_assistant_message
            and (state.last_doc_source_paths or state.last_live_namespace or state.last_live_resource_names)
        ):
            break
    return state
