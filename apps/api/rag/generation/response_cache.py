from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from apps.api.schemas.chat import CopilotChatResponse


class ChatResponseCache:
    def __init__(self, *, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._items: OrderedDict[str, CopilotChatResponse] = OrderedDict()

    def build_key(self, payload: dict) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str) -> CopilotChatResponse | None:
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item

    def set(self, key: str, value: CopilotChatResponse) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

