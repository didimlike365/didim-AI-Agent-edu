from __future__ import annotations

from typing import Any


class TriageStateService:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._states.get(thread_id)

    def set(self, thread_id: str, state: dict[str, Any]) -> None:
        self._states[thread_id] = state

    def clear(self, thread_id: str) -> None:
        self._states.pop(thread_id, None)


triage_state_service = TriageStateService()
