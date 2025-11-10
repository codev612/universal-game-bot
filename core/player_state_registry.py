from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PyQt6.QtCore import QObject, pyqtSignal


DEFAULT_PLAYER_STATES = [
    "Low blood",
    "No moving",
    "No progressing",
    "Progressing",
    "Heavy weight",
]


class PlayerStateRegistry(QObject):
    states_changed = pyqtSignal(list)

    def __init__(self, storage_path: Path) -> None:
        super().__init__()
        self._storage_path = Path(storage_path)
        self._states: List[str] = []
        self._load()

    def player_states(self) -> List[str]:
        return list(self._states)

    def add_state(self, state: str) -> None:
        normalized = state.strip()
        if not normalized:
            return
        if normalized in self._states:
            return
        self._states.append(normalized)
        self._states.sort(key=str.lower)
        self._persist()

    def remove_state(self, state: str) -> None:
        try:
            self._states.remove(state)
        except ValueError:
            return
        self._persist()

    def reset_states(self) -> None:
        self._states = list(DEFAULT_PLAYER_STATES)
        self._states.sort(key=str.lower)
        self._persist()

    def _load(self) -> None:
        if not self._storage_path.exists():
            self.reset_states()
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    cleaned = [str(item).strip() for item in data if str(item).strip()]
                    if cleaned:
                        self._states = sorted(set(cleaned), key=str.lower)
                    else:
                        self._states = list(DEFAULT_PLAYER_STATES)
                else:
                    self._states = list(DEFAULT_PLAYER_STATES)
        except (OSError, json.JSONDecodeError):
            self._states = list(DEFAULT_PLAYER_STATES)
        self._states.sort(key=str.lower)
        self.states_changed.emit(self.player_states())

    def _persist(self) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._storage_path.open("w", encoding="utf-8") as handle:
                json.dump(self._states, handle, ensure_ascii=False, indent=2)
        except OSError:
            pass
        self.states_changed.emit(self.player_states())

