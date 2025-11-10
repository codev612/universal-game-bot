from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger


class ScenarioRegistry(QObject):
    """Persistent storage for reusable scenario labels."""

    scenarios_changed = pyqtSignal(list)

    DEFAULT_SCENARIOS: List[str] = [
        "Loading",
        "Auto Questing",
        "Purchasing",
        "Teleporting",
        "Boss Fight",
        "Idle",
    ]

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._storage_path = Path(storage_path) if storage_path else Path("scenarios.json")
        self._scenarios: List[str] = []
        self._load()
        if not self._scenarios:
            self._scenarios = list(self.DEFAULT_SCENARIOS)
        else:
            self._scenarios.sort(key=str.lower)

    def scenarios(self) -> List[str]:
        return list(self._scenarios)

    def add_scenario(self, name: str) -> bool:
        trimmed = name.strip()
        if not trimmed:
            return False

        lower = trimmed.lower()
        if any(existing.lower() == lower for existing in self._scenarios):
            return False

        self._scenarios.append(trimmed)
        self._scenarios.sort(key=str.lower)
        self._persist()
        self._emit_change()
        logger.debug("Added scenario '{}'", trimmed)
        return True

    def remove_scenario(self, name: str) -> bool:
        trimmed = name.strip()
        if not trimmed:
            return False

        lower = trimmed.lower()
        filtered = [scenario for scenario in self._scenarios if scenario.lower() != lower]
        if len(filtered) == len(self._scenarios):
            return False

        self._scenarios = filtered
        self._persist()
        self._emit_change()
        logger.debug("Removed scenario '{}'", trimmed)
        return True

    def replace_all(self, scenarios: List[str]) -> None:
        cleaned: List[str] = []
        seen = set()
        for scenario in scenarios:
            trimmed = scenario.strip()
            if not trimmed:
                continue
            lower = trimmed.lower()
            if lower in seen:
                continue
            cleaned.append(trimmed)
            seen.add(lower)

        self._scenarios = sorted(cleaned, key=str.lower)
        self._persist()
        self._emit_change()
        logger.debug("Scenario list replaced with {} entries", len(self._scenarios))

    def reset_to_defaults(self) -> None:
        self._scenarios = list(self.DEFAULT_SCENARIOS)
        self._persist()
        self._emit_change()
        logger.debug("Scenario list reset to defaults")

    def _emit_change(self) -> None:
        self.scenarios_changed.emit(self.scenarios())

    def _load(self) -> None:
        if not self._storage_path.exists():
            logger.debug("Scenario registry file not found; using defaults")
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse scenario registry file {}: {}", self._storage_path, exc)
            return
        except OSError as exc:
            logger.error("Unable to read scenario registry file {}: {}", self._storage_path, exc)
            return

        if not isinstance(raw, list):
            logger.error("Invalid scenario registry format; expected list at root")
            return

        cleaned: List[str] = []
        seen = set()
        for entry in raw:
            if not isinstance(entry, str):
                continue
            trimmed = entry.strip()
            if not trimmed:
                continue
            lower = trimmed.lower()
            if lower in seen:
                continue
            cleaned.append(trimmed)
            seen.add(lower)

        if cleaned:
            self._scenarios = cleaned
            self._scenarios.sort(key=str.lower)
        logger.debug("Loaded {} scenarios from disk", len(self._scenarios))

    def _persist(self) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._scenarios
            self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to write scenario registry file {}: {}", self._storage_path, exc)

