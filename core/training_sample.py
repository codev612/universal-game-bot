from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from loguru import logger

ActionType = Literal["tap", "swipe", "idle"]


@dataclass(slots=True)
class ActionRecord:
    type: ActionType
    position: Optional[Tuple[int, int]] = None
    start: Optional[Tuple[int, int]] = None
    end: Optional[Tuple[int, int]] = None
    duration_ms: Optional[int] = None


@dataclass(slots=True)
class TrainingSample:
    screenshot_path: str
    game: Optional[str]
    scenario_current: Optional[str]
    scenario_next: Optional[str]
    player_state: Optional[str]
    state_board_values: Dict[str, str]
    action: ActionRecord
    timestamp: str

    @staticmethod
    def timestamp_now() -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class TrainingSampleLogger:
    """Append-only logger for training samples stored as JSON Lines."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, sample: TrainingSample) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(sample), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.error("Unable to append training sample to {}: {}", self._log_path, exc)

