from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger


@dataclass(slots=True, frozen=True)
class GameConfig:
    name: str
    package: str


class GameRegistry(QObject):
    """Persistent registry of known games."""

    games_changed = pyqtSignal(list)

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._storage_path = Path(storage_path) if storage_path else Path("games.json")
        self._games: List[GameConfig] = []
        self._load_games()

    @property
    def games(self) -> List[GameConfig]:
        return list(self._games)

    def add_game(self, name: str, package: str) -> None:
        config = GameConfig(name=name, package=package)
        if config in self._games:
            logger.info("Game already registered name={} package={}", name, package)
            return
        self._games.append(config)
        self._save_games()
        self._emit_change()

    def find_by_name(self, name: str) -> Optional[GameConfig]:
        return next((game for game in self._games if game.name == name), None)

    def remove_game(self, name: str) -> bool:
        original_count = len(self._games)
        self._games = [game for game in self._games if game.name != name]
        if len(self._games) == original_count:
            logger.info("Game not found for removal name={}", name)
            return False

        self._save_games()
        self._emit_change()
        logger.info("Removed game name={}", name)
        return True

    def _emit_change(self) -> None:
        self.games_changed.emit(self.games)

    def _load_games(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse game registry file {}: {}", self._storage_path, exc)
            return
        except OSError as exc:
            logger.error("Unable to read game registry file {}: {}", self._storage_path, exc)
            return

        games: List[GameConfig] = []
        for item in raw or []:
            try:
                games.append(GameConfig(name=item["name"], package=item["package"]))
            except KeyError:
                logger.warning("Skipping invalid game entry: {}", item)

        self._games = games
        self._emit_change()

    def _save_games(self) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(game) for game in self._games]
            self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to write game registry file {}: {}", self._storage_path, exc)

