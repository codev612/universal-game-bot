from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass(slots=True)
class RegionConfig:
    name: str
    category: str
    x: int
    y: int
    width: int
    height: int
    value_format: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, object]) -> RegionConfig:
        return RegionConfig(
            name=str(data["name"]),
            category=str(data["category"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
            value_format=str(data["value_format"]) if data.get("value_format") not in {None, ""} else None,
        )


class LayoutRegistry:
    """
    Persistent storage for on-screen control regions keyed by game name.

    Layouts are stored as a mapping:
        game_name -> list[RegionConfig]
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else Path("layouts.json")
        self._regions: Dict[str, List[RegionConfig]] = {}
        self._load()

    def regions_for(self, game_name: str) -> List[RegionConfig]:
        return list(self._regions.get(game_name, []))

    def save_regions(self, game_name: str, regions: List[RegionConfig]) -> None:
        self._regions[game_name] = list(regions)
        self._persist()

    def clear_regions(self, game_name: str) -> None:
        if game_name in self._regions:
            del self._regions[game_name]
            self._persist()

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse layout registry file {}: {}", self._storage_path, exc)
            return
        except OSError as exc:
            logger.error("Unable to read layout registry file {}: {}", self._storage_path, exc)
            return

        if not isinstance(raw, dict):
            logger.error("Invalid layout registry format; expected object at root.")
            return

        for key, value in raw.items():
            if isinstance(value, dict):
                # Backward compatibility: old format keyed by device -> game -> regions
                for game_name, regions in value.items():
                    parsed: List[RegionConfig] = []
                    for entry in regions or []:
                        try:
                            parsed.append(RegionConfig.from_dict(entry))
                        except (KeyError, ValueError) as exc:
                            logger.warning("Skipping invalid layout entry for game {}: {}", game_name, exc)
                    if parsed:
                        self._regions.setdefault(game_name, []).extend(parsed)
            else:
                game_name = key
                regions = value or []
                parsed = []
                for entry in regions:
                    try:
                        parsed.append(RegionConfig.from_dict(entry))
                    except (KeyError, ValueError) as exc:
                        logger.warning("Skipping invalid layout entry for game {}: {}", game_name, exc)
                if parsed:
                    self._regions[game_name] = parsed

    def _persist(self) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            serialised = {
                game_name: [region.to_dict() for region in regions]
                for game_name, regions in self._regions.items()
            }
            self._storage_path.write_text(json.dumps(serialised, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to write layout registry file {}: {}", self._storage_path, exc)

