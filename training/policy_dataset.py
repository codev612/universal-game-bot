from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from core.training_sample import ActionType

@dataclass(slots=True)
class SampleEntry:
    screenshot_path: Path
    game: Optional[str]
    scenario_current: Optional[str]
    scenario_next: Optional[str]
    player_state: Optional[str]
    state_board_values: Dict[str, str]
    action_type: ActionType
    action_payload: Dict[str, object]


def _parse_numeric(value: str) -> float:
    if not value:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0

    cleaned = value.replace(",", "").strip()

    # Percentages
    if cleaned.endswith("%"):
        try:
            return float(cleaned[:-1]) / 100.0
        except ValueError:
            return 0.0

    # Parentheses removal
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]

    # Ratio a/b
    if "/" in cleaned:
        num, denom = cleaned.split("/", 1)
        try:
            denom_value = float(denom)
            if math.isclose(denom_value, 0.0):
                return float(num)
            return float(num) / denom_value
        except ValueError:
            pass

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class PolicyDataset(Dataset):
    """Dataset backed by training_samples.jsonl for supervised policy learning."""

    def __init__(
        self,
        jsonl_path: Path,
        max_samples: Optional[int] = None,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        if not self._jsonl_path.exists():
            raise FileNotFoundError(f"Training samples file not found: {self._jsonl_path}")

        self.entries: List[SampleEntry] = []
        scenario_names: Dict[str, None] = {}
        state_keys: Dict[str, None] = {}
        player_state_names: Dict[str, None] = {}

        with self._jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                path = Path(payload.get("screenshot_path", ""))
                if not path.exists():
                    continue

                action = payload.get("action") or {}
                action_type = action.get("type", "none")
                if action_type not in {"tap", "swipe"}:
                    action_type = "none"

                entry = SampleEntry(
                    screenshot_path=path,
                    game=payload.get("game"),
                    scenario_current=payload.get("scenario_current"),
                    scenario_next=payload.get("scenario_next"),
                    player_state=payload.get("player_state"),
                    state_board_values=payload.get("state_board_values") or {},
                    action_type=action_type,  # type: ignore[assignment]
                action_payload=action,
                )
                self.entries.append(entry)

                if entry.scenario_current:
                    scenario_names.setdefault(entry.scenario_current, None)
                if entry.scenario_next:
                    scenario_names.setdefault(entry.scenario_next, None)
                if entry.player_state:
                    player_state_names.setdefault(entry.player_state, None)
                for key in entry.state_board_values:
                    state_keys.setdefault(key, None)

                if max_samples is not None and len(self.entries) >= max_samples:
                    break

        if not self.entries:
            raise ValueError("No valid training samples were found.")

        self.scenario_to_index = {"<none>": 0}
        for name in sorted(scenario_names.keys()):
            self.scenario_to_index[name] = len(self.scenario_to_index)

        self.state_keys: List[str] = sorted(state_keys.keys())
        self.player_state_to_index = {"<none>": 0}
        for name in sorted(player_state_names.keys()):
            self.player_state_to_index[name] = len(self.player_state_to_index)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.screenshot_path).convert("RGB")
        width, height = image.size
        np_image = np.asarray(image, dtype=np.float32) / 255.0  # H, W, C
        image_tensor = torch.from_numpy(np_image).permute(2, 0, 1)  # C, H, W

        state_vector = torch.zeros(len(self.state_keys), dtype=torch.float32)
        for i, key in enumerate(self.state_keys):
            state_vector[i] = _parse_numeric(entry.state_board_values.get(key, "0"))

        current_idx = self.scenario_to_index.get(entry.scenario_current, 0)
        next_idx = self.scenario_to_index.get(entry.scenario_next, 0)
        player_state_idx = self.player_state_to_index.get(entry.player_state or "<none>", 0)

        action_type_idx = {"tap": 0, "swipe": 1, "idle": 2}.get(entry.action_type, 2)
        coords = torch.zeros(5, dtype=torch.float32)
        mask = torch.zeros(5, dtype=torch.float32)

        if entry.action_type == "tap":
            pos = entry.action_payload.get("position")
            if isinstance(pos, list) or isinstance(pos, tuple):
                x, y = pos
                coords[0] = float(x) / max(width, 1)
                coords[1] = float(y) / max(height, 1)
                mask[0] = mask[1] = 1.0
        elif entry.action_type == "swipe":
            start = entry.action_payload.get("start")
            end = entry.action_payload.get("end")
            duration = entry.action_payload.get("duration_ms") or 0.0
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)):
                coords[0] = float(start[0]) / max(width, 1)
                coords[1] = float(start[1]) / max(height, 1)
                coords[2] = float(end[0]) / max(width, 1)
                coords[3] = float(end[1]) / max(height, 1)
                coords[4] = float(duration) / 1000.0
                mask[:] = 1.0
        else:
            mask[:] = 0.0

        meta = {
            "path": str(entry.screenshot_path),
            "image_size": (height, width),
            "game": entry.game,
            "player_state": entry.player_state,
            "target_snippet": entry.action_payload.get("target_snippet") if isinstance(entry.action_payload, dict) else None,
        }

        return {
            "image": image_tensor,
            "state": state_vector,
            "scenario_current": torch.tensor(current_idx, dtype=torch.long),
            "scenario_next": torch.tensor(next_idx, dtype=torch.long),
            "player_state": torch.tensor(player_state_idx, dtype=torch.long),
            "action_type": torch.tensor(action_type_idx, dtype=torch.long),
            "coords": coords,
            "coords_mask": mask,
            "meta": meta,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        images = torch.stack([item["image"] for item in batch], dim=0)
        states = torch.stack([item["state"] for item in batch], dim=0) if batch[0]["state"].numel() else None
        scenario_current = torch.stack([item["scenario_current"] for item in batch], dim=0)
        scenario_next = torch.stack([item["scenario_next"] for item in batch], dim=0)
        player_state = torch.stack([item["player_state"] for item in batch], dim=0)
        action_type = torch.stack([item["action_type"] for item in batch], dim=0)
        coords = torch.stack([item["coords"] for item in batch], dim=0)
        mask = torch.stack([item["coords_mask"] for item in batch], dim=0)
        meta = [item["meta"] for item in batch]

        return {
            "images": images,
            "states": states,
            "scenario_current": scenario_current,
            "scenario_next": scenario_next,
            "player_state": player_state,
            "action_type": action_type,
            "coords": coords,
            "coords_mask": mask,
            "meta": meta,
        }

