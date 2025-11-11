from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:  # pragma: no cover - graceful import fallback
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

if TYPE_CHECKING:
    from training.policy_model import PolicyModel
    import torch as torch_type


ACTION_LABELS: Sequence[str] = ("tap", "swipe", "idle")


@dataclass(frozen=True)
class PolicyActionPrediction:
    action: str
    probabilities: Tuple[float, float, float]
    coordinates: Tuple[float, float, float, float, float]
    image_size: Tuple[int, int]

    @property
    def confidence(self) -> float:
        return max(self.probabilities)


def _parse_numeric_value(value: str) -> float:
    if not value:
        return 0.0
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return 0.0
    if cleaned.endswith("%"):
        try:
            return float(cleaned[:-1]) / 100.0
        except ValueError:
            return 0.0
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
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


class PolicyRunner:
    """Loads a trained policy checkpoint and runs single-image inference."""

    def __init__(self, device: Optional[str] = None) -> None:
        self._require_torch()
        self._device = self._resolve_device(device)
        self._model: Optional["PolicyModel"] = None
        self._state_keys: Tuple[str, ...] = ()
        self._scenario_to_index: Dict[str, int] = {"<none>": 0}
        self._player_state_to_index: Dict[str, int] = {"<none>": 0}
        self._checkpoint_path: Optional[Path] = None
        self._metadata_path: Optional[Path] = None

    @staticmethod
    def _require_torch() -> None:
        if torch is None:
            message = "PyTorch is required for policy inference."
            if _TORCH_IMPORT_ERROR is not None:
                raise RuntimeError(message) from _TORCH_IMPORT_ERROR
            raise RuntimeError(message)

    @staticmethod
    def _resolve_device(preferred: Optional[str]):
        PolicyRunner._require_torch()
        assert torch is not None
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_path

    @property
    def metadata_path(self) -> Optional[Path]:
        return self._metadata_path

    @property
    def state_keys(self) -> Tuple[str, ...]:
        return self._state_keys

    @property
    def scenario_to_index(self) -> Dict[str, int]:
        return self._scenario_to_index

    @property
    def player_state_to_index(self) -> Dict[str, int]:
        return self._player_state_to_index

    def to(self, device: Optional[str]) -> None:
        target = self._resolve_device(device)
        if target == self._device:
            return
        self._device = target
        if self._model is not None:
            self._model.to(self._device)  # type: ignore[call-arg]

    def load(self, checkpoint: Path, metadata: Optional[Path] = None) -> None:
        self._require_torch()
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        meta_path = metadata or checkpoint.with_name("policy_metadata.json")
        meta_path = meta_path.expanduser().resolve()
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}\n"
                "Ensure policy_metadata.json is saved alongside the checkpoint."
            )

        payload = torch.load(checkpoint, map_location="cpu")
        if "model_state" not in payload:
            raise ValueError("Checkpoint does not contain 'model_state'.")
        state_dict = payload["model_state"]

        with meta_path.open("r", encoding="utf-8") as handle:
            metadata_payload = json.load(handle)

        scenario_to_index = dict(metadata_payload.get("scenario_to_index", {"<none>": 0}))
        if "<none>" not in scenario_to_index:
            scenario_to_index = {"<none>": 0, **scenario_to_index}
        player_state_to_index = dict(metadata_payload.get("player_state_to_index", {"<none>": 0}))
        if "<none>" not in player_state_to_index:
            player_state_to_index = {"<none>": 0, **player_state_to_index}
        state_keys = tuple(metadata_payload.get("state_keys", []))

        scenario_vocab_size = max(scenario_to_index.values(), default=0) + 1
        player_state_vocab_size = max(player_state_to_index.values(), default=0) + 1

        scenario_embed_dim = state_dict["scenario_embedding.weight"].shape[1]
        player_state_embed_dim = state_dict.get("player_state_embedding.weight", torch.empty(0, 0)).shape[1]

        from training.policy_model import PolicyModel as PolicyModelCls

        model = PolicyModelCls(
            state_dim=len(state_keys),
            scenario_vocab_size=scenario_vocab_size,
            player_state_vocab_size=player_state_vocab_size,
            scenario_embed_dim=scenario_embed_dim,
            player_state_embed_dim=player_state_embed_dim or 16,
        )
        model.load_state_dict(state_dict)
        model.to(self._device)  # type: ignore[call-arg]
        model.eval()

        self._model = model
        self._state_keys = state_keys
        self._scenario_to_index = scenario_to_index
        self._player_state_to_index = player_state_to_index
        self._checkpoint_path = checkpoint
        self._metadata_path = meta_path

    def predict(
        self,
        png_bytes: bytes,
        state_board: Dict[str, str],
        scenario_current: Optional[str],
        scenario_next: Optional[str],
        player_state: Optional[str],
    ) -> Optional[PolicyActionPrediction]:
        self._require_torch()
        if self._model is None:
            return None

        image = Image.open(BytesIO(png_bytes)).convert("RGB")
        width, height = image.size
        np_image = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).to(self._device)

        sc_curr_idx = self._scenario_to_index.get(scenario_current or "<none>", 0)
        sc_next_idx = self._scenario_to_index.get(scenario_next or "<none>", 0)
        ps_idx = self._player_state_to_index.get(player_state or "<none>", 0)

        sc_curr_tensor = torch.tensor([sc_curr_idx], dtype=torch.long, device=self._device)
        sc_next_tensor = torch.tensor([sc_next_idx], dtype=torch.long, device=self._device)
        player_state_tensor = torch.tensor([ps_idx], dtype=torch.long, device=self._device)

        state_tensor = None
        if self._state_keys:
            vector = torch.zeros(len(self._state_keys), dtype=torch.float32)
            for idx, key in enumerate(self._state_keys):
                vector[idx] = float(_parse_numeric_value(state_board.get(key, "")))
            state_tensor = vector.unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, coords = self._model(
                image_tensor,
                sc_curr_tensor,
                sc_next_tensor,
                state_tensor,
                player_state_tensor,
            )

        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
        coords_vec = coords.squeeze(0).cpu().numpy().tolist()

        best_idx = int(np.argmax(probabilities))
        action = ACTION_LABELS[best_idx]

        return PolicyActionPrediction(
            action=action,
            probabilities=(
                float(probabilities[0]),
                float(probabilities[1]),
                float(probabilities[2]),
            ),
            coordinates=(
                float(coords_vec[0]),
                float(coords_vec[1]),
                float(coords_vec[2]),
                float(coords_vec[3]),
                float(coords_vec[4]),
            ),
            image_size=(width, height),
        )

