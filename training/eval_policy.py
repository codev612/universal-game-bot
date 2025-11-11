from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.policy_dataset import PolicyDataset
from training.policy_model import PolicyModel
from training.train_policy import coord_loss, evaluate


ACTION_LABELS = ["tap", "swipe", "idle"]


def load_metadata(metadata_path: Optional[Path]) -> Dict[str, Any]:
    if metadata_path is None:
        return {}
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(
    checkpoint: Dict[str, Any],
    dataset: PolicyDataset,
    metadata: Dict[str, Any],
    device: torch.device,
) -> PolicyModel:
    scenario_vocab_size = len(metadata.get("scenario_to_index", dataset.scenario_to_index))
    player_state_vocab_size = len(metadata.get("player_state_to_index", dataset.player_state_to_index))
    state_dim = len(metadata.get("state_keys", dataset.state_keys))

    player_state_embed_dim = checkpoint.get("metadata", {}).get("player_state_embed_dim")
    scenario_embed_dim = checkpoint.get("metadata", {}).get("scenario_embed_dim")

    model = PolicyModel(
        state_dim=state_dim,
        scenario_vocab_size=scenario_vocab_size,
        player_state_vocab_size=player_state_vocab_size,
        scenario_embed_dim=scenario_embed_dim or 32,
        player_state_embed_dim=player_state_embed_dim or 16,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def run_inference(
    model: PolicyModel,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            states = batch["states"]
            if states is not None:
                states = states.to(device)
            scenarios_curr = batch["scenario_current"].to(device)
            scenarios_next = batch["scenario_next"].to(device)
            player_state = batch["player_state"].to(device)
            logits, coords_pred = model(images, scenarios_curr, scenarios_next, states, player_state)

            probs = torch.softmax(logits, dim=1).cpu()
            coords_pred = coords_pred.cpu()

            action_gt = batch["action_type"]
            coords_gt = batch["coords"]
            mask_gt = batch["coords_mask"]

            for idx, meta in enumerate(batch["meta"]):
                height, width = meta["image_size"]
                pred_vec = coords_pred[idx]
                pred_pixels = [
                    float(pred_vec[0] * width),
                    float(pred_vec[1] * height),
                    float(pred_vec[2] * width),
                    float(pred_vec[3] * height),
                    float(pred_vec[4]),
                ]
                gt_vec = coords_gt[idx]
                gt_pixels = [
                    float(gt_vec[0] * width),
                    float(gt_vec[1] * height),
                    float(gt_vec[2] * width),
                    float(gt_vec[3] * height),
                    float(gt_vec[4]),
                ]

                probs_vec = probs[idx]
                pred_type_idx = int(torch.argmax(probs_vec).item())

                results.append(
                    {
                        "path": meta["path"],
                        "prediction": {
                            "action_type": ACTION_LABELS[pred_type_idx],
                            "action_probs": [float(x) for x in probs_vec],
                            "coords": pred_pixels,
                        },
                        "ground_truth": {
                            "action_type": ACTION_LABELS[int(action_gt[idx])],
                            "coords": gt_pixels,
                            "mask": [float(x) for x in mask_gt[idx]],
                        },
                        "meta": {
                            "game": meta.get("game"),
                            "player_state": meta.get("player_state"),
                            "target_snippet": meta.get("target_snippet"),
                            "image_size": meta.get("image_size"),
                        },
                    }
                )

    return results


def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = 0
    for item in results:
        if item["prediction"]["action_type"] == item["ground_truth"]["action_type"]:
            correct += 1
    return correct / len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and run inference with a trained policy model.")
    parser.add_argument("--samples", type=Path, default=Path("data/training/training_samples.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None, help="Optional policy_metadata.json path.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL file to write inference results.")
    args = parser.parse_args()

    dataset = PolicyDataset(Path(args.samples))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PolicyDataset.collate_fn,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = load_metadata(args.metadata)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(checkpoint, dataset, metadata, device)

    type_criterion = nn.CrossEntropyLoss()
    val_type_loss, val_coord_loss = evaluate(model, loader, device, type_criterion)
    print(f"type_loss={val_type_loss:.4f} coord_loss={val_coord_loss:.4f}")

    results = run_inference(model, loader, device)
    accuracy = compute_accuracy(results)
    print(f"action_accuracy={accuracy * 100:.2f}% over {len(results)} samples")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item) + "\n")
        print(f"Inference results written to {output_path}")


if __name__ == "__main__":
    main()

