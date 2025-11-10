from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from training.policy_dataset import PolicyDataset
from training.policy_model import PolicyModel


def build_loaders(
    dataset: PolicyDataset,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
) -> tuple[DataLoader, Optional[DataLoader]]:
    if val_ratio <= 0.0 or len(dataset) < 2:
        train_subset = dataset
        val_loader = None
    else:
        total = len(dataset)
        val_size = max(1, int(total * val_ratio))
        indices = torch.randperm(total).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        if not train_indices:
            train_indices, val_indices = val_indices[val_size:], val_indices[:val_size]
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=PolicyDataset.collate_fn,
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=PolicyDataset.collate_fn,
    )
    return train_loader, val_loader


def coord_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    denom = mask.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff.pow(2).sum()) / denom


def evaluate(
    model: PolicyModel,
    loader: Optional[DataLoader],
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    if loader is None:
        return 0.0, 0.0

    model.eval()
    total_type_loss = 0.0
    total_coord_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            states = batch["states"]
            if states is not None:
                states = states.to(device)
            scenarios_curr = batch["scenario_current"].to(device)
            scenarios_next = batch["scenario_next"].to(device)
            player_state = batch["player_state"].to(device)
            action_type = batch["action_type"].to(device)
            coords = batch["coords"].to(device)
            mask = batch["coords_mask"].to(device)

            logits, coords_pred = model(images, scenarios_curr, scenarios_next, states, player_state)
            loss_type = criterion(logits, action_type)
            loss_coords = coord_loss(coords_pred, coords, mask)

            total_type_loss += loss_type.item()
            total_coord_loss += loss_coords.item()
            batches += 1

    if batches == 0:
        return 0.0, 0.0
    return total_type_loss / batches, total_coord_loss / batches


def train(args: argparse.Namespace) -> None:
    dataset = PolicyDataset(Path(args.samples), max_samples=args.max_samples)
    train_loader, val_loader = build_loaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = PolicyModel(
        state_dim=len(dataset.state_keys),
        scenario_vocab_size=len(dataset.scenario_to_index),
        player_state_vocab_size=len(dataset.player_state_to_index),
        scenario_embed_dim=args.scenario_embed_dim,
        player_state_embed_dim=args.player_state_embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    type_criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_type_loss = 0.0
        running_coord_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            states = batch["states"]
            if states is not None:
                states = states.to(device)
            scenarios_curr = batch["scenario_current"].to(device)
            scenarios_next = batch["scenario_next"].to(device)
            player_state = batch["player_state"].to(device)
            action_type = batch["action_type"].to(device)
            coords = batch["coords"].to(device)
            mask = batch["coords_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, coords_pred = model(images, scenarios_curr, scenarios_next, states, player_state)

            loss_type = type_criterion(logits, action_type)
            loss_coords = coord_loss(coords_pred, coords, mask)
            loss = loss_type + args.coord_loss_weight * loss_coords
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_type_loss += loss_type.item()
            running_coord_loss += loss_coords.item()
            batches += 1

        scheduler.step()

        avg_type_loss = running_type_loss / max(1, batches)
        avg_coord_loss = running_coord_loss / max(1, batches)
        val_type_loss, val_coord_loss = evaluate(model, val_loader, device, type_criterion)

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_type_loss={avg_type_loss:.4f} train_coord_loss={avg_coord_loss:.4f} "
            f"val_type_loss={val_type_loss:.4f} val_coord_loss={val_coord_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scenario_to_index": dataset.scenario_to_index,
            "state_keys": dataset.state_keys,
            "metadata": {
                "samples": str(args.samples),
                "trained_epochs": epoch,
                "val_ratio": args.val_ratio,
            },
        }
        torch.save(checkpoint, output_dir / f"policy_model_epoch_{epoch:02d}.pt")

    final_path = output_dir / "policy_model_latest.pt"
    torch.save(checkpoint, final_path)

    with (output_dir / "policy_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scenario_to_index": dataset.scenario_to_index,
                "player_state_to_index": dataset.player_state_to_index,
                "state_keys": dataset.state_keys,
            },
            handle,
            indent=2,
        )
    print(f"Training finished. Latest checkpoint saved to {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy model for game actions.")
    parser.add_argument("--samples", type=Path, default=Path("data/training/training_samples.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/policy"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--scenario-embed-dim", type=int, default=32)
    parser.add_argument("--player-state-embed-dim", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

