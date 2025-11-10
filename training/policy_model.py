from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PolicyModel(nn.Module):
    """Simple vision + context model predicting tap or swipe actions."""

    def __init__(
        self,
        state_dim: int,
        scenario_vocab_size: int,
        player_state_vocab_size: int,
        scenario_embed_dim: int = 32,
        player_state_embed_dim: int = 16,
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.visual_projection = nn.Linear(128, 256)

        self.scenario_embedding = nn.Embedding(scenario_vocab_size, scenario_embed_dim)
        scenario_feat_dim = scenario_embed_dim * 2

        if player_state_vocab_size > 0:
            self.player_state_embedding: Optional[nn.Embedding] = nn.Embedding(
                player_state_vocab_size, player_state_embed_dim
            )
            player_state_feat_dim = player_state_embed_dim
        else:
            self.player_state_embedding = None
            player_state_feat_dim = 0

        self.state_encoder: Optional[nn.Module]
        if state_dim > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(inplace=True),
            )
            state_feat_dim = 128
        else:
            self.state_encoder = None
            state_feat_dim = 0

        fused_dim = 256 + scenario_feat_dim + state_feat_dim + player_state_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.action_type_head = nn.Linear(128, 3)  # tap, swipe, none
        self.coords_head = nn.Linear(128, 5)  # normalized coord targets

    def forward(
        self,
        images: torch.Tensor,
        scenario_current: torch.Tensor,
        scenario_next: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        player_state: Optional[torch.Tensor] = None,
    ):
        visual_feat = self.backbone(images).flatten(1)
        visual_feat = self.visual_projection(visual_feat)

        scenario_feat = torch.cat(
            [
                self.scenario_embedding(scenario_current),
                self.scenario_embedding(scenario_next),
            ],
            dim=1,
        )

        features = [visual_feat, scenario_feat]

        if self.state_encoder is not None and states is not None:
            features.append(self.state_encoder(states))
        if self.player_state_embedding is not None and player_state is not None:
            features.append(self.player_state_embedding(player_state))

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        action_type_logits = self.action_type_head(fused)
        coords_pred = self.coords_head(fused)
        return action_type_logits, coords_pred

