# src/train/nn_model.py
from __future__ import annotations

import torch
import torch.nn as nn

class YCMLP(nn.Module):
    """
    Simple MLP binary classifier.
    Input: [text_embedding || 4 frame features]
    Output: logits (apply sigmoid for probability)
    """
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.25):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits shape (N,)
        return self.net(x).squeeze(1)