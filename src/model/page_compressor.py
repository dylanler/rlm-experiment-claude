"""
Page Compressor: compresses multi-layer hidden states into a single
fixed-size latent page vector.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PageCompressor(nn.Module):
    """
    Compresses multi-layer hidden states into a single fixed-size latent page vector.

    Input:  [num_extraction_layers, D_model]  (e.g., [4, 2048])
    Output: [D_page]                          (e.g., [512])
    """

    def __init__(self, num_layers: int, d_model: int, d_page: int = 512):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_page = d_page
        self.flatten_dim = num_layers * d_model

        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_page),
            nn.LayerNorm(d_page),
        )

    def forward(self, multi_layer_states: Tensor) -> Tensor:
        """
        Args:
            multi_layer_states: [batch, num_layers, D_model] or [num_layers, D_model]

        Returns: [batch, d_page] or [d_page]
        """
        squeeze = False
        if multi_layer_states.dim() == 2:
            multi_layer_states = multi_layer_states.unsqueeze(0)
            squeeze = True

        flat = multi_layer_states.reshape(-1, self.flatten_dim)
        out = self.net(flat)  # [batch, d_page]

        if squeeze:
            out = out.squeeze(0)
        return out
