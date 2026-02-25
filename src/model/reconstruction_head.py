"""
Reconstruction Head: decodes compressed page vectors back to approximate
original hidden states. Used as auxiliary training signal to ensure the
compressor preserves information.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ReconstructionHead(nn.Module):
    """
    Decodes compressed page vectors back to approximate original hidden states.

    Input:  [d_page] (compressed page vector)
    Output: [num_layers, D_model] (reconstructed multi-layer hidden states)
    """

    def __init__(self, d_page: int = 512, num_layers: int = 4, d_model: int = 2048):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.target_dim = num_layers * d_model

        self.net = nn.Sequential(
            nn.Linear(d_page, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.target_dim),
        )

    def forward(self, page_vector: Tensor) -> Tensor:
        """
        Args:
            page_vector: [batch, d_page] or [d_page]

        Returns: [batch, num_layers, D_model] or [num_layers, D_model]
        """
        squeeze = False
        if page_vector.dim() == 1:
            page_vector = page_vector.unsqueeze(0)
            squeeze = True

        out = self.net(page_vector)  # [batch, num_layers * D_model]
        out = out.view(-1, self.num_layers, self.d_model)

        if squeeze:
            out = out.squeeze(0)
        return out
