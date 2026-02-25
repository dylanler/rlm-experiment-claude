"""
Page Aggregator: aggregates multiple latent pages into a fixed number
of soft-prompt embeddings using a Perceiver-style cross-attention bottleneck.

Supports question-conditioned aggregation: when question embeddings are
provided, query tokens are biased toward question-relevant page retrieval.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PageAggregator(nn.Module):
    """
    Aggregates multiple latent pages into a fixed number of soft-prompt embeddings.

    Input:  page_vectors [num_pages, d_page], optional question_embed [q_len, D_model]
    Output: [num_soft_tokens, D_model]  — ready for injection into the LM
    """

    def __init__(
        self,
        d_page: int = 512,
        d_model: int = 2048,
        num_soft_tokens: int = 16,
        num_heads: int = 8,
        num_agg_layers: int = 1,
    ):
        super().__init__()
        self.d_page = d_page
        self.d_model = d_model
        self.num_soft_tokens = num_soft_tokens

        # Project pages up to model dimension
        self.page_proj = nn.Linear(d_page, d_model)

        # Learnable query tokens (base queries)
        self.query_tokens = nn.Parameter(
            torch.randn(num_soft_tokens, d_model) * 0.02
        )

        # Question conditioning via bottleneck projection
        # Maps mean-pooled question embedding to per-query-token bias
        d_bottleneck = 128
        self.q_down = nn.Linear(d_model, d_bottleneck)
        self.q_up = nn.Linear(d_bottleneck, num_soft_tokens * d_model)

        # Cross-attention layers: queries attend to pages
        agg_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.cross_attn = nn.TransformerDecoder(agg_layer, num_layers=num_agg_layers)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, page_vectors: Tensor, question_embed: Tensor = None) -> Tensor:
        """
        Args:
            page_vectors: [num_pages, d_page]
            question_embed: [q_len, D_model] optional question token embeddings

        Returns: [num_soft_tokens, D_model]
        """
        # Project pages: [num_pages, D_model]
        memory = self.page_proj(page_vectors).unsqueeze(0)  # [1, num_pages, D_model]

        # Start from base query tokens
        queries = self.query_tokens  # [num_soft_tokens, D_model]

        # Add question-conditioned bias if question is provided
        if question_embed is not None:
            q_pooled = question_embed.mean(dim=0)  # [D_model]
            q_bias = self.q_up(torch.nn.functional.silu(self.q_down(q_pooled)))
            q_bias = q_bias.view(self.num_soft_tokens, self.d_model)
            queries = queries + q_bias

        queries = queries.unsqueeze(0)  # [1, num_soft_tokens, D_model]

        # Cross-attend
        out = self.cross_attn(queries, memory)  # [1, num_soft_tokens, D_model]

        return self.output_norm(out.squeeze(0))  # [num_soft_tokens, D_model]
