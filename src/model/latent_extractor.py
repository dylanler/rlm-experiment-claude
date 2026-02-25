"""
Latent state extraction from frozen transformer hidden layers.

Extracts hidden states from specified layers and pools across
the sequence dimension to produce fixed-size representations per chunk.
"""

import torch
from torch import Tensor


def extract_latent_states(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    extraction_layers: list[int],
    pooling: str = "mean",
) -> Tensor:
    """
    Forward pass with output_hidden_states=True.
    Extract hidden states from specified layers.
    Pool across sequence dimension.

    Args:
        model: Frozen Qwen3-1.7B model
        input_ids: [1, seq_len]
        attention_mask: [1, seq_len]
        extraction_layers: which layers to extract from (0-indexed, 0=embedding output)
        pooling: "mean" | "last_token"

    Returns: [num_extraction_layers, D_model]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple of (num_layers+1) tensors, each [batch, seq_len, D_model]
    selected = torch.stack(
        [outputs.hidden_states[l] for l in extraction_layers]
    )  # [num_layers_selected, batch, seq, D_model]

    if pooling == "mean":
        mask = attention_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, 1, seq, 1]
        pooled = (selected * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-9)
    elif pooling == "last_token":
        last_idx = attention_mask.sum(dim=-1) - 1  # [batch]
        # Gather last valid token for each layer
        last_idx_expanded = last_idx.view(1, -1, 1, 1).expand(
            selected.shape[0], -1, 1, selected.shape[-1]
        )
        pooled = selected.gather(2, last_idx_expanded).squeeze(2)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    return pooled.squeeze(1).float()  # [num_layers_selected, D_model], always float32
