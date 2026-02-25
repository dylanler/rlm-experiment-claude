"""
Information retention probes: tests whether compressed latent pages
retain specific factual information from the original document.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class InformationRetentionProbe(nn.Module):
    """
    Linear probe that tests if a latent page vector can recover specific facts.

    Trained to predict binary labels (fact present/absent) from page vectors.
    High accuracy = good information retention.
    """

    def __init__(self, d_page: int, num_facts: int):
        super().__init__()
        self.probe = nn.Linear(d_page, num_facts)

    def forward(self, page_vectors: Tensor) -> Tensor:
        """
        Args:
            page_vectors: [batch, d_page]
        Returns: [batch, num_facts] logits
        """
        return self.probe(page_vectors)


def train_probe(
    probe: InformationRetentionProbe,
    page_vectors: Tensor,
    fact_labels: Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
) -> dict:
    """
    Train a linear probe and return accuracy metrics.

    Args:
        probe: InformationRetentionProbe
        page_vectors: [num_samples, d_page]
        fact_labels: [num_samples, num_facts] binary labels
        epochs: training epochs
        lr: learning rate

    Returns: dict with train_acc, val_acc
    """
    device = page_vectors.device

    # Split 80/20
    n = len(page_vectors)
    split = int(0.8 * n)
    train_vecs, val_vecs = page_vectors[:split], page_vectors[split:]
    train_labels, val_labels = fact_labels[:split], fact_labels[split:]

    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        logits = probe(train_vecs)
        loss = criterion(logits, train_labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_vecs)
            val_preds = (val_logits > 0).float()
            val_acc = (val_preds == val_labels).float().mean().item()
            best_val_acc = max(best_val_acc, val_acc)

    train_logits = probe(train_vecs)
    train_preds = (train_logits > 0).float()
    train_acc = (train_preds == train_labels).float().mean().item()

    return {
        "train_acc": train_acc,
        "val_acc": best_val_acc,
    }
