"""
Learning rate scheduler and early stopping utilities.
"""

import math
import logging

import torch
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine decay with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best: {self.best_score:.4f}"
                )
                self.should_stop = True
                return True

        return False
