"""
Latent Page Store: in-memory store for compressed latent pages.
Analogous to a virtual memory paging system.
"""

import torch
from torch import Tensor


class LatentPageStore:
    """
    In-memory store for compressed latent pages.
    Analogous to a virtual memory paging system.
    """

    def __init__(self):
        self.pages: dict[int, dict] = {}

    def write(self, chunk_id: int, page_vector: Tensor, metadata: dict | None = None):
        self.pages[chunk_id] = {
            "vector": page_vector.detach().cpu(),
            "metadata": metadata or {},
        }

    def read_all(self) -> Tensor:
        """Returns all page vectors stacked: [num_pages, d_page]"""
        ordered = sorted(self.pages.keys())
        return torch.stack([self.pages[k]["vector"] for k in ordered])

    def read_by_ids(self, chunk_ids: list[int]) -> Tensor:
        return torch.stack([self.pages[cid]["vector"] for cid in chunk_ids])

    def num_pages(self) -> int:
        return len(self.pages)

    def clear(self):
        self.pages = {}
