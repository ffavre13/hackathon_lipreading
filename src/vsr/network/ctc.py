"""CTC projection used to score hypotheses, without a training loss."""

import torch
from torch import nn


class CTC(nn.Module):
    def __init__(self, odim: int, encoder_dim: int):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_dim, odim)

    def log_softmax(self, features: torch.Tensor) -> torch.Tensor:
        return self.ctc_lo(features).log_softmax(dim=-1)
