# Adapted from auto_avsr (Copyright 2023 Imperial College London, Pingchuan Ma).
# Apache 2.0. Modified to retain only the video inference network.
"""Video-only Auto-AVSR base architecture, compatible with its checkpoints."""

import torch
from torch import nn

from .ctc import CTC
from .decoder import TransformerDecoder
from .encoder import ConformerEncoder
from .frontend import video_resnet


class VideoModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.frontend = video_resnet()
        self.proj_encoder = nn.Linear(512, 768)
        self.encoder = ConformerEncoder(
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=12,
            cnn_module_kernel=31,
        )
        self.decoder = TransformerDecoder(
            odim=vocab_size,
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=6,
        )
        self.ctc = CTC(vocab_size, 768)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a prepared clip of shape (T, 1, H, W) as (T, 768)."""
        features = self.proj_encoder(self.frontend(video.unsqueeze(0)))
        encoded, _ = self.encoder(features, None)
        return encoded.squeeze(0)
