"""Public inference API for prepared, video-only lip-reading clips."""

from pathlib import Path
from typing import TypedDict

import torch

from .decoding.decoder import build_beam_search
from .network.model import VideoModel
from .vocabulary import DEFAULT_VOCABULARY, Vocabulary


class Prediction(TypedDict):
    rank: int
    text: str
    score: float


class AutoVSR:
    """Load a base VSR checkpoint once and predict the top-k hypotheses.

    Input: one floating-point tensor (T, 1, 88, 88), already cropped,
    grayscale and normalized as required by Auto-AVSR.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        beam_size: int = 5,
        ctc_weight: float = 0.1,
        vocabulary_path: str | Path = DEFAULT_VOCABULARY,
    ):
        if (
            isinstance(beam_size, bool)
            or not isinstance(beam_size, int)
            or beam_size < 1
        ):
            raise ValueError("beam_size must be a positive integer")
        if not 0.0 <= ctc_weight <= 1.0:
            raise ValueError("ctc_weight must be between 0 and 1")
        self.device = torch.device(device)
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.vocabulary = Vocabulary(vocabulary_path)
        if beam_size > len(self.vocabulary.tokens):
            raise ValueError("beam_size cannot exceed the vocabulary size")
        self.model = VideoModel(len(self.vocabulary.tokens))
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, video: torch.Tensor, k: int = 5) -> list[Prediction]:
        """Return up to k completed hypotheses, sorted by descending score.

        The search uses at least k beams. Fewer results can be returned if
        fewer hypotheses finish; results are never padded or duplicated.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if k > len(self.vocabulary.tokens):
            raise ValueError("k cannot exceed the vocabulary size")
        if not isinstance(video, torch.Tensor):
            raise TypeError("video must be a torch.Tensor of shape (T, 1, 88, 88)")
        if (
            video.ndim != 4
            or video.shape[0] == 0
            or tuple(video.shape[1:]) != (1, 88, 88)
        ):
            raise ValueError("video must have shape (T, 1, 88, 88) with T > 0")
        if not video.is_floating_point():
            raise TypeError("video must already be a normalized floating-point tensor")
        if not torch.isfinite(video).all().item():
            raise ValueError("video must contain only finite values")

        video = video.to(device=self.device, dtype=torch.float32)
        features = self.model(video)
        # Each clip has its own CTC prefix state.
        search = build_beam_search(
            self.model, self.vocabulary, max(self.beam_size, k), self.ctc_weight
        )
        search.eval()
        hypotheses = search(features)
        return [
            Prediction(
                rank=rank,
                text=self.vocabulary.decode(hyp.yseq[1:].tolist()),
                score=float(hyp.score),
            )
            for rank, hyp in enumerate(hypotheses[:k], start=1)
        ]
