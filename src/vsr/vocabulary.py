"""Decode output tokens using the checkpoint's ordered subword vocabulary."""

from pathlib import Path
from collections.abc import Iterable


DEFAULT_VOCABULARY = Path(__file__).parent / "assets" / "unigram5000_units.txt"


class Vocabulary:
    def __init__(self, path: str | Path = DEFAULT_VOCABULARY):
        units = Path(path).read_text(encoding="utf-8").splitlines()
        entries = [line.split() for line in units]
        if not entries or any(
            len(entry) != 2 or entry[1] != str(index)
            for index, entry in enumerate(entries, start=1)
        ):
            raise ValueError("Vocabulary must contain ordered 'token id' rows starting at 1")
        self.tokens = ["<blank>"] + [entry[0] for entry in entries] + ["<eos>"]
        self.eos = len(self.tokens) - 1

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces = [self.tokens[int(i)] for i in token_ids if int(i) not in (-1, 0, self.eos)]
        return "".join(pieces).replace("▁", " ").replace("<space>", " ").strip()
