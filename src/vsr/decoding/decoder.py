"""Joint attention/CTC beam search, without an external language model."""

from .batch_beam_search import BatchBeamSearch
from .ctc_scorer import CTCPrefixScorer


def build_beam_search(model, vocabulary, beam_size: int, ctc_weight: float):
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(vocabulary.tokens),
        weights={"decoder": 1.0 - ctc_weight, "ctc": ctc_weight},
        scorers={"decoder": model.decoder, "ctc": CTCPrefixScorer(model.ctc, vocabulary.eos)},
        sos=vocabulary.eos,
        eos=vocabulary.eos,
        token_list=vocabulary.tokens,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
