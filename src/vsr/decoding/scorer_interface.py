# Adapted from auto_avsr/ESPnet under Apache 2.0.
# Modified to retain only the interfaces used by batch inference.
"""State selection and scoring contracts for batched beam search."""


class ScorerInterface:
    def select_state(self, state, index, new_id=None):
        return None if state is None else state[index]

    def final_score(self, state):
        return 0.0


class BatchScorerInterface(ScorerInterface):
    def batch_init_state(self, x):
        return None

    def batch_score(self, ys, states, xs):
        raise NotImplementedError


class PartialScorerInterface(ScorerInterface):
    """Marker for scorers evaluated on a subset of candidate tokens."""


class BatchPartialScorerInterface(BatchScorerInterface, PartialScorerInterface):
    def batch_score_partial(self, ys, ids, states, xs):
        raise NotImplementedError
