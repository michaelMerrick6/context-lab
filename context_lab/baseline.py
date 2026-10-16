"""A fitted count bigram with additive smoothing; no gradient training needed."""

import torch
from torch.nn import functional as F


class BigramLM:
    def __init__(self, vocab_size=258, alpha=0.1):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.vocab_size, self.alpha = vocab_size, alpha
        self.counts = torch.full((vocab_size, vocab_size), alpha, dtype=torch.float64)

    def fit(self, documents):
        self.counts.fill_(self.alpha)
        for doc in documents:
            for a, b in zip(doc[:-1], doc[1:], strict=True):
                self.counts[a, b] += 1
        return self

    @property
    def log_probabilities(self):
        return (self.counts / self.counts.sum(dim=-1, keepdim=True)).log()

    def __call__(self, tokens, targets=None):
        logits = self.log_probabilities.to(tokens.device)[tokens]
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100
            )
        )
        return logits, loss
