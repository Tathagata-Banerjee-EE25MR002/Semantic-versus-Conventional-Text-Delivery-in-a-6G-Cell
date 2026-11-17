"""
metrics2.py -- ONE metric function for every system (the "same metric code" of the Task-4 protocol).

Scorer(ref, hyp) compares the metric normal forms norm(ref) and norm(hyp) (lower-cased tokens, see tokens.py)
and returns  BLEU-4,  word accuracy (1 - WER),  sentence similarity  and the exact-reproduction flag.
BLEU / word accuracy / similarity are the functions of Exercise 1 (text_metrics.py), unchanged.
Similarity = cosine of mean-centred fastText sentence embeddings (the SBERT replacement of Exercise 1,
trained on the 20 000 training sentences and cached in data/fasttext.model).
"""
import numpy as np
from text_metrics import bleu, word_accuracy, SentenceEmbedder
from tokens import norm


class Scorer:
    def __init__(self, train_sentences):
        self.emb = SentenceEmbedder(train_sentences)
        self._memo = {}

    def __call__(self, ref, hyp):
        r, h = norm(ref), norm(hyp)
        if r == h:
            return dict(bleu=1.0, wacc=1.0, sim=1.0, exact=1.0)
        key = (r, h)
        v = self._memo.get(key)
        if v is None:
            v = dict(bleu=bleu(r, h), wacc=word_accuracy(r, h), sim=self.emb.sim(r, h), exact=0.0)
            if len(self._memo) < 400000:
                self._memo[key] = v
        return v

    def mean(self, refs, hyps):
        s = [self(r, h) for r, h in zip(refs, hyps)]
        return {k: float(np.mean([x[k] for x in s])) for k in ("bleu", "wacc", "sim", "exact")}
