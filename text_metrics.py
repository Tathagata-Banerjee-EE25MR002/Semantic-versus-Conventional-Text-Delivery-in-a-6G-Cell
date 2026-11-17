"""
text_metrics.py -- the three text-quality measures requested in Task 1(c):

1. BLEU          (bilingual evaluation understudy): geometric mean of modified
                 1..4-gram precisions times a brevity penalty, sentence level, with
                 the standard "add-epsilon" smoothing (method 1 of Chen & Cherry)
                 so a single missing 4-gram does not zero the whole score.
2. Word accuracy = 1 - WER, WER = (substitutions + deletions + insertions)/#ref words
                 from a word-level Levenshtein alignment, clipped to [0, 1].
3. Sentence similarity: cosine similarity of sentence embeddings.

About the embedding model (honest note): a pre-trained sentence encoder
(e.g. SBERT) cannot be downloaded in the sandbox where this code was developed
(no access to huggingface.co).  We therefore train a *fastText* model (Bojanowski
et al., 2017; gensim implementation) on the 20 000 training sentences.  fastText
represents a word as the sum of its character n-gram vectors, so a *corrupted*
word such as "commanded" -> "commandtd" still gets a vector close to the original,
which is the behaviour a semantic similarity metric needs.  A sentence embedding is
the mean of its word vectors with the corpus-mean vector removed (the standard
"common component removal" trick; without it all sentence vectors are strongly
correlated and cosine similarities of unrelated sentences would be ~0.9).  The
cosine is clipped to [0, 1].  With a pre-trained SBERT model the absolute numbers
would change slightly, the cliff behaviour would not; swap `SentenceEmbedder`
if one is available (the interface is just `.sim(ref, hyp)`).
"""
import os, re, math
from collections import Counter
import numpy as np
from config import DATA_DIR, SEED

_TOK = re.compile(r"\w+|[^\w\s]")


# ------------------------------------------------------------------ BLEU
def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(ref, hyp, max_n=4, eps=0.1):
    """Sentence BLEU-4 in [0,1] (single reference, add-eps smoothing)."""
    r, h = ref.split(), hyp.split()
    if len(h) == 0:
        return 0.0
    logp = 0.0
    for n in range(1, max_n + 1):
        rc, hc = _ngrams(r, n), _ngrams(h, n)
        tot = max(sum(hc.values()), 0)
        match = sum(min(c, rc[g]) for g, c in hc.items())
        if tot == 0:
            p = eps / 1.0                       # hypothesis shorter than n words
        else:
            p = (match if match > 0 else eps) / tot
        logp += math.log(p) / max_n
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return float(min(1.0, bp * math.exp(logp)))


# ------------------------------------------------------------------ WER
def word_accuracy(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return float(max(0.0, 1.0 - prev[-1] / len(r)))


# ------------------------------------------------------------------ similarity
class SentenceEmbedder:
    def __init__(self, train_sentences, dim=100, epochs=10, cache=True):
        from gensim.models import FastText
        path = os.path.join(DATA_DIR, "fasttext.model")
        if cache and os.path.exists(path):
            self.ft = FastText.load(path)
        else:
            toks = [t for t in (_TOK.findall(s.lower()) for s in train_sentences)]
            self.ft = FastText(vector_size=dim, window=5, min_count=2, min_n=2, max_n=5,
                               workers=2, seed=SEED, epochs=epochs, sg=1)
            self.ft.build_vocab(toks)
            self.ft.train(toks, total_examples=len(toks), epochs=epochs)
            self.ft.save(path)
        self._cache = {}
        vecs = [self._sent_raw(s) for s in train_sentences[:3000]]
        self.mean = np.mean([v for v in vecs if v is not None], axis=0)

    def _wv(self, w):
        v = self._cache.get(w)
        if v is None:
            v = self.ft.wv[w]
            if len(self._cache) < 300000:
                self._cache[w] = v
        return v

    def _sent_raw(self, s):
        toks = _TOK.findall(s.lower())
        if not toks:
            return None
        return np.mean([self._wv(t) for t in toks], axis=0)

    def vec(self, s):
        v = self._sent_raw(s)
        return None if v is None else v - self.mean

    def sim(self, ref, hyp):
        a, b = self.vec(ref), self.vec(hyp)
        if a is None or b is None:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(a @ b / (na * nb), 0.0, 1.0))


if __name__ == "__main__":
    from corpus import load_corpus
    tr, te = load_corpus()
    emb = SentenceEmbedder(tr)
    a = te[0]
    print(a)
    for b in (a, a.replace("man", "mxn"), a[:40], te[1], "zzzz qqq"):
        print(f"{b[:50]!r:55} bleu={bleu(a,b):.3f} wacc={word_accuracy(a,b):.3f} sim={emb.sim(a,b):.3f}")
