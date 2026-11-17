"""
corpus.py -- English text corpus for the text-communication link.

Source: the Brown Corpus (Kucera & Francis, ~57k English sentences from 15 genres,
distributed with NLTK's `nltk_data`).  It is downloaded once from the public
nltk_data GitHub repository and cached under data/.

Pre-processing
--------------
* The corpus is POS-tagged ("word/tag").  We keep the word, drop the tag.
* Tokens are de-tokenised to normal running text (no space before , . ; : ? !,
  PTB-style quotes `` '' become ").
* Only printable 7-bit ASCII is kept (95 symbols) so the character-level Huffman
  alphabet is small and well defined.
* Sentences with 40 <= length <= 110 characters are kept.  The upper bound keeps
  the frame length (and the Viterbi trellis) bounded; the lower bound removes
  fragments such as headings.
* Duplicates are removed, the list is shuffled with a fixed seed and split into
  N_TRAIN_SENT (>= 20 000) training and N_TEST_SENT (>= 2 000) test sentences.
  The training set is used for the Huffman code statistics and the word-embedding
  model; the test set is used for *all* link-level measurements.
"""
import os, re, io, json, zipfile, urllib.request
import numpy as np
from config import DATA_DIR, SEED, N_TRAIN_SENT, N_TEST_SENT

BROWN_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
             "packages/corpora/brown.zip")
CACHE = os.path.join(DATA_DIR, "sentences.json")


def _download_brown():
    """Download brown.zip (3.3 MB) and return {filename: text}."""
    path = os.path.join(DATA_DIR, "brown.zip")
    if not os.path.exists(path):
        print("[corpus] downloading Brown corpus ...")
        urllib.request.urlretrieve(BROWN_URL, path)
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            if re.fullmatch(r"c[a-r]\d\d", base):          # the 500 text files
                out[base] = z.read(n).decode("latin-1")
    return out


def _detok(tokens):
    """Join PTB-like tokens into normal text."""
    s = " ".join(tokens)
    s = s.replace("``", '"').replace("''", '"')
    s = re.sub(r"\s+([,.;:?!%)])", r"\1", s)    # no space before punctuation
    s = re.sub(r"([(])\s+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_sentences():
    """Extract clean sentences from the raw tagged corpus."""
    files = _download_brown()
    sents, seen = [], set()
    for name in sorted(files):
        for line in files[name].splitlines():
            line = line.strip()
            if not line:
                continue
            toks = []
            for t in line.split():
                if "/" not in t:
                    continue
                w = t.rsplit("/", 1)[0]
                toks.append(w)
            if not toks:
                continue
            s = _detok(toks)
            if not (40 <= len(s) <= 110):
                continue
            if not all(32 <= ord(c) <= 126 for c in s):   # printable ASCII only
                continue
            if "/" in s or s in seen:
                continue
            seen.add(s)
            sents.append(s)
    return sents


def load_corpus():
    """Return (train_sentences, test_sentences); cached on disk."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            d = json.load(f)
        return d["train"], d["test"]
    sents = build_sentences()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(sents))
    sents = [sents[i] for i in idx]
    need = N_TRAIN_SENT + N_TEST_SENT
    assert len(sents) >= need, f"corpus too small: {len(sents)} < {need}"
    train, test = sents[:N_TRAIN_SENT], sents[N_TRAIN_SENT:N_TRAIN_SENT + N_TEST_SENT]
    with open(CACHE, "w") as f:
        json.dump(dict(train=train, test=test), f)
    print(f"[corpus] {len(sents)} usable sentences -> {len(train)} train / {len(test)} test")
    return train, test


if __name__ == "__main__":
    tr, te = load_corpus()
    print(len(tr), len(te)); print(tr[0]); print(te[0])
