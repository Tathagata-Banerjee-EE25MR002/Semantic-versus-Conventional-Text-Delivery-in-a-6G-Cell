"""
tokens.py -- tokenisation, word vocabulary, corpus statistics and the metric "normal form".

Why this file exists
--------------------
Exercise 1 worked on characters.  Exercise 2 works on WORDS: the semantic encoder embeds words,
word-level Huffman codes words, and Task 1 asks for vocabulary size, average sentence length and
word entropy.  Everything that is word-related and shared by several modules lives here.

Conventions
-----------
* A "word" (token) is a run of letters/digits OR a single punctuation mark, lower-cased
  (regex  \\w+|[^\\w\\s]).  Punctuation marks are counted as words, as in DeepSC.
* Special ids:  PAD = 0 (padding, never transmitted),  EOS = 1 (end of sentence, transmitted so that the
  receiver knows where the sentence ends),  UNK = 2 (a word that is not in the knowledge base).
* The vocabulary (= the "knowledge base" of the word-level systems) contains every word that occurs
  >= MIN_WORD_COUNT times in the 20 000 TRAINING sentences.  Rarer words and all words never seen in
  training are mapped to UNK.  An UNK decoded at the receiver is simply omitted from the output text,
  because the receiver cannot know which word it was.  The semantic codec AND the word-level Huffman
  code use exactly the same vocabulary, so they have exactly the same knowledge.
* norm(text) = the metric normal form (lower-cased tokens separated by one space).  BLEU, word accuracy,
  similarity and "exact reproduction" are ALWAYS computed on norm(reference) versus norm(hypothesis), for
  every system (character Huffman, word Huffman, semantic).  Only then are the numbers comparable.
"""
import re
from collections import Counter
import numpy as np
from config import MIN_WORD_COUNT

_TOK = re.compile(r"\w+|[^\w\s]")
PAD, EOS, UNK = 0, 1, 2
SPECIALS = ["<pad>", "<eos>", "<unk>"]


def tokenize(s):
    """Text -> list of lower-case word / punctuation tokens."""
    return _TOK.findall(s.lower())


def norm(s):
    """Metric normal form of a sentence."""
    return " ".join(tokenize(s))


class Vocab:
    """Word vocabulary built from the training sentences only (the knowledge base)."""

    def __init__(self, train_sentences, min_count=MIN_WORD_COUNT):
        self.counts = Counter(w for s in train_sentences for w in tokenize(s))       # ALL training words
        words = sorted((w for w, c in self.counts.items() if c >= min_count), key=lambda w: (-self.counts[w], w))
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.min_count = min_count

    def __len__(self):
        return len(self.itos)

    def encode(self, s, add_eos=True):
        """Sentence -> list of ids (OOV -> UNK), optionally followed by EOS."""
        ids = [self.stoi.get(w, UNK) for w in tokenize(s)]
        return ids + [EOS] if add_eos else ids

    def decode(self, ids):
        """ids -> text.  Stops at the first EOS; PAD and UNK are omitted (the receiver cannot recover them)."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i > UNK:
                out.append(self.itos[i])
        return " ".join(out)

    def unk_free(self, s):
        """True if every word of s is in the vocabulary (so exact reproduction is possible)."""
        return all(w in self.stoi for w in tokenize(s))


def corpus_stats(train, test, vocab):
    """Corpus statistics requested in Task 1: vocabulary size, average sentence length, word entropy."""
    c = vocab.counts
    n = sum(c.values())
    p = np.array(list(c.values()), float) / n
    H = float(-(p * np.log2(p)).sum())                                    # unigram word entropy [bit/word]
    lens = np.array([len(tokenize(s)) for s in train])
    n_words_only = np.array([sum(1 for w in tokenize(s) if w[0].isalnum() or w[0] == "_") for s in train])
    ct = Counter(w for s in test for w in tokenize(s))
    nt = sum(ct.values())
    oov_tok = sum(f for w, f in ct.items() if w not in vocab.stoi) / nt
    sent_ok = float(np.mean([vocab.unk_free(s) for s in test]))
    return dict(
        n_train_sentences=len(train), n_test_sentences=len(test), n_train_tokens=int(n),
        vocab_size_all=len(c), vocab_size_model=len(vocab), min_count=vocab.min_count,
        avg_len_tokens=float(lens.mean()), avg_len_words_only=float(n_words_only.mean()),
        median_len=float(np.median(lens)), max_len=int(lens.max()),
        avg_len_tx=float(lens.mean() + 1),                                # + <eos> = units actually transmitted
        entropy_bits_per_word=H, entropy_bits_per_sentence=float(H * lens.mean()),
        avg_chars=float(np.mean([len(s) for s in train])),
        test_oov_token_rate=float(oov_tok), test_sentences_without_oov=sent_ok)
