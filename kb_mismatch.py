"""
kb_mismatch.py -- Task 2, last part: mismatch of the knowledge base (KB).

The KB of the semantic system is the training corpus + the trained weights; the KB of the word-level Huffman
code is the same vocabulary; the character-level Huffman code knows only the alphabet.  Two kinds of mismatch:

  set A  "unseen templates"   : sentence STRUCTURES that do not occur in the (1960s) Brown corpus, e.g.
                                "please send me the report before friday ."   -- but only vocabulary words.
  set B  "unseen words"       : the same templates filled with words that are NOT in the vocabulary
                                (modern / local words such as smartphone, bengaluru, hubballi ...).
  set T  "in-distribution"    : 200 ordinary test sentences (reference).

Each set has 200 sentences generated with a fixed seed.  A sentence of set B contains >= 1 word outside the
vocabulary (checked by assertion).  For every set we measure similarity and the fraction of exactly reproduced
sentences of  (i) the semantic transceivers (K = 1,2,4,8, AWGN) and (ii) the conventional chain with
16-QAM 1/2 (character and word Huffman) at the same SNR.
"""
import os, json
import numpy as np
from config import *
from corpus import load_corpus
from tokens import Vocab, tokenize
from metrics2 import Scorer
from source_coding import CharHuffman
from word_huffman import WordHuffman
import link
import semantic_model as sm
import task2_semantic as t2

TEMPLATES = [
    "please send me the {thing} before {day} .",
    "the meeting will start at {num} on {day} in {place} .",
    "the {thing} is ready and you can collect it from {place} .",
    "can you tell me where the {thing} is ?",
    "we will travel from {place} to {place2} by {vehicle} on {day} .",
    "{person} said that the {thing} will arrive on {day} .",
    "thank you for your help with the {thing} .",
    "the price of the {thing} is {num} dollars .",
]
SEEN = dict(thing=["report", "letter", "book", "package", "paper", "money", "picture", "message"],
            day=["monday", "tuesday", "friday", "sunday", "saturday"],
            num=["5", "10", "20", "100", "50"],
            place=["london", "chicago", "boston", "texas", "paris", "washington"],
            place2=["paris", "london", "chicago", "boston"],
            vehicle=["train", "car", "plane", "boat"],
            person=["john", "mary", "the doctor", "the president"])
UNSEEN = dict(thing=["smartphone", "laptop", "router", "blockchain", "podcast", "spreadsheet", "wifi", "drone"],
              day=SEEN["day"],
              num=SEEN["num"],
              place=["dharwad", "bengaluru", "hubballi", "mumbai", "kolkata", "chennai"],
              place2=["dharwad", "bengaluru", "hubballi", "pune"],
              vehicle=["metro", "scooter", "rickshaw", "helicopter"],
              person=["priya", "rahul", "the engineer", "the technician"])


def make_set(fillers, n=200, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = TEMPLATES[i % len(TEMPLATES)]
        f = {k: v[rng.integers(len(v))] for k, v in fillers.items()}
        out.append(t.format(**f))
    return out


def build_sets(vocab, te):
    A, B = make_set(SEEN), make_set(UNSEEN)
    assert all(vocab.unk_free(s) for s in A), [s for s in A if not vocab.unk_free(s)][:3]
    assert all(not vocab.unk_free(s) for s in B)
    return {"in-distribution": te[:200], "unseen template": A, "unseen words": B}


def conv_eval(huff, sentences, mcs, snr, scorer, seed=3):
    tx = link.TxFrames(huff, sentences, mcs)
    r = link.run(tx, snr, np.random.default_rng(seed), chan="awgn", want_text=True)
    hyp = [huff.decode(huff.encode(s)) if not r["err_frames"][i] else r["text"][i] for i, s in enumerate(sentences)]
    if isinstance(huff, CharHuffman) and not isinstance(huff, WordHuffman):
        hyp = [s if not r["err_frames"][i] else r["text"][i] for i, s in enumerate(sentences)]
    return scorer.mean(sentences, hyp)


def main(snrs=(0.0, 6.0, 12.0, 18.0)):
    tr, te = load_corpus()
    vocab = Vocab(tr); scorer = Scorer(tr)
    sets = build_sets(vocab, te)
    json.dump({k: v[:8] for k, v in sets.items()}, open(os.path.join(RES_DIR, "task2_mismatch_examples.json"), "w"), indent=1)
    out = dict(snr=list(snrs), semantic={}, conv={})
    for K in K_LIST:
        model = sm.load_codec(K, vocab, t2.ckpt(K))
        out["semantic"][str(K)] = {}
        for name, S in sets.items():
            res = {"sim": [], "exact": []}
            for snr in snrs:
                hyp = t2.decode_sentences(model, vocab, S, snr, seed=SEED + 5)
                m = scorer.mean(S, hyp); res["sim"].append(m["sim"]); res["exact"].append(m["exact"])
            out["semantic"][str(K)][name] = res
        print(f"[2] mismatch K={K}: sim at {snrs[2]} dB " + str({n: round(r['sim'][2], 3) for n, r in out['semantic'][str(K)].items()}), flush=True)
    huffs = {"char": CharHuffman(tr), "word": WordHuffman(vocab, tr)}
    for mcs_name in ("QPSK 1/2", "16-QAM 1/2"):
        mcs = next(m for m in MCS_TABLE if m["name"] == mcs_name)
        for src, h in huffs.items():
            key = f"{src} Huffman, {mcs_name}"
            out["conv"][key] = {}
            for name, S in sets.items():
                res = {"sim": [], "exact": []}
                for snr in snrs:
                    m = conv_eval(h, S, mcs, snr, scorer); res["sim"].append(m["sim"]); res["exact"].append(m["exact"])
                out["conv"][key][name] = res
            print(f"[2] mismatch {key}: sim at {snrs[2]} dB " + str({n: round(r['sim'][2], 3) for n, r in out['conv'][key].items()}), flush=True)
    json.dump(out, open(os.path.join(RES_DIR, "task2_mismatch.json"), "w"))
    return out


if __name__ == "__main__":
    main()
