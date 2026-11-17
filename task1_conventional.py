"""
task1_conventional.py -- Task 1, part 2: judge the conventional link of Exercise 1 BY MEANING.

For every combination of
    source coder : character-level Huffman (Exercise 1)  |  word-level Huffman (new)
    channel      : AWGN | Rayleigh block fading + zero forcing   (one gain per sentence, as in Exercise 1)
    MCS          : the six MCS of Exercise 1 (QPSK/16-QAM/64-QAM, rates 1/2 ... 3/4)
all 2000 test sentences are sent through the chain of Exercise 1 (link.py, unchanged: Huffman -> terminated
K=7 convolutional code -> puncturing -> Gray QAM -> channel -> ZF -> max-log LLR -> soft Viterbi -> Huffman)
and scored with the common metric code (metrics2.Scorer):  BLEU, word accuracy, similarity, exact match, plus
block error rate and the average number of CHANNEL USES (= symbols) per sentence.

Result file: results/task1_conv_sweeps.json
    sweeps[source][channel][mcs] = dict(snr, ber, bler, bleu, wacc, sim, exact, avg_symbols, avg_info_bits)
    clean[source] = the "knowledge-base ceiling": scores when NO bit error occurs (for the word code this is
                    < 1 because words outside the vocabulary are lost -- a limit of the source code, not of
                    the channel).
Why the 'clean' scores matter: a sentence without bit errors is NOT automatically a perfect sentence for the
word-level code, so error-free frames use the clean-decode text, not the original text.
"""
import os, json, time
import numpy as np
from multiprocessing import Pool
from config import *
from corpus import load_corpus
from source_coding import CharHuffman
from word_huffman import WordHuffman
from tokens import Vocab
from metrics2 import Scorer
import link

SOURCES = ("char", "word")
_G = {}


def _init():
    tr, te = load_corpus()
    vocab = Vocab(tr)
    _G["huff"] = {"char": CharHuffman(tr), "word": WordHuffman(vocab, tr)}
    _G["test"] = te[:LINK_NFRAMES]
    _G["scorer"] = Scorer(tr)
    clean = {}
    for src, h in _G["huff"].items():
        hyp0 = [s if src == "char" else h.decode(h.encode(s)) for s in _G["test"]]
        sc = [_G["scorer"](s, x) for s, x in zip(_G["test"], hyp0)]
        clean[src] = {k: np.array([c[k] for c in sc]) for k in ("bleu", "wacc", "sim", "exact")}
    _G["clean"] = clean


def _sweep(args):
    """One worker job: one (source, channel, MCS) over the whole SNR grid."""
    src, chan, k = args
    mcs = MCS_TABLE[k]
    sents, sc, c0 = _G["test"], _G["scorer"], _G["clean"][src]
    grid = CONV_SNR_AWGN if chan == "awgn" else CONV_SNR_RAYL
    tx = link.TxFrames(_G["huff"][src], sents, mcs)
    rng = np.random.default_rng(SEED + 1000 * k + (7 if chan == "rayleigh" else 0) + (500 if src == "word" else 0))
    res = {n: [] for n in ("ber", "bler", "bleu", "wacc", "sim", "exact")}
    t0 = time.time()
    for snr in grid:
        r = link.run(tx, snr, rng, chan=chan, want_text=True)
        acc = {m: c0[m].copy() for m in ("bleu", "wacc", "sim", "exact")}     # error-free frames: clean decode
        for f in np.flatnonzero(r["err_frames"]):                           # frames with bit errors: score the garbled text
            s = sc(sents[f], r["text"][f])
            for m in acc:
                acc[m][f] = s[m]
        res["ber"].append(float(r["bit_errors"].sum() / tx.L.sum()))
        res["bler"].append(float(r["err_frames"].mean()))
        for m in acc:
            res[m].append(float(acc[m].mean()))
    print(f"[1] {src:4s} {chan:8s} {mcs['name']:11s} done in {time.time()-t0:.0f}s", flush=True)
    out = {n: v for n, v in res.items()}
    out.update(snr=[float(x) for x in grid], avg_symbols=float(tx.n_sym.mean()), avg_info_bits=float(tx.L.mean()),
               avg_tx_bits=float(tx.n_tx_bits.mean()))
    return src, chan, MCS_TABLE[k]["name"], out


def run_sweeps():
    jobs = [(s, c, k) for s in SOURCES for c in ("awgn", "rayleigh") for k in range(len(MCS_TABLE))]
    nw = int(os.environ.get("EX2_WORKERS", "2"))
    if nw <= 1:
        _init(); results = [_sweep(j) for j in jobs]
    else:
        with Pool(nw, initializer=_init) as p:
            results = p.map(_sweep, jobs, chunksize=1)
    sweeps = {s: {c: {} for c in ("awgn", "rayleigh")} for s in SOURCES}
    for src, chan, name, out in results:
        sweeps[src][chan][name] = out
    if not _G:
        _init()
    clean = {s: {m: float(v.mean()) for m, v in _G["clean"][s].items()} for s in SOURCES}
    json.dump(dict(sweeps=sweeps, clean=clean, n_frames=len(_G["test"])), open(os.path.join(RES_DIR, "task1_conv_sweeps.json"), "w"))
    print("[1] clean (error-free) scores:", clean)
    return sweeps, clean


def main():
    t0 = time.time()
    run_sweeps()
    print(f"[task1-conv] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
