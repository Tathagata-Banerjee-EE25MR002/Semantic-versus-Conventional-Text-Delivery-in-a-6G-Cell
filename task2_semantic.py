"""
task2_semantic.py -- Task 2: train the semantic transceivers (K = 1, 2, 4, 8) and evaluate them.

Stages (python task2_semantic.py --train 8,4 | --eval | --mismatch)
  train     one DeepSC-style model per K (semantic_model.py), saved in data/sem_K<K>.pt
  eval      BLEU / word accuracy / similarity / exact-match versus SNR on the 2000 test sentences,
            AWGN and Rayleigh + ZF -> results/task2_sem_curves.json, similarity LUT eps(K, gamma)
  mismatch  sentences with unseen words / unseen templates (knowledge-base mismatch)
"""
import os, sys, json, time
import numpy as np
import torch
from config import *
from corpus import load_corpus
from tokens import Vocab
import semantic_model as sm


def ckpt(K):
    return os.path.join(DATA_DIR, f"sem_K{K}.pt")


def train_models(Ks, retrain=False):
    tr, te = load_corpus()
    vocab = Vocab(tr)
    for K in Ks:
        if os.path.exists(ckpt(K)) and not retrain:
            print(f"[2] K={K}: checkpoint exists, skipping (EX2_RETRAIN=1 to force)"); continue
        steps = SEM_STEPS_BY_K.get(K, SEM_TRAIN_STEPS)       # longer schedule for K=2 (see config.py)
        model, hist = sm.train_codec(K, vocab, tr, steps=steps, path=ckpt(K))
        json.dump(dict(K=K, hist=hist, params=sm.count_params(model), steps=steps, batch=SEM_BATCH),
                  open(os.path.join(RES_DIR, f"sem_train_hist_K{K}.json"), "w"))


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def decode_sentences(model, vocab, sentences, snr_db, kind="awgn", seed=0, batch=500, fade_per_word=False):
    """Send sentences through the semantic transceiver at one SNR and return the decoded texts.
    The receiver knows the burst length, so it decodes exactly (words + <eos>) positions and cuts at the first <eos>."""
    torch.manual_seed(seed)
    data = sm.encode_set(vocab, sentences)
    out = []
    for a in range(0, len(data), batch):
        chunk = data[a:a + batch]
        ids = sm.pad_batch(chunk)
        logits, mask = model(ids, float(snr_db), kind, fade_per_word)
        pred = logits.argmax(-1)
        for i, seq in enumerate(chunk):
            out.append(vocab.decode(pred[i, :len(seq)].tolist()))
    return out


def evaluate_curve(model, vocab, sentences, scorer, grid, kind="awgn", fade_per_word=False):
    """BLEU, word accuracy, similarity and exact-match fraction versus SNR."""
    res = {m: [] for m in ("bleu", "wacc", "sim", "exact")}
    for snr in grid:
        hyp = decode_sentences(model, vocab, sentences, snr, kind, seed=SEED + int(round(snr * 10)) + 1000, fade_per_word=fade_per_word)
        mean = scorer.mean(sentences, hyp)
        for m in res:
            res[m].append(mean[m])
    res["snr"] = [float(x) for x in grid]
    return res


def evaluate_all(Ks=K_LIST):
    """Curves for every K: AWGN (the training channel), Rayleigh + ZF (mismatched channel), and Rayleigh with an
    independent gain per word.  Also stores the similarity look-up table eps(K, gamma) (= the AWGN curves)."""
    from metrics2 import Scorer
    tr, te = load_corpus()
    vocab = Vocab(tr); scorer = Scorer(tr)
    sents = te[:SEM_EVAL_SENT]
    L_tx = float(np.mean([len(vocab.encode(s)) for s in sents]))        # words + <eos> per sentence (what is transmitted)
    out = dict(L_bar_tx=L_tx, n_sentences=len(sents), K={}, snr_awgn=[float(x) for x in SEM_SNR_GRID])
    old = os.path.join(RES_DIR, "task2_sem_curves.json")
    if list(Ks) != list(K_LIST) and os.path.exists(old):                 # re-evaluating a subset: keep the other K's results
        out["K"] = json.load(open(old))["K"]
    t0 = time.time()
    for K in Ks:
        model = sm.load_codec(K, vocab, ckpt(K))
        r = dict(params=sm.count_params(model), channel_uses=K * L_tx)
        r["awgn"] = evaluate_curve(model, vocab, sents, scorer, SEM_SNR_GRID, "awgn")
        r["rayleigh"] = evaluate_curve(model, vocab, sents, scorer, SEM_SNR_GRID[::2], "rayleigh")
        r["rayleigh_word"] = evaluate_curve(model, vocab, sents, scorer, SEM_SNR_GRID[::4], "rayleigh", fade_per_word=True)
        out["K"][str(K)] = r
        print(f"[2] K={K}: eval done ({time.time()-t0:.0f}s); sim at 0/6/12 dB = " +
              "/".join(f"{r['awgn']['sim'][list(SEM_SNR_GRID).index(v)]:.3f}" for v in (0.0, 6.0, 12.0) if v in list(SEM_SNR_GRID)), flush=True)
    out["K"] = {k: out["K"][k] for k in sorted(out["K"], key=int)}
    json.dump(out, open(os.path.join(RES_DIR, "task2_sem_curves.json"), "w"))
    write_lut_csv(out)
    return out


def write_lut_csv(out):
    """Task 2: the look-up table eps(K, gamma) (mean similarity of the 2000 test sentences, AWGN) as a CSV file."""
    Ks = list(out["K"].keys())
    with open(os.path.join(RES_DIR, "lut_semantic_eps.csv"), "w") as f:
        f.write("snr_db," + ",".join(f"eps[K={k}]" for k in Ks) + "," + ",".join(f"exact[K={k}]" for k in Ks) + "\n")
        for j, s in enumerate(out["snr_awgn"]):
            f.write(f"{s:.1f}," + ",".join(f"{out['K'][k]['awgn']['sim'][j]:.4f}" for k in Ks) + "," +
                    ",".join(f"{out['K'][k]['awgn']['exact'][j]:.4f}" for k in Ks) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--train" in args:
        Ks = [int(k) for k in args[args.index("--train") + 1].split(",")]
        train_models(Ks, retrain=os.environ.get("EX2_RETRAIN", "0") == "1")
    if "--eval" in args:
        i = args.index("--eval")
        evaluate_all([int(k) for k in args[i + 1].split(",")] if len(args) > i + 1 and not args[i + 1].startswith("--") else K_LIST)
