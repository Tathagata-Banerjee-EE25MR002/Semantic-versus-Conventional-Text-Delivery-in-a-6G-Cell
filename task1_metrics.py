"""
task1_metrics.py -- Task 1, part 1: corpus statistics and "what do the metrics really measure?".

(a) corpus statistics: vocabulary size, average sentence length (words), word entropy.
(b) 25 sentence pairs = 5 base sentences x 5 kinds of change:
        exact copy | synonym substitution | change of word order | one wrong number or place | unrelated sentence
    scored with BLEU-4, word accuracy and sentence similarity.
    What to see: BLEU / word accuracy are LEXICAL (a synonym looks like an error, and so does a harmless
    re-ordering), the similarity is SEMANTIC (synonyms stay close) -- but a bag-of-embeddings similarity is
    blind to word order and almost blind to a wrong number or place.  This limitation is reported in the
    paper because eps_th = 0.9 would accept those errors.
"""
import os, json, csv
import numpy as np
from config import *
from corpus import load_corpus
from tokens import Vocab, corpus_stats
from metrics2 import Scorer

# (base sentence, synonym, re-ordering, wrong number/place, unrelated)
PAIR_SETS = [
    ("the committee approved the new budget in london on friday .",
     "the committee accepted the fresh budget in london on friday .",
     "on friday in london the committee approved the new budget .",
     "the committee approved the new budget in paris on friday .",
     "my dog likes to run on the beach in the summer ."),
    ("the doctor told the patient to rest at home for 5 days .",
     "the physician advised the patient to relax at home for 5 days .",
     "for 5 days the patient was told by the doctor to rest at home .",
     "the doctor told the patient to rest at home for 50 days .",
     "the stock market fell sharply after the announcement yesterday ."),
    ("she bought a big car in boston last year .",
     "she purchased a large automobile in boston last year .",
     "last year in boston she bought a big car .",
     "she bought a big car in chicago last year .",
     "the river flows through the valley and into the sea ."),
    ("the train from chicago arrives at 6 pm on monday .",
     "the railway from chicago reaches at 6 pm on monday .",
     "on monday at 6 pm the train arrives from chicago .",
     "the train from chicago arrives at 8 pm on monday .",
     "please close the window before you leave the house ."),
    ("the children were happy to see their father after 3 years .",
     "the kids were glad to see their dad after 3 years .",
     "after 3 years the children were happy to see their father .",
     "the children were happy to see their father after 30 years .",
     "three teams will play in the final game tomorrow ."),
]
KINDS = ["exact copy", "synonym", "word order", "wrong number/place", "unrelated"]


def sentence_pair_table(scorer):
    rows = []
    for base, *variants in PAIR_SETS:
        for kind, hyp in zip(KINDS, [base] + variants):
            sc = scorer(base, hyp)
            rows.append(dict(kind=kind, ref=base, hyp=hyp, **{k: round(sc[k], 4) for k in ("bleu", "wacc", "sim")}))
    summary = {k: {m: float(np.mean([r[m] for r in rows if r["kind"] == k])) for m in ("bleu", "wacc", "sim")} for k in KINDS}
    return rows, summary


def main():
    tr, te = load_corpus()
    vocab = Vocab(tr)
    stats = corpus_stats(tr, te, vocab)
    stats["avg_len_tx_test"] = float(np.mean([len(vocab.encode(s)) for s in te]))        # tokens + <eos>, test set
    json.dump(stats, open(os.path.join(RES_DIR, "task1_corpus_stats.json"), "w"), indent=1)
    print("[1] corpus:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
    scorer = Scorer(tr)
    rows, summary = sentence_pair_table(scorer)
    with open(os.path.join(RES_DIR, "task1_sentence_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(RES_DIR, "task1_pair_summary.json"), "w"), indent=1)
    print(f"[1] {len(rows)} sentence pairs scored; mean per kind (BLEU / word acc / similarity):")
    for k, v in summary.items():
        print(f"    {k:20s} {v['bleu']:.2f}  {v['wacc']:.2f}  {v['sim']:.2f}")
    return stats, rows, summary


if __name__ == "__main__":
    main()
