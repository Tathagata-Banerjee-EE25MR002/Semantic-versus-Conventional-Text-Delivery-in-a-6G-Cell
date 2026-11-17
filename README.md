# Exercise 2 — Deep-learning semantic communication vs. the conventional link, with semantic-aware resource allocation

Everything needed to reproduce the figures, numbers, paper and slides of Exercise 2.
It builds on Exercise 1 (same cell, channel, corpus, modules `link.py`, `resource_alloc.py`, ...).

```
Exercise2/
├── code/          Python modules (one file per block, heavily commented) + run_all.py
├── notebooks/     Exercise2_Semantic_Communication.ipynb   (same code, markdown explanation above EVERY code cell)
├── paper/         main.tex, main.pdf         5-page IEEE double-column conference paper
├── slides/        slides.tex / slides.pdf (16:9 Beamer), slides_with_notes.pdf (speaker script)
├── results/       JSON / CSV with every number quoted in the paper and the slides
├── figures/       paper/*.pdf (IEEE column sized) and slide/*.png
└── data/          corpus split, fastText model, trained semantic models sem_K{1,2,4,8}.pt, imitation DNN
```

## Run it

```
pip install -r requirements.txt

# (A) scripts: ONE command reproduces every figure and result and saves the trained models
cd code
python run_all.py                    # full run (about 3.5 h on 2 CPU cores, see table below)
python run_all.py --only 3 4 figures # selected stages: 1 | 2 | 3 | 4 | figures (existing models are re-used)
EX1_QUICK=1 EX2_ROOT=/tmp/ex2_quick python run_all.py   # smoke test, ~12 min, separate output folder

# (B) notebook
cd notebooks
jupyter notebook Exercise2_Semantic_Communication.ipynb
```

The notebook has three run modes (environment variable `EX2_MODE` or the first code cell):

| mode | what happens | time |
|---|---|---|
| `load` (default) | uses the saved `results/` and trained models in `data/`, regenerates all figures | ~2 min |
| `quick` | re-runs every experiment at reduced size in a separate folder `quick_run/` | ~12 min |
| `full` | re-runs everything at the size reported in the paper (trains the four models) | ~3.5 h |

Each `%%module <name>` cell of the notebook holds the verbatim source of `code/<name>.py`
(`notebooks/build_notebook.py` regenerates the notebook after a module is edited), so notebook and scripts are the same code.

Existing trained models are re-used; delete `data/sem_K*.pt` (or set `EX2_RETRAIN=1`) to retrain.
Time of the full run: Task 1 sweeps 14 min, training K = 8/4/1 ≈ 30 min each and K = 2 ≈ 55 min (7 000 steps), evaluation ≈ 25 min,
Task 3 ≈ 8 min (incl. 20 000 MILP labels), Task 4 ≈ 10 min. All randomness is seeded (`config.SEED`).
Build the paper with `cd paper && pdflatex main.tex` (twice); the slides with `cd slides && pdflatex slides.tex` (twice); the speaker script with
`pdflatex -jobname=slides_with_notes "\def\withnotes{}\input{slides.tex}"` (twice).

## Code map (every file starts with a docstring that explains the design)

| file | content | task |
|---|---|---|
| `config.py` | all parameters of Table 1, training / experiment sizes (single source of truth; `EX2_ROOT` redirects outputs) | all |
| `corpus.py`, `source_coding.py`, `channel_coding.py`, `modem.py`, `channel.py`, `link.py`, `resource_alloc.py` | the Exercise-1 chain and cell, re-used unchanged | 1, 3 |
| `tokens.py` | word/punctuation tokeniser, vocabulary = knowledge base, corpus statistics | 1 |
| `word_huffman.py` | word-level Huffman source code (KB-dependent) | 1 |
| `text_metrics.py`, `metrics2.py` | BLEU, word accuracy, fastText similarity; one `Scorer` for **all** systems | 1 |
| `task1_metrics.py` | corpus statistics + 25 sentence pairs (5 kinds) | 1 |
| `task1_conventional.py` | conventional link judged by meaning: BLEU / similarity vs SNR, 2 source coders x 6 MCS x AWGN / Rayleigh | 1 |
| `semantic_model.py` | DeepSC-style transceiver (Transformer + channel codec), training loop | 2 |
| `task2_semantic.py` | training of K = 1, 2, 4, 8, evaluation curves, LUT `results/lut_semantic_eps.csv` | 2 |
| `kb_mismatch.py` | unseen words / unseen templates test | 2 |
| `rate_tables.py` | SNR → semantic rate look-up tables for **both** systems (PAV monotone, AMC and meaning-aware rules) | 3, 4 |
| `sem_ra.py` | decoupling proof, exact MILP, greedy heuristic, evaluation, brute-force verification | 3 |
| `sem_dnn.py` | permutation-equivariant imitation network, dataset, training, repair step | 3 |
| `task3_semantic_ra.py` | Task-3 experiment (6 allocators, 11 powers, 300 drops, timing) | 3 |
| `task4_compare.py` | common-protocol comparison, bandwidth, robustness, complexity | 4 |
| `make_figures2.py` | every figure (paper PDF and slide PNG) from the saved results | all |
| `run_all.py` | one command for everything | all |

## Headline results (30 dBm, 300 common drops, 2 000 test sentences)

| system | rate [ksuts/s] | users served | channel uses / sentence | exact |
|---|---|---|---|---|
| semantic, adaptive K (MILP) | 76.1 | 99.5 % | 35.0 | 4.8 % |
| semantic, fixed K = 4 | 41.5 | 98.4 % | 67.4 | 23.7 % |
| conventional, char Huffman + AMC | 34.9 | 99.0 % | 82.6 | 99.8 % |
| conventional, word Huffman + AMC | 80.9 | 99.4 % | 34.9 | 33.2 % |

Greedy is within 0.05 % of the MILP (80x faster); the imitation DNN serves only 66 % of the users until a repair step
brings it to 99 % (within 1 % of the optimum). See the paper for all comparisons.

## Honest notes

* Quick mode (150 training steps) only proves that the pipeline runs: its weak semantic models never reach ε = 0.9, so semantic rates/bandwidths there are empty (NaN) — use the saved models or the full run for numbers.
* The Transformer semantic decoder is **non-autoregressive** (all words in parallel) and small (2 layers, width 128):
  a deliberate simplification so that four models can be trained on a CPU. K = 1, 2, 4, 8 use 3 000 steps (K = 2: 7 000, because with
  3 000 steps its similarity saturated at 0.88 < ε_th = 0.9).
* Sentence similarity uses fastText embeddings trained on the corpus (the pre-trained SBERT model could not be downloaded offline).
  It is nearly blind to word order and to wrong numbers/places (Task 1 shows this); it also hides the effect of unknown words.
* K = 1 never reaches ε = 0.9 (similarity ceiling 0.69), so it is never admissible; K = 2 needs ≥ 15 dB.
* Under the common protocol the conventional word-Huffman link with AMC is 6 % *better* in total semantic rate than the adaptive-K semantic
  system; the semantic system wins against character-level Huffman (2.2x) and fixed K (1.8x) and below the cliff SNR of each MCS.
* Only 33 % of the test sentences are free of out-of-vocabulary words, so no word-level system can reproduce more than 33 % exactly.
* The imitation DNN does not guarantee feasibility; the repair step is a cheap post-processing, reported separately.
* Fading is block-constant per RB, models are trained on AWGN only (Rayleigh + ZF is evaluated without retraining).

## Note on the package
`data/fasttext.model*` (800 MB) is not included: it is re-trained automatically (seeded, about 1 min) from `data/sentences.json`
the first time the similarity metric is used. `data/sem_imit_dataset.npz` (MILP labels) is regenerated by `task3_semantic_ra.py` when needed.
