"""
run_all.py -- ONE command that reproduces every number and figure of Exercise 2 and saves the trained models.

    python run_all.py                 full run (about 3.5 h on 2 CPU cores: training 3 x 3000 + 7000 steps, all sweeps, all MILPs)
    EX1_QUICK=1 EX2_ROOT=/tmp/ex2_quick python run_all.py     smoke test with reduced sizes (about 12 min) in a separate folder
                                      (copy data/sentences.json and data/fasttext.model* there first so that the same corpus and metric are used)
    python run_all.py --only 3 figures   selected stages: 1 | 2 | 3 | 4 | figures

Stages (each one writes into results/, data/ and figures/):
    1  Task 1  corpus statistics, 25 sentence pairs, conventional links (char / word Huffman, 6 MCS, AWGN + Rayleigh)
    2  Task 2  train the four semantic transceivers (K = 1,2,4,8) -> data/sem_K*.pt, evaluate, LUT, KB mismatch
    3  Task 3  semantic-aware resource allocation: MILP, greedy, fixed K, bit-rate-optimal + K, imitation DNN
    4  Task 4  protocol-matched comparison, bandwidth, robustness, complexity
    figures    every figure of the paper and the slides
Existing trained models are re-used (delete data/sem_K*.pt or set EX2_RETRAIN=1 to retrain).
"""
import sys, time


def main():
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else ["1", "2", "3", "4", "figures"]
    t0 = time.time()
    if "1" in only:
        import task1_metrics, task1_conventional
        task1_metrics.main(); task1_conventional.main()
    if "2" in only:
        import task2_semantic, kb_mismatch
        from config import K_LIST
        task2_semantic.train_models(K_LIST); task2_semantic.evaluate_all(); kb_mismatch.main()
    if "3" in only:
        import task3_semantic_ra
        task3_semantic_ra.main()
    if "4" in only:
        import task4_compare
        task4_compare.main()
    if "figures" in only:
        import make_figures2
        make_figures2.main()
    print(f"[run_all] finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
