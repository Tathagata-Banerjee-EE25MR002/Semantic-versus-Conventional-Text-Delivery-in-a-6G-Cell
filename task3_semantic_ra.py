"""
task3_semantic_ra.py -- Task 3 experiment: semantic-aware resource allocation on the SAME 300 drops as Exercise 1.

For every Pmax in 15 ... 40 dBm (equal power Pmax/N on all RBs) the following allocators are compared:
    MILP (optimal)            exact optimum with adaptive K (K decouples per pair -> sem_ra.py)
    Greedy                    hardest-user-first heuristic, adaptive K
    Fixed K = 4 (MILP)        the same optimal solver but K forced to 4 on every RB
    Bit-rate-optimal + K      assignment of Exercise 1 (MILP maximising sum B log2(1+gamma) with Rmin = 1 Mbit/s,
                              resource_alloc.milp_equal), the best admissible K is chosen AFTERWARDS per RB
    DNN (imitation)           network trained on MILP labels (sem_dnn.py), one forward pass (raw arg-max per RB)
    DNN + repair              the same plus a cheap deficit-repair step that moves RBs to users below Smin
Metrics: total semantic rate, % users with S_u >= Smin, 5th-percentile user rate, channel uses per sentence,
fraction of sentences reproduced exactly, optimality gap to the MILP, computation time per drop.
"""
import os, json, time
import numpy as np
from config import *
import resource_alloc as ra
import sem_ra
import sem_dnn
from rate_tables import SemanticRates

METHODS = ["MILP (optimal)", "Greedy", "Fixed K=4 (MILP)", "Bit-rate-optimal + K", "DNN (imitation)", "DNN + repair"]


def get_dnn(sem):
    path = os.path.join(DATA_DIR, "sem_imit_dnn.pt")
    if os.path.exists(path) and os.environ.get("EX2_RETRAIN", "0") != "1":
        print("[3] loaded imitation DNN"); return sem_dnn.load(path), None
    ds = os.path.join(DATA_DIR, "sem_imit_dataset.npz")
    S, p, lab = sem_dnn.make_dataset(RA2_IMIT_TRAIN, sem, SEED + 3, path=ds)
    net, hist = sem_dnn.train_imitation(S, p, lab, path=path)
    json.dump(hist, open(os.path.join(RES_DIR, "task3_sem_train_hist.json"), "w"))
    return net, hist


def main():
    t_all = time.time()
    sem = SemanticRates()
    net, _ = get_dnn(sem)
    rng = np.random.default_rng(SEED + 77)                      # the very same generator call as Exercise 1 -> same drops
    a = ra.generate_drops(rng, RA2_DROPS)
    D = a.shape[0]
    keys = ("total_rate", "pct_served", "p5_rate", "uses_per_sentence", "exact_frac", "mean_eps")
    out = {m: {k: [] for k in keys + ("objective",)} for m in METHODS}
    out["pmax_dbm"] = [float(x) for x in PMAX_DBM_SWEEP]
    out["dnn_rb_agreement"] = []
    out["k_usage"] = []
    for pdbm in PMAX_DBM_SWEEP:
        P = ra.dbm_to_w(pdbm)
        gam = 10 * np.log10(np.maximum(a * (P / N), 1e-30))
        pr = sem.pair_rates(gam)                                 # adaptive K
        pr4 = sem.pair_rates(gam, Ks=[4])                        # K fixed to 4
        Xm = sem_ra.milp_assign(pr["S"])
        Xd = sem_dnn.dnn_assign(net, pr["S"], pdbm)
        alloc = {
            METHODS[0]: (Xm, pr),
            METHODS[1]: (sem_ra.greedy_assign(pr["S"]), pr),
            METHODS[2]: (sem_ra.milp_assign(pr4["S"]), pr4),
            METHODS[3]: (ra.milp_equal(a, P)[0], pr),
            METHODS[4]: (Xd, pr),
            METHODS[5]: (sem_dnn.repair(Xd, pr["S"]), pr),
        }
        for m, (X, p_) in alloc.items():
            ev = sem_ra.evaluate(X, p_)
            for k in keys:
                out[m][k].append(ev[k])
            # objective actually maximised by the MILP: total rate minus kappa * sum of shortfalls (all in suts/s)
            short = np.maximum(S_MIN - ev["Ru"], 0.0).sum(1).mean()
            out[m]["objective"].append(float(ev["total_rate"] - sem_ra.KAPPA * short))
        out["dnn_rb_agreement"].append(float(np.mean((alloc[METHODS[4]][0] * Xm).sum(1) > 0.5)))
        used = (Xm > 0.5) & (pr["mode"] > 0)
        out["k_usage"].append({str(K): float(np.mean(pr["mode"][used] == K)) for K in K_LIST})
        print(f"[3] Pmax {pdbm:5.1f} dBm | " + " | ".join(
            f"{m.split()[0][:7]} {out[m]['total_rate'][-1]/1e3:6.1f}k {out[m]['pct_served'][-1]:5.1f}%" for m in METHODS), flush=True)
    base = np.array(out[METHODS[0]]["total_rate"])
    for m in METHODS:
        out[m]["gap_pct"] = [float(100 * (1 - r / b)) for r, b in zip(out[m]["total_rate"], base)]
    obase = np.array(out[METHODS[0]]["objective"])
    for m in METHODS:                                          # gap in the penalised objective (never negative)
        out[m]["obj_gap_pct"] = [float(100 * (1 - r / b)) for r, b in zip(out[m]["objective"], obase)]

    # ---- computation time per drop at 30 dBm, ONE drop at a time (what a scheduler sees)
    P = ra.dbm_to_w(30.0)
    gam = 10 * np.log10(np.maximum(a * (P / N), 1e-30))
    pr = sem.pair_rates(gam); pr4 = sem.pair_rates(gam, Ks=[4])
    nt = 10 if QUICK else 40
    fns = {
        METHODS[0]: lambda d: sem_ra.milp_assign_one(pr["S"][d]),
        METHODS[1]: lambda d: sem_ra.greedy_assign_one(pr["S"][d]),
        METHODS[2]: lambda d: sem_ra.milp_assign_one(pr4["S"][d]),
        METHODS[3]: lambda d: ra.milp_equal(a[d:d + 1], P),
        METHODS[4]: lambda d: sem_dnn.dnn_assign(net, pr["S"][d:d + 1], 30.0),
        METHODS[5]: lambda d: sem_dnn.repair(sem_dnn.dnn_assign(net, pr["S"][d:d + 1], 30.0), pr["S"][d:d + 1]),
    }
    out["time_ms_per_drop"] = {m: 1e3 * sem_ra.single_instance_time(fns[m], nt) for m in METHODS}
    t = time.perf_counter(); sem_dnn.dnn_assign(net, pr["S"], 30.0); out["dnn_batched_ms_per_drop"] = 1e3 * (time.perf_counter() - t) / D
    out["decoupling_max_abs_error"] = sem_ra.verify_decoupling(trials=10 if QUICK else 30)
    out["n_drops"] = D
    json.dump(out, open(os.path.join(RES_DIR, "task3_sem_results.json"), "w"))
    print("[3] time/drop at 30 dBm [ms]:", {k: round(v, 2) for k, v in out["time_ms_per_drop"].items()})
    print(f"[task3] done in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
