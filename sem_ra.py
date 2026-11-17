"""
sem_ra.py -- Task 3: semantic-aware resource allocation (system model, MILP, greedy heuristic, evaluation).

Problem (equal power p = Pmax/N on every RB, gamma_{u,n} = p a_{u,n}; a = SNR per watt from resource_alloc.generate_drops)
-------
    max   sum_{u,n} x_{u,n} S_{u,n}(K_{u,n})               S_{u,n}(K) = B I eps(K, gamma_{u,n}) / (K Lbar)   [suts/s]
    s.t.  sum_u x_{u,n} <= 1                  (one user per RB)
          eps(K_{u,n}, gamma_{u,n}) >= eps_th  on every used RB
          S_u = sum_n x_{u,n} S_{u,n}(K_{u,n}) >= Smin   for every user
          x_{u,n} in {0,1},  K_{u,n} in {1,2,4,8}.

Why K decouples (the proof requested in Task 3)
-----------------------------------------------
K_{u,n} appears ONLY in the term S_{u,n}(K_{u,n}) and in the admissibility test eps(K_{u,n},gamma_{u,n}) >= eps_th of its own
pair.  Let F_{u,n} = { K : eps(K, gamma_{u,n}) >= eps_th } be the admissible set and
        S~_{u,n} = max_{K in F_{u,n}} S_{u,n}(K)   (= 0 if F_{u,n} is empty).
Take ANY feasible (x, K).  Replacing every K_{u,n} by its maximiser K*_{u,n} keeps all admissibility constraints,
leaves one-user-per-RB untouched and can only increase every S_u (S_u is a sum of non-negative, individually
increased terms), so the objective and the constraints S_u >= Smin are not harmed.  Hence an optimal solution exists with
K = K*(gamma) for all pairs, and the problem is EQUIVALENT to
    max_x  sum x_{u,n} S~_{u,n}   s.t.  sum_u x_{u,n} <= 1,  sum_n x_{u,n} S~_{u,n} >= Smin,  x_{u,n} <= 1[S~_{u,n} > 0],
a MILP with N U binary variables and N + U linear constraints -- the SAME structure as the bit-rate MILP of Exercise 1, only
the rate matrix r_{u,n} = B log2(1+gamma) is replaced by S~_{u,n}.  (verify_decoupling() checks this by brute force.)

Allocators compared (all return a hard assignment X (D,U,N)):
    milp_assign   optimal (HiGHS, exact);  an elastic slack s_u (penalty kappa) keeps it feasible when some user cannot reach Smin
    greedy_assign hardest-user-first heuristic (no solver)
    + 'fixed K=4' (MILP on the rate matrix with K restricted to 4) and 'bit-rate-optimal' (resource_alloc.milp_equal of Exercise 1,
      K chosen afterwards) -- built in task3_semantic_ra.py;  + the imitation DNN of sem_dnn.py.
"""
import time, itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from config import *

KAPPA = 100.0           # slack penalty per k-sut/s (larger than the ratio of best to worst usable pair rate)


# ------------------------------------------------------------------------------------ MILP
def milp_assign_one(S, smin=S_MIN, kappa=KAPPA, time_limit=20.0):
    """S: (U,N) best admissible semantic rate [suts/s] of every pair (0 = pair not admissible).  Returns X (U,N) in {0,1}."""
    Uu, Nn = S.shape
    r = S / 1e3                                                          # k-suts/s keeps the numbers O(1..10)
    nv = Uu * Nn + Uu
    c = np.concatenate([-r.ravel(), kappa * np.ones(Uu)])                # maximise sum x r - kappa sum slack
    A = np.zeros((Nn + Uu, nv))
    for n in range(Nn):
        A[n, [u * Nn + n for u in range(Uu)]] = 1.0                      # one user per RB
    lb = np.full(Nn + Uu, -np.inf)
    ub = np.concatenate([np.ones(Nn), np.full(Uu, np.inf)])
    for u in range(Uu):                                                  # S_u + slack >= Smin
        A[Nn + u, u * Nn:(u + 1) * Nn] = r[u]
        A[Nn + u, Uu * Nn + u] = 1.0
        lb[Nn + u] = smin / 1e3
    xub = (S > 0).astype(float).ravel()                                  # inadmissible pairs (eps < eps_th) cannot be used
    bounds = Bounds(np.zeros(nv), np.concatenate([xub, np.full(Uu, np.inf)]))
    integrality = np.concatenate([np.ones(Uu * Nn), np.zeros(Uu)])
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, mip_rel_gap=1e-6))
    if res.x is None:                                                    # (practically never) fall back to best-user
        x = np.zeros((Uu, Nn)); x[S.argmax(0), np.arange(Nn)] = 1.0
        return x * (S > 0)
    return np.round(res.x[:Uu * Nn]).reshape(Uu, Nn)


def milp_assign(S3, **kw):
    return np.stack([milp_assign_one(S3[d], **kw) for d in range(S3.shape[0])])


# ------------------------------------------------------------------------------------ greedy heuristic
def greedy_assign_one(S, smin=S_MIN):
    """Hardest-user-first greedy (no optimisation solver).
       1. order users by their best single-RB rate, smallest first (the users that are hardest to serve choose first);
       2. each user in that order takes its best free RBs until S_u >= Smin;
       3. every RB that is still free goes to the user with the largest rate on it (rate > 0)."""
    Uu, Nn = S.shape
    X = np.zeros((Uu, Nn)); free = np.ones(Nn, bool); Su = np.zeros(Uu)
    for u in np.argsort(S.max(1)):
        for n in np.argsort(-S[u]):
            if Su[u] >= smin or S[u, n] <= 0:
                break
            if free[n]:
                X[u, n] = 1.0; free[n] = False; Su[u] += S[u, n]
    for n in np.flatnonzero(free):
        u = int(S[:, n].argmax())
        if S[u, n] > 0:
            X[u, n] = 1.0
    return X


def greedy_assign(S3, **kw):
    return np.stack([greedy_assign_one(S3[d], **kw) for d in range(S3.shape[0])])


# ------------------------------------------------------------------------------------ evaluation
def evaluate(X, pr, smin=S_MIN):
    """X (D,U,N) assignment, pr = pair_rates dict (arrays (D,U,N)).  Returns the metrics of Task 3 / Task 4.
    A pair that is not admissible (S = 0) contributes nothing even if the allocator used it."""
    S = pr["S"]
    Ru = (X * S).sum(2)                                                  # (D,U) semantic rate of every user
    sent = (X * pr["sps"]).sum(2)                                        # sentences/s per user
    tot_sent = max(float((X * pr["sps"]).sum()), 1e-9)
    return dict(
        total_rate=float(Ru.sum(1).mean()),                              # mean over drops of the total semantic rate [suts/s]
        pct_served=float(100.0 * np.mean(Ru >= smin * (1 - 1e-9))),      # users with S_u >= Smin
        p5_rate=float(np.percentile(Ru, 5)),                             # cell-edge: 5th-percentile user rate
        mean_rate_user=float(Ru.mean()),
        uses_per_sentence=float((X * pr["sps"] * pr["uses"]).sum() / tot_sent),      # sentence-weighted channel uses
        exact_frac=float((X * pr["sps"] * pr["exact"]).sum() / tot_sent),            # fraction of sentences reproduced exactly
        mean_eps=float((X * pr["sps"] * pr["eps"]).sum() / tot_sent),
        Ru=Ru)


def single_instance_time(fn, n=30):
    """Mean wall-clock time per drop when drops are processed ONE AT A TIME (latency); fn(d) allocates drop d."""
    t = time.perf_counter()
    for d in range(n):
        fn(d)
    return (time.perf_counter() - t) / n


# ------------------------------------------------------------------------------------ brute-force check of the decoupling
def verify_decoupling(trials=30, U_=3, N_=4, seed=0):
    """Brute force on tiny random instances: maximise over ALL assignments x and ALL K choices with the original constraints
    (admissibility, S_u >= Smin) and compare with the MILP on S~ (K eliminated).  Both must give the same optimum."""
    rng = np.random.default_rng(seed)
    Ks = np.array(K_LIST)
    worst = 0.0
    for _ in range(trials):
        eps = np.sort(rng.uniform(0.6, 1.0, (U_, N_, len(Ks))), axis=2)[:, :, ::-1]      # larger K = more robust = higher eps
        base = rng.uniform(0.3, 1.0, (U_, N_, 1)) * 10                                   # k-suts: B I / (K Lbar) with an arbitrary scale
        S_K = base * eps / Ks[None, None, :] * 4
        adm = eps >= EPS_TH
        smin = float(rng.uniform(0.5, 4.0))
        best, found = -1.0, False
        for owner in itertools.product(range(U_ + 1), repeat=N_):                        # owner = U means "RB unused"
            for kc in itertools.product(range(len(Ks)), repeat=N_):
                Su = np.zeros(U_); tot = 0.0; ok = True
                for n in range(N_):
                    if owner[n] == U_:
                        continue
                    if not adm[owner[n], n, kc[n]]:
                        ok = False; break
                    Su[owner[n]] += S_K[owner[n], n, kc[n]]; tot += S_K[owner[n], n, kc[n]]
                if ok and np.all(Su >= smin - 1e-12) and tot > best:
                    best, found = tot, True
        St = np.where(adm, S_K, 0.0).max(2)                                              # S~ : K eliminated
        X = milp_assign_one(St * 1e3, smin=smin * 1e3, kappa=1e6)
        feasible = np.all((X * St).sum(1) >= smin - 1e-9)
        if found and feasible:
            worst = max(worst, abs(best - float((X * St).sum())))
        elif found != feasible:
            worst = max(worst, 1e9)
    return worst


if __name__ == "__main__":
    print("max |brute force - MILP on S~| over 30 random instances:", verify_decoupling())
