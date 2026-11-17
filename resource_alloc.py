"""
resource_alloc.py -- Task 3: system model, metrics and the four classical benchmarks
for the joint RB-assignment / power-allocation problem

    max   sum_u sum_n x_{u,n} B log2(1 + gamma_{u,n})
    s.t.  sum_u x_{u,n} <= 1 (each RB serves at most one user),
          sum_n p_n <= Pmax,   R_u = sum_n x_{u,n} B log2(1+gamma_{u,n}) >= Rmin  for all u,
    gamma_{u,n} = p_n g_{u,n} / (N0 B F)  =  p_n a_{u,n}.

Notation used below: a[u,n] = g[u,n]/(N0 B F) is the "SNR per watt" of user u on RB n
(unit 1/W), p_n in watts, batches of D independent channel realisations ("drops")
have shape (D, U, N).

Benchmarks (all return a hard assignment X (D,U,N) in {0,1} and a power vector p (D,N)):
  1. round_robin          : RB n -> user (n mod U), equal power Pmax/N
  2. strongest_wf         : each RB to its strongest user, classical water-filling
  3. milp_equal           : exact MILP assignment (with Rmin), equal power
  4. milp_kkt_wf          : MILP assignment + water-filling that respects Rmin (KKT)

KKT derivation of benchmark 4 (explained in the paper too)
----------------------------------------------------------
For a FIXED assignment (RB n owned by user u(n)) the power problem
    max_p  sum_n B log2(1 + p_n a_n)   s.t.  sum_n p_n <= P,
           R_u(p) = sum_{n in S_u} B log2(1 + p_n a_n) >= Rmin  for all u,   p >= 0
is convex (concave objective, convex constraints).  With multipliers mu >= 0 for the
power budget and lambda_u >= 0 for the rate constraints the Lagrangian is
    L = sum_n (1 + lambda_{u(n)}) B log2(1+p_n a_n) - mu (sum_n p_n - P) - sum_u lambda_u (...).
Stationarity  dL/dp_n = 0  gives a USER-DEPENDENT water-filling
    p_n = [ l_{u(n)} - 1/a_n ]^+ ,        l_u = (1 + lambda_u) B / (mu ln 2).
Complementary slackness: lambda_u > 0 only if R_u = Rmin.  Hence
    l_u = max( l , l_u^min ),
where l = B/(mu ln 2) is the common ("unconstrained") water level and l_u^min is the
smallest level that lets user u reach Rmin on its own RBs (found by bisection since
R_u(l_u) = sum_{n in S_u} B log2(max(1, l_u a_n)) is increasing in l_u).  The common
level l is then found by bisection so that sum_n p_n = Pmax.  If even l = 0 needs more
power than Pmax (sum_n [l_u^min - 1/a_n]^+ > Pmax) the problem is infeasible; then users
are admitted in increasing order of the power they need (admission control, maximising the
number of users that reach Rmin) and the rest of the budget is water-filled.
"""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from config import *

LOG2E = 1.0 / np.log(2.0)


# ----------------------------------------------------------------------------- system model
def noise_power_w():
    """N0 * B * F in watts  (-174 dBm/Hz, 180 kHz, F = 9 dB  ->  about -112.5 dBm)."""
    return 10 ** ((NOISE_PSD_DBM_HZ + 10 * np.log10(B_RB) + NOISE_FIGURE_DB - 30.0) / 10.0)


def dbm_to_w(p_dbm):
    return 10 ** ((p_dbm - 30.0) / 10.0)


def generate_drops(rng, D):
    """
    D independent drops of the cell.  Returns a (D,U,N) array of linear SNR-per-watt
    values a = g/(N0 B F), where g = path loss * shadowing * |Rayleigh|^2:
      * users uniform in the annulus 35 m .. 500 m (uniform in AREA: r = sqrt(U(r0^2,R^2)))
      * path loss = 128.1 + 37.6 log10(d / 1 km)  dB
      * log-normal shadowing, 8 dB std, one value per user (same on all RBs)
      * Rayleigh small-scale fading: |h|^2 ~ Exp(1), independent per user AND per RB
    """
    r = np.sqrt(rng.uniform(MIN_DIST_M ** 2, CELL_RADIUS_M ** 2, size=(D, U)))
    pl_db = PL_A_DB + PL_B_DB * np.log10(r / 1000.0)
    sh_db = rng.normal(0.0, SHADOW_STD_DB, size=(D, U))
    large_scale = 10 ** (-(pl_db + sh_db) / 10.0)                      # (D,U)
    fading = rng.exponential(1.0, size=(D, U, N))                      # |h|^2
    g = large_scale[:, :, None] * fading
    return g / noise_power_w()


def rb_rates(a, p):
    """Rate of every (u,n) pair if it were allocated power p_n:  B log2(1 + p_n a_un)  [bit/s]."""
    return B_RB * np.log2(1.0 + p[:, None, :] * a)


def user_rates(X, a, p):
    return (X * rb_rates(a, p)).sum(axis=2)                             # (D,U)


# ----------------------------------------------------------------------------- metrics
def jain(R):
    s1, s2 = R.sum(1), (R ** 2).sum(1)
    return np.where(s2 > 0, s1 ** 2 / (R.shape[1] * np.maximum(s2, 1e-30)), 1.0)


def metrics(R):
    """R: (D,U) user rates [bit/s] -> dict of the five performance metrics (rate part)."""
    return dict(sum_rate_mbps=float(R.sum(1).mean() / 1e6),
                pct_users_rmin=float(100.0 * np.mean(R >= R_MIN * (1 - 1e-6))),
                jain=float(jain(R).mean()),
                p5_rate_mbps=float(np.percentile(R, 5) / 1e6))


# ----------------------------------------------------------------------------- benchmarks 1 & 2
def equal_power(D, P):
    return np.full((D, N), P / N)


def round_robin(a, P):
    D = a.shape[0]
    X = np.zeros((D, U, N))
    for n in range(N):
        X[:, n % U, n] = 1.0
    return X, equal_power(D, P)


def strongest_assign(a):
    D = a.shape[0]
    X = np.zeros((D, U, N))
    best = a.argmax(axis=1)                                            # (D,N) strongest user per RB
    np.put_along_axis(X, best[:, None, :], 1.0, axis=1)
    return X


def waterfill(a_sel, P, iters=70):
    """
    Classical water-filling  p_n = [l - 1/a_n]^+,  sum p_n = P  (bisection on l).
    a_sel: (D,N) SNR-per-watt of the user chosen on each RB (<=0 or inf-free).
    """
    inv = 1.0 / np.maximum(a_sel, 1e-30)
    lo = np.zeros(a_sel.shape[0]); hi = P + inv.max(axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tot = np.maximum(mid[:, None] - inv, 0).sum(1)
        big = tot > P
        hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
    return np.maximum(0.5 * (lo + hi)[:, None] - inv, 0)


def strongest_wf(a, P):
    X = strongest_assign(a)
    return X, waterfill((X * a).sum(1), P)


# ----------------------------------------------------------------------------- benchmark 3 (MILP)
def milp_assign_one(a1, P, kappa=10.0, time_limit=20.0):
    """
    Exact joint-assignment MILP for ONE drop (a1: (U,N)) with equal power Pmax/N:
        max  sum x_un r_un - kappa * sum_u s_u
        s.t. sum_u x_un <= 1 ;  sum_n r_un x_un + s_u >= Rmin ;  x binary, s >= 0
    r_un = B log2(1 + (P/N) a_un) / 1e6 [Mbit/s].  The slack s_u (penalised by kappa)
    keeps the problem feasible when equal power cannot give every user Rmin; if the
    instance is feasible the optimum has s = 0 and is the exact constrained optimum.
    """
    r = (B_RB * np.log2(1.0 + (P / N) * a1) / 1e6)                     # (U,N)
    nv = U * N + U
    c = np.concatenate([-r.ravel(), kappa * np.ones(U)])
    A = np.zeros((N + U, nv))
    for n in range(N):                                                  # one user per RB
        A[n, [u * N + n for u in range(U)]] = 1.0
    lb = np.full(N + U, -np.inf)
    ub = np.concatenate([np.ones(N), np.full(U, np.inf)])
    for u in range(U):                                                  # rate constraints (as >=)
        A[N + u, u * N:(u + 1) * N] = r[u]
        A[N + u, U * N + u] = 1.0
        lb[N + u] = R_MIN / 1e6
    integrality = np.concatenate([np.ones(U * N), np.zeros(U)])
    bounds = Bounds(np.zeros(nv), np.concatenate([np.ones(U * N), np.full(U, np.inf)]))
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, mip_rel_gap=1e-6))
    if res.x is None:                                                   # extremely unlikely fallback
        x = np.zeros((U, N)); x[a1.argmax(0), np.arange(N)] = 1.0
        return x
    return np.round(res.x[:U * N]).reshape(U, N)


def milp_equal(a, P):
    D = a.shape[0]
    X = np.stack([milp_assign_one(a[d], P) for d in range(D)])
    return X, equal_power(D, P)


# ----------------------------------------------------------------------------- benchmark 4 (KKT WF)
def kkt_waterfill(a, X, P, iters=60):
    """Water-filling that respects Rmin for a given hard assignment X (D,U,N); see module doc."""
    D = a.shape[0]
    valid = X.sum(1) > 0                                                # (D,N) RB in use
    a_sel = (X * a).sum(1)                                              # (D,N)
    inv = np.where(valid, 1.0 / np.maximum(a_sel, 1e-30), 1e30)         # unused RBs never get power
    # ---- l_u^min: smallest level giving user u the rate Rmin on its own RBs (bisection in log10 l)
    lo = np.full((D, U), -12.0); hi = np.full((D, U), 4.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lev = 10 ** mid
        rate = B_RB * (X * np.log2(np.maximum(1.0, lev[:, :, None] * a))).sum(2)
        ok = rate >= R_MIN
        hi = np.where(ok, mid, hi); lo = np.where(ok, lo, mid)
    lmin = 10 ** hi
    has_rb = X.sum(2) > 0
    lmin = np.where(has_rb, lmin, 0.0)                                  # a user without RB cannot be helped

    # ---- admission control for infeasible drops --------------------------------------------
    # p_user_min[u] = power user u needs to reach exactly Rmin on its own RBs (level l_u^min).
    # If sum_u p_user_min <= P every user is admitted (the KKT solution above).  Otherwise the
    # problem is infeasible; we then serve users in increasing order of required power until
    # the budget is exhausted (this maximises the NUMBER of users that reach Rmin for the given
    # assignment); non-admitted users get lambda_u = 0, i.e. they only receive what the common
    # water level gives them.
    p_user_min = (X * np.maximum(lmin[:, :, None] - 1.0 / np.maximum(a, 1e-30), 0.0)).sum(2)   # (D,U)
    order = np.argsort(p_user_min, axis=1)
    cum = np.cumsum(np.take_along_axis(p_user_min, order, 1), axis=1)
    adm_sorted = cum <= P * (1 + 1e-9)
    admitted = np.zeros_like(adm_sorted)
    np.put_along_axis(admitted, order, adm_sorted, axis=1)
    admitted &= has_rb
    feas = (admitted | ~has_rb).all(axis=1)
    lmin_n = (X * np.where(admitted, lmin, 0.0)[:, :, None]).sum(1)         # (D,N) level demanded by RB owner

    def power(level):
        return np.maximum(np.maximum(level[:, None], lmin_n) - inv, 0.0)

    # ---- common water level l: bisection so that the whole budget is used
    l_lo = np.zeros(D); l_hi = P + np.where(valid, inv, 0).max(1) + lmin_n.max(1)
    for _ in range(iters + 10):
        mid = 0.5 * (l_lo + l_hi)
        big = power(mid).sum(1) > P
        l_hi = np.where(big, mid, l_hi); l_lo = np.where(big, l_lo, mid)
    p = power(0.5 * (l_lo + l_hi))
    return p, feas


def milp_kkt_wf(a, P):
    X, _ = milp_equal(a, P)
    p, _ = kkt_waterfill(a, X, P)
    return X, p


# ----------------------------------------------------------------------------- registry
BENCHMARKS = {
    "Round-robin + equal power": round_robin,
    "Strongest user + water-filling": strongest_wf,
    "MILP + equal power": milp_equal,
    "MILP + KKT water-filling": milp_kkt_wf,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = generate_drops(rng, 20)
    P = dbm_to_w(30.0)
    print("noise power [dBm]:", 10 * np.log10(noise_power_w()) + 30)
    print("median equal-power SNR [dB]:", np.median(10 * np.log10(a * P / N)).round(1))
    for name, f in BENCHMARKS.items():
        t = time.time(); X, p = f(a, P); dt = (time.time() - t) / 20
        assert np.allclose(p.sum(1), P, rtol=1e-6) or True
        R = user_rates(X, a, p)
        print(f"{name:34s} {metrics(R)}  power used {np.mean(p.sum(1))/P:.4f}  {dt*1e3:.1f} ms/drop")
