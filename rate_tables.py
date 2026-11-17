"""
rate_tables.py -- look-up tables (LUT) that turn an SNR into a semantic rate, for EVERY system (Task 3 / Task 4).

The resource allocator only needs, for every user-RB pair, the best "mode" the link can run at the pair's SNR
gamma = p_n a_{u,n} and what it delivers.  This file converts the measured curves into exactly that:

                         semantic system                          conventional system
   mode                  K in {1,2,4,8} symbols / word            one of 6 MCS (QPSK 1/2 ... 64-QAM 3/4)
   quality eps           eps(K, gamma) = mean similarity          eps(MCS, gamma) = mean similarity of the received text
   sentences per second  B / (K Lbar)                             B / (symbols per sentence of the MCS)
   semantic rate S       B I eps / (K Lbar)  [suts/s]             sentences/s x eps   (Task-4 definition)
   admissible if         eps >= eps_th                            eps >= eps_th  and the MCS is chosen by the AMC rule
   mode choice           largest S among admissible K             AMC = highest spectral efficiency with BLER <= 10 %
                                                                   ('meaning-aware' variant: largest S among admissible MCS)

Measured curves are noisy (finite number of sentences) so each LUT column is made monotone in SNR with
isotonic regression (pool-adjacent-violators): similarity and exact-match fraction non-decreasing, BLER non-increasing.
Between grid points the LUT is linearly interpolated; below / above the grid the end values are used.

Every function accepts arrays of any shape (D drops, U users, N RBs) and returns arrays of the same shape.
"""
import os, json
import numpy as np
from config import *


def pav_increasing(y):
    """Isotonic (non-decreasing) least-squares fit by pool-adjacent-violators."""
    y = np.asarray(y, float)
    vals, wts = [], []
    for v in y:
        vals.append(v); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]; v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:] = [v2]; wts[-2:] = [w]
    return np.repeat(vals, np.array(wts, int))


def monotone(y, increasing=True):
    return pav_increasing(y) if increasing else -pav_increasing(-np.asarray(y, float))


def _thresholds(snr, bler):
    """SNR at which BLER stays <= BLER_TARGET (log-domain interpolation) -- same rule as Exercise 1."""
    thr = []
    for b in bler:
        bad = np.flatnonzero(b > BLER_TARGET)
        j = (bad[-1] + 1) if len(bad) else 0
        if j >= len(snr):
            thr.append(np.inf); continue
        if j == 0:
            thr.append(float(snr[0])); continue
        y0, y1 = np.log10(max(b[j - 1], 1e-6)), np.log10(max(b[j], 1e-6))
        t = (np.log10(BLER_TARGET) - y0) / (y1 - y0) if y1 != y0 else 1.0
        thr.append(float(snr[j - 1] + t * (snr[j] - snr[j - 1])))
    return np.array(thr)


# ------------------------------------------------------------------------------------ semantic
class SemanticRates:
    """eps(K, gamma) and the semantic rate of the semantic transceivers."""

    def __init__(self, path=None, Ks=K_LIST, L_bar=None):
        d = json.load(open(path or os.path.join(RES_DIR, "task2_sem_curves.json")))
        self.Ks = list(Ks)
        self.L = float(L_bar if L_bar is not None else d["L_bar_tx"])         # transmitted words per sentence (+<eos>)
        self.snr = np.array(d["K"][str(self.Ks[0])]["awgn"]["snr"])
        self.eps = {K: monotone(d["K"][str(K)]["awgn"]["sim"]) for K in self.Ks}
        self.exact = {K: monotone(d["K"][str(K)]["awgn"]["exact"]) for K in self.Ks}
        self.bleu = {K: monotone(d["K"][str(K)]["awgn"]["bleu"]) for K in self.Ks}

    def eps_at(self, K, snr_db):
        return np.interp(snr_db, self.snr, self.eps[K])

    def exact_at(self, K, snr_db):
        return np.interp(snr_db, self.snr, self.exact[K])

    def uses(self, K):
        """Average channel uses (complex symbols) per sentence."""
        return K * self.L

    def sps(self, K):
        """Sentences per second carried by one RB."""
        return B_RB / (K * self.L)

    def pair_rates(self, snr_db, Ks=None):
        """Best admissible K for every pair.  Returns dict(S, mode, eps, exact, uses, sps); mode = K or 0 (no K admissible)."""
        Ks = list(Ks or self.Ks)
        snr_db = np.asarray(snr_db, float)
        S = np.zeros(snr_db.shape + (len(Ks),))
        for i, K in enumerate(Ks):
            e = self.eps_at(K, snr_db)
            S[..., i] = np.where(e >= EPS_TH, B_RB * I_SUT * e / (K * self.L), 0.0)       # S_{u,n}(K), 0 if eps < eps_th
        best = S.argmax(-1)
        Sb = np.take_along_axis(S, best[..., None], -1)[..., 0]
        Karr = np.array(Ks)[best]
        ok = Sb > 0
        mode = np.where(ok, Karr, 0)
        eps = np.zeros_like(Sb); exact = np.zeros_like(Sb)
        for K in Ks:
            m = mode == K
            if m.any():
                eps[m] = self.eps_at(K, snr_db[m]); exact[m] = self.exact_at(K, snr_db[m])
        uses = np.where(ok, mode * self.L, 0.0)
        sps = np.where(ok, B_RB / np.maximum(mode * self.L, 1e-9), 0.0)
        return dict(S=Sb, mode=mode, eps=eps, exact=exact, uses=uses, sps=sps)


# ------------------------------------------------------------------------------------ conventional
class ConvRates:
    """Semantic rate of the conventional chain (character or word Huffman + AMC)."""

    def __init__(self, source, path=None):
        d = json.load(open(path or os.path.join(RES_DIR, "task1_conv_sweeps.json")))["sweeps"][source]["awgn"]
        self.source = source
        self.names = [m["name"] for m in MCS_TABLE]
        self.eta = np.array([m["eta"] for m in MCS_TABLE])
        self.snr = np.array(d[self.names[0]]["snr"])
        self.bler = np.array([monotone(d[n]["bler"], increasing=False) for n in self.names])
        self.sim = np.array([monotone(d[n]["sim"]) for n in self.names])
        self.exact = np.array([monotone(d[n]["exact"]) for n in self.names])
        self.sym = np.array([d[n]["avg_symbols"] for n in self.names])         # channel uses per sentence
        self.thr = _thresholds(self.snr, self.bler)                            # AMC thresholds (BLER <= 10 %)

    def _interp(self, tab, i, snr_db):
        return np.interp(snr_db, self.snr, tab[i])

    def pair_rates(self, snr_db, rule="amc"):
        """rule = 'amc'      : MCS with the highest spectral efficiency whose threshold is below the SNR (Exercise 1)
                   'meaning' : MCS with the largest semantic rate among those with similarity >= eps_th."""
        snr_db = np.asarray(snr_db, float)
        M = len(self.names)
        S_all = np.zeros(snr_db.shape + (M,))
        for i in range(M):
            e = self._interp(self.sim, i, snr_db)
            S_all[..., i] = (B_RB / self.sym[i]) * e                           # sentences/s x average similarity
            adm = e >= EPS_TH
            if rule == "amc":
                adm &= snr_db >= self.thr[i]
            S_all[..., i] = np.where(adm, S_all[..., i], 0.0)
        if rule == "amc":     # highest spectral efficiency among admissible MCS
            score = np.where(S_all > 0, self.eta, -1.0)
            best = score.argmax(-1)
        else:                 # largest semantic rate
            best = S_all.argmax(-1)
        Sb = np.take_along_axis(S_all, best[..., None], -1)[..., 0]
        ok = Sb > 0
        mode = np.where(ok, best, -1)
        eps = np.zeros_like(Sb); exact = np.zeros_like(Sb)
        for i in range(M):
            m = mode == i
            if m.any():
                eps[m] = self._interp(self.sim, i, snr_db[m]); exact[m] = self._interp(self.exact, i, snr_db[m])
        uses = np.where(ok, self.sym[np.maximum(mode, 0)], 0.0)
        sps = np.where(ok, B_RB / np.maximum(uses, 1e-9), 0.0)
        return dict(S=Sb, mode=mode, eps=eps, exact=exact, uses=uses, sps=sps)


def get_systems():
    """All systems of the comparison: name -> function snr_db -> pair_rates dict."""
    sem = SemanticRates()
    sy = {}
    sy["Semantic, adaptive K"] = lambda g: sem.pair_rates(g)
    for K in sem.Ks:
        sy[f"Semantic, fixed K={K}"] = (lambda g, K=K: sem.pair_rates(g, Ks=[K]))
    for src, nm in (("char", "char Huffman"), ("word", "word Huffman")):
        cr = ConvRates(src)
        sy[f"Conventional, {nm} + AMC"] = (lambda g, cr=cr: cr.pair_rates(g, "amc"))
        sy[f"Conventional, {nm} + meaning-aware MCS"] = (lambda g, cr=cr: cr.pair_rates(g, "meaning"))
    return sy, sem
