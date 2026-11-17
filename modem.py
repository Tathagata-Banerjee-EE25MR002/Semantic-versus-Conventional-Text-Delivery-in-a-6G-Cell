"""
modem.py -- Gray-mapped QPSK / 16-QAM / 64-QAM modulator and max-log LLR demapper,
plus exact closed-form (numerically exact) uncoded BER theory used in Task 1(a).

Constellation construction
--------------------------
Square M-QAM (M = L^2, L = 2, 4, 8) is the product of two L-PAM constellations
(levels -(L-1), ..., -1, +1, ..., (L-1)).  Each PAM uses a *Gray labelling*:
adjacent levels differ in exactly one bit.  The first m/2 bits of the m-bit label
select the in-phase (I) level, the last m/2 bits the quadrature (Q) level.
QPSK is simply 2-PAM x 2-PAM (m = 2), so the three schemes share one code path.
Points are scaled to unit average symbol energy  Es = 1  so that the channel SNR is
Es/N0 = 1/N0.

Max-log LLR demapper
--------------------
For bit k of a received (equalised) symbol y,
    LLR_k = ( min_{s: b_k(s)=1} |y-s|^2  -  min_{s: b_k(s)=0} |y-s|^2 ) / N0_eff
(the max-log approximation replaces log-sum-exp by the largest term).  After
zero-forcing, the effective noise variance of y/h is N0/|h|^2, so the LLR is scaled
by |h|^2: reliable symbols (strong channel) get large |LLR|, faded ones small |LLR|.
This is essential for soft Viterbi decoding in fading.
"""
import numpy as np
from scipy.special import erfc


def Q(x):
    """Gaussian tail probability Q(x) = P(N(0,1) > x)."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def _gray_inverse(g):
    """Inverse of  g = i ^ (i >> 1)."""
    i = 0
    while g:
        i ^= g
        g >>= 1
    return i


class QAM:
    """Gray-mapped square QAM with m bits/symbol (m = 2, 4, 6)."""

    def __init__(self, m):
        assert m in (2, 4, 6)
        self.m, self.M, self.L = m, 1 << m, 1 << (m // 2)
        L = self.L
        self.scale = 1.0 / np.sqrt(2.0 * (L ** 2 - 1) / 3.0)      # unit average energy
        half = m // 2
        pts = np.zeros(self.M, complex)
        labels = np.zeros((self.M, m), np.uint8)
        for lab in range(self.M):
            bits = [(lab >> (m - 1 - k)) & 1 for k in range(m)]
            labels[lab] = bits
            iI = _gray_inverse(int("".join(map(str, bits[:half])), 2))
            iQ = _gray_inverse(int("".join(map(str, bits[half:])), 2))
            pts[lab] = complex(2 * iI - (L - 1), 2 * iQ - (L - 1)) * self.scale
        self.points, self.labels = pts, labels
        self._pow = 1 << np.arange(m - 1, -1, -1)
        self.idx0 = [np.where(labels[:, k] == 0)[0] for k in range(m)]
        self.idx1 = [np.where(labels[:, k] == 1)[0] for k in range(m)]

    # ---------------------------------------------------------------- Tx
    def modulate(self, bits):
        """bits (multiple of m) -> complex symbols."""
        idx = bits.reshape(-1, self.m).astype(np.int64) @ self._pow
        return self.points[idx]

    # ---------------------------------------------------------------- Rx
    def llr(self, y, n0_eff, chunk=40000):
        """Max-log LLRs, shape (Nsym*m,), bit order = transmit order.
        n0_eff : scalar or per-symbol array of effective noise variance (complex)."""
        n0_eff = np.broadcast_to(np.asarray(n0_eff, float), y.shape)
        out = np.empty((len(y), self.m), np.float32)
        for a in range(0, len(y), chunk):
            d2 = np.abs(y[a:a + chunk, None] - self.points[None, :]) ** 2       # (n, M)
            for k in range(self.m):
                out[a:a + chunk, k] = (d2[:, self.idx1[k]].min(1) - d2[:, self.idx0[k]].min(1)) \
                                      / n0_eff[a:a + chunk]
        return out.reshape(-1)


# ---------------------------------------------------------------------------
# Theory: exact uncoded BER of Gray-mapped square QAM
# ---------------------------------------------------------------------------
def ber_qam_awgn(es_n0_lin, m):
    """
    Exact uncoded BER of Gray square M-QAM in AWGN (averaged over the m bits).

    Because I and Q are independent, BER = BER of one L-PAM with Gray labelling.
    For transmit level index i (levels spaced 2*scale) the sample falls in decision
    interval j with a probability given by Gaussian CDF differences (outer intervals
    are open-ended).  The bit-k error probability is the average over i of
    P(i -> j) * [label_k(i) != label_k(j)].  No high-SNR approximation is used.
    """
    es_n0_lin = np.atleast_1d(np.asarray(es_n0_lin, float))
    L, half = 1 << (m // 2), m // 2
    scale = 1.0 / np.sqrt(2.0 * (L ** 2 - 1) / 3.0)
    gray = [i ^ (i >> 1) for i in range(L)]
    lab = np.array([[(g >> (half - 1 - k)) & 1 for k in range(half)] for g in gray])   # (L, half)
    res = []
    for s in es_n0_lin:
        sigma = np.sqrt(1.0 / s / 2.0)             # per-dimension noise std (Es=1 -> N0=1/s)
        d = 2.0 * scale                             # spacing between adjacent levels
        # decision thresholds between level j and j+1 sit at the midpoints
        # P(i->j) = Phi((t_{j}-x_i)/sigma) - Phi((t_{j-1}-x_i)/sigma)
        x = (2 * np.arange(L) - (L - 1)) * scale
        thr = np.concatenate([[-np.inf], (x[:-1] + x[1:]) / 2, [np.inf]])
        P = np.zeros((L, L))
        for i in range(L):
            cdf = 1 - Q((thr - x[i]) / sigma)       # Phi((t - x_i)/sigma)
            P[i] = np.diff(cdf)
        err = 0.0
        for i in range(L):
            for j in range(L):
                err += P[i, j] * np.sum(lab[i] != lab[j])
        res.append(err / (L * half))
    return np.array(res)


def ber_qam_rayleigh(avg_es_n0_lin, m, n_pts=3000):
    """
    Uncoded BER over flat Rayleigh fading with coherent detection (== ZF with known h).
    The instantaneous SNR is gamma = avg * x with x ~ Exp(1), so
        BER = int_0^inf e^{-x} BER_awgn(avg * x) dx .
    At high average SNR the integrand lives at x ~ 1/avg (deep fades dominate the
    error probability -> BER ~ 1/SNR), where Gauss-Laguerre nodes are far too sparse.
    We therefore integrate in the log domain, x = e^u, u in [-25, 5]  (dx = x du),
    with the trapezoid rule on a dense grid, which resolves every SNR we need.
    """
    u = np.linspace(-25.0, 5.0, n_pts)
    x = np.exp(u)
    out = []
    for g in np.atleast_1d(avg_es_n0_lin):
        f = np.exp(-x) * ber_qam_awgn(g * x, m) * x
        out.append(np.trapz(f, u) if hasattr(np, "trapz") else np.trapezoid(f, u))
    return np.array(out)


if __name__ == "__main__":
    for m in (2, 4, 6):
        q = QAM(m)
        print(m, "avg energy", np.mean(np.abs(q.points) ** 2).round(6),
              "min dist", np.min(np.abs(q.points[:, None] - q.points[None, :]) + np.eye(q.M) * 9).round(4))
    print("BER 16QAM Es/N0=15dB:", ber_qam_awgn(10 ** 1.5, 4))
