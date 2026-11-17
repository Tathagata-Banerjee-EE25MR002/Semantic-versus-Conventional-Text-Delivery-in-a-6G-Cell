"""
channel_coding.py -- rate-1/2, K=7 convolutional code (generators 133, 171 octal),
puncturing to rates 2/3 and 3/4, and a *soft-decision* Viterbi decoder.

This is the NASA / IEEE-802.11 code.  Conventions used consistently in encoder and
decoder (the whole trellis is derived from them):

* The shift register is  reg = (u << 6) | s,   s = 6 previous input bits with the
  MOST RECENT one in bit 5.  Next state = reg >> 1 = (u << 5) | (s >> 1).
* Generator g = 0o133 = 1011011b, g = 0o171 = 1111001b.  Bit 6 (MSB) of the
  generator multiplies the *current* input.  Output bit j = parity(reg & g_j).
* Output order for every input bit: (A = g0 = 133, B = g1 = 171).
* The frame is *terminated*: K-1 = 6 zero tail bits drive the encoder back to the
  all-zero state, so the decoder can start and end in state 0.

Puncturing (IEEE 802.11 patterns, applied to the stream A1 B1 A2 B2 A3 B3 ...):
    1/2 : [1 1]            (nothing removed)
    2/3 : [1 1 1 0]        (B2 removed,            2 input bits -> 3 coded bits)
    3/4 : [1 1 1 0 0 1]    (B2 and A3 removed,     3 input bits -> 4 coded bits)

LLR convention (used everywhere in this project):
    LLR = ln P(bit=0 | y) / P(bit=1 | y)   ->   LLR > 0 means "bit is 0".
Punctured positions are re-inserted with LLR = 0 (no information) before decoding.
"""
import numpy as np

K = 7
GENS = (0o133, 0o171)
NSTATES = 1 << (K - 1)                  # 64
PUNCT = {"1/2": [1, 1], "2/3": [1, 1, 1, 0], "3/4": [1, 1, 1, 0, 0, 1]}

# taps[j][i] = 1 if generator j taps the input delayed by i samples
_TAPS = [np.array([(g >> (K - 1 - i)) & 1 for i in range(K)], dtype=np.int64) for g in GENS]


def conv_encode(bits):
    """Terminated rate-1/2 encoding.  bits: uint8 (L,) -> uint8 (2*(L+K-1),)."""
    x = np.concatenate([bits.astype(np.int64), np.zeros(K - 1, np.int64)])
    out = np.empty(2 * len(x), np.uint8)
    for j, taps in enumerate(_TAPS):
        out[j::2] = (np.convolve(x, taps)[: len(x)] & 1).astype(np.uint8)   # GF(2) convolution
    return out


def puncture_mask(n_coded, rate):
    """Boolean keep-mask over the rate-1/2 stream of length n_coded (= 2*(L+6))."""
    return np.resize(np.array(PUNCT[rate], dtype=bool), n_coded)


def puncture(coded, rate):
    return coded[puncture_mask(len(coded), rate)]


def depuncture_llr(llr_rx, n_coded, rate):
    """Insert zero-LLRs at punctured positions -> length n_coded array."""
    mask = puncture_mask(n_coded, rate)
    out = np.zeros(n_coded, np.float32)
    out[mask] = llr_rx
    return out


# ---------------------------------------------------------------------------
# Trellis tables for the Viterbi decoder
# ---------------------------------------------------------------------------
def _parity(v):
    return bin(v).count("1") & 1


def _build_trellis():
    """
    For every next-state `ns` there are exactly two predecessor states
        s_b = ((ns & 31) << 1) | b ,  b in {0,1},   and the input bit is u = ns >> 5.
    outidx[ns,b] encodes the two coded bits (A,B) of that branch as 2*A + B.
    """
    pred = np.zeros((NSTATES, 2), np.int64)
    outidx = np.zeros((NSTATES, 2), np.int64)
    for ns in range(NSTATES):
        u = ns >> 5
        for b in (0, 1):
            s = ((ns & 31) << 1) | b
            reg = (u << 6) | s
            A, Bc = _parity(reg & GENS[0]), _parity(reg & GENS[1])
            pred[ns, b] = s
            outidx[ns, b] = 2 * A + Bc
    return pred, outidx


_PRED, _OUTIDX = _build_trellis()


def viterbi_decode_batch(llr, tail_fill=1e3):
    """
    Soft-decision Viterbi decoding of a batch of terminated frames.

    llr : float32 (B, 2*T) de-punctured LLRs; frames shorter than T trellis steps
          must already be padded with +tail_fill (= "coded bit 0, certain"), which
          is exactly what the zero tail of a terminated code looks like.  This lets
          frames of different length share one vectorised pass.
    Returns uint8 (B, T) decoded input bits (information bits followed by the
    6 tail bits and padding).

    Metric: we MAXIMISE the correlation  sum_j (1-2c_j) * LLR_j / 2  along a path
    (equivalent to minimising Euclidean distance for BPSK-like soft values).
    The final state is forced to 0 (terminated trellis).
    """
    B, T2 = llr.shape
    T = T2 // 2
    pm = np.full((B, NSTATES), -1e9, np.float32)
    pm[:, 0] = 0.0                                   # encoder starts in state 0
    dec = np.empty((T, B, NSTATES), np.uint8)        # survivor decisions (which predecessor)
    pred, outidx = _PRED, _OUTIDX
    for t in range(T):
        L0, L1 = llr[:, 2 * t], llr[:, 2 * t + 1]
        # branch metric for the four possible coded pairs (A,B): +/- L0 +/- L1, halved
        bm4 = 0.5 * np.stack([L0 + L1, L0 - L1, -L0 + L1, -L0 - L1], axis=1)   # (B,4)
        cand = pm[:, pred] + bm4[:, outidx]                                      # (B,64,2)
        d = cand[:, :, 1] > cand[:, :, 0]                                        # pick best predecessor
        dec[t] = d
        pm = np.where(d, cand[:, :, 1], cand[:, :, 0])
        pm -= pm.max(axis=1, keepdims=True)           # normalise to avoid drift
    # trace back from state 0
    bits = np.empty((B, T), np.uint8)
    state = np.zeros(B, np.int64)
    ar = np.arange(B)
    for t in range(T - 1, -1, -1):
        bits[:, t] = state >> 5                       # input bit that led into `state`
        b = dec[t, ar, state].astype(np.int64)
        state = ((state & 31) << 1) | b
    return bits


if __name__ == "__main__":
    # self-test: noiseless round trip for all three rates
    rng = np.random.default_rng(0)
    for rate in PUNCT:
        info = rng.integers(0, 2, 200).astype(np.uint8)
        c = conv_encode(info); tx = puncture(c, rate)
        llr = (1 - 2.0 * tx).astype(np.float32) * 5
        full = depuncture_llr(llr, len(c), rate)[None, :]
        dec = viterbi_decode_batch(full)[0][:200]
        print(rate, "tx bits", len(tx), "ok" if np.array_equal(dec, info) else "FAIL")
