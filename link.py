"""
link.py -- the complete conventional (bit-level) text link of Task 1, vectorised.

Transmitter : text -> char Huffman -> conv. encoder (133,171; K=7) -> puncturing
              -> Gray QPSK/16-QAM/64-QAM
Channel     : AWGN or Rayleigh block fading (+ AWGN)
Receiver    : zero-forcing equaliser -> max-log LLR demapper -> de-puncturing
              -> soft-decision Viterbi -> Huffman decoder -> text

Speed design.  The transmitter side does not depend on the SNR, so it is computed
ONCE per (sentence set, MCS) in `prepare()`; every SNR point then only runs
channel -> receiver (`run()`).  Frames of different lengths are processed together
by sorting them by length and padding (see channel_coding.viterbi_decode_batch).

Definitions
-----------
* information bits of a frame = Huffman bits of the sentence + <EOS> code.
* BLER (block = sentence error rate): a frame is in error if ANY information bit is
  wrong after Viterbi decoding (then the Huffman decoder desynchronises and the text
  differs).  Cross-checked in `run()` against text equality.
* coded BER: bit error rate of the information bits after Viterbi decoding.
"""
import numpy as np
from config import MCS_TABLE
from modem import QAM
from channel_coding import (conv_encode, puncture, puncture_mask, depuncture_llr,
                            viterbi_decode_batch, K)
import channel

_QAMS = {}


def get_qam(m):
    if m not in _QAMS:
        _QAMS[m] = QAM(m)
    return _QAMS[m]


class TxFrames:
    """Everything the transmitter produces for one (sentence set, MCS)."""

    def __init__(self, huff, sentences, mcs):
        self.mcs, self.sentences, self.huff = mcs, sentences, huff
        m, rate = mcs["m"], mcs["rate"]
        self.qam = get_qam(m)
        self.info = [huff.encode(s) for s in sentences]                  # list of uint8 arrays
        self.L = np.array([len(b) for b in self.info])                   # info bits / frame
        self.T = self.L + (K - 1)                                        # trellis steps
        coded = [puncture(conv_encode(b), rate) for b in self.info]      # transmitted coded bits
        self.n_tx_bits = np.array([len(c) for c in coded])
        # pad with zeros to a multiple of m, then modulate all frames in one go
        self.n_sym = (self.n_tx_bits + m - 1) // m
        padded = [np.concatenate([c, np.zeros(ns * m - len(c), np.uint8)]) for c, ns in zip(coded, self.n_sym)]
        self.symbols = self.qam.modulate(np.concatenate(padded))
        self.sym_off = np.r_[0, np.cumsum(self.n_sym)]
        self.frame_id = np.repeat(np.arange(len(sentences)), self.n_sym)


def run(tx, snr_db, rng, chan="awgn", block_len=None, want_text=False, chunk=600):
    """
    Send all frames of `tx` through the channel at `snr_db` and decode.
    Returns dict(err_frames bool[F], bit_errors int[F], text list|None).
    """
    mcs, qam = tx.mcs, tx.qam
    F = len(tx.sentences)
    # ------------------------------ channel + equaliser + demapper (all frames at once)
    if chan == "awgn":
        y, h = channel.awgn(tx.symbols, snr_db, rng)
    else:
        y, h = channel.rayleigh_block(tx.symbols, snr_db, rng, tx.frame_id, block_len)
    x_eq = channel.zf_equalize(y, h)                       # zero forcing
    n0 = 10.0 ** (-snr_db / 10.0)
    llr_all = qam.llr(x_eq, n0 / np.abs(h) ** 2)           # noise after ZF has variance N0/|h|^2
    llr_all = np.clip(llr_all, -200, 200)                  # keep Viterbi metrics well scaled
    m = mcs["m"]
    # ------------------------------ per-frame de-puncturing, padding, Viterbi
    order = np.argsort(tx.T)                               # decode similar lengths together
    bit_errors = np.zeros(F, np.int64)
    err = np.zeros(F, bool)
    texts = [None] * F if want_text else None
    for a in range(0, F, chunk):
        ids = order[a:a + chunk]
        Tmax = int(tx.T[ids].max())
        mat = np.full((len(ids), 2 * Tmax), 1e3, np.float32)       # +big = "coded 0" tail padding
        for r, f in enumerate(ids):
            seg = llr_all[tx.sym_off[f] * m: tx.sym_off[f] * m + tx.n_tx_bits[f]]   # drop modulation padding
            mat[r, : 2 * tx.T[f]] = depuncture_llr(seg, 2 * tx.T[f], mcs["rate"])
        dec = viterbi_decode_batch(mat)
        for r, f in enumerate(ids):
            L = tx.L[f]
            d = dec[r, :L]
            be = int(np.count_nonzero(d != tx.info[f]))
            bit_errors[f] = be
            err[f] = be > 0
            if want_text:
                texts[f] = tx.sentences[f] if be == 0 else tx.huff.decode(d)
    return dict(err_frames=err, bit_errors=bit_errors, text=texts)


def uncoded_ber(m, snr_es_n0_db, n_bits, rng, chan="awgn"):
    """Uncoded Gray-QAM BER by Monte Carlo (Task 1a).  Hard decision == sign of max-log LLR."""
    qam = get_qam(m)
    n_bits = (n_bits // m) * m
    bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    x = qam.modulate(bits)
    n0 = 10.0 ** (-snr_es_n0_db / 10.0)
    if chan == "awgn":
        y, h = channel.awgn(x, snr_es_n0_db, rng)
    else:                                                   # fast fading: independent h per symbol
        fid = np.arange(len(x))
        y, h = channel.rayleigh_block(x, snr_es_n0_db, rng, fid, None)
    xe = channel.zf_equalize(y, h)
    llr = qam.llr(xe, n0 / np.abs(h) ** 2)
    hard = (llr < 0).astype(np.uint8)                       # LLR>0 -> bit 0
    return float(np.mean(hard != bits))


if __name__ == "__main__":
    import time
    from corpus import load_corpus
    from source_coding import CharHuffman
    tr, te = load_corpus(); huff = CharHuffman(tr)
    rng = np.random.default_rng(1)
    for mc in MCS_TABLE[:2] + MCS_TABLE[-1:]:
        tx = TxFrames(huff, te[:300], mc)
        for snr in (0, 8, 14, 20, 26):
            t = time.time(); r = run(tx, snr, rng)
            print(mc["name"], snr, "BLER", r["err_frames"].mean().round(3),
                  "BER", (r["bit_errors"].sum() / tx.L.sum()).round(5), f"{time.time()-t:.1f}s")
