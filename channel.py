"""
channel.py -- AWGN and Rayleigh block-fading channels, and the zero-forcing equaliser.

SNR definition.  Symbols have unit average energy (Es = 1), so for a target
SNR = Es/N0 [dB] the complex noise has variance N0 = 10^(-SNR/10) (N0/2 per real
dimension).  This "SNR" is the same quantity as gamma_{u,n} = p_n g_{u,n}/(N0 B F)
of Task 3, which makes the look-up tables of Task 1 directly usable in Task 3.
For the fading channel SNR is the *average* SNR; the instantaneous SNR is
|h|^2 * SNR with |h|^2 ~ Exp(1).

Rayleigh BLOCK fading: the complex gain h ~ CN(0,1) is constant within a block of
`block_len` symbols and independent across blocks (block_len=None -> one gain per
frame, i.e. a quasi-static channel like one RB of Task 3).  The receiver knows h
(perfect CSI), as is standard.

Zero forcing: with y = h x + n the ZF equaliser is simply  x_hat = y / h .  It
removes the channel completely but colours the noise: var(n/h) = N0/|h|^2, which
the demapper must know (see modem.QAM.llr).
"""
import numpy as np


def awgn(x, snr_db, rng, frame_id=None, block_len=None):
    """Returns (y, h) with h = 1 (no fading)."""
    n0 = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise, np.ones_like(x)


def rayleigh_block(x, snr_db, rng, frame_id, block_len=None):
    """
    Rayleigh block fading + AWGN.
    frame_id[i] = index of the frame that symbol i belongs to (symbols of all frames
    are concatenated in one array for speed).  Returns (y, h_per_symbol).
    """
    n0 = 10.0 ** (-snr_db / 10.0)
    if block_len is None:
        block_id = frame_id
    else:
        # position of each symbol inside its frame
        starts = np.r_[0, np.flatnonzero(np.diff(frame_id)) + 1]
        pos = np.arange(len(x)) - np.repeat(starts, np.diff(np.r_[starts, len(x)]))
        block_id = frame_id * 100000 + pos // block_len
    uniq, inv = np.unique(block_id, return_inverse=True)
    hb = (rng.standard_normal(len(uniq)) + 1j * rng.standard_normal(len(uniq))) / np.sqrt(2.0)
    h = hb[inv]
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return h * x + noise, h


def zf_equalize(y, h):
    """Zero-forcing equaliser: divide by the (known) channel gain."""
    return y / h
