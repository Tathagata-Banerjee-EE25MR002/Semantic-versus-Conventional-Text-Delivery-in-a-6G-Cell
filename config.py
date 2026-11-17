"""
config.py -- every tunable number of Exercise 1 lives here (single source of truth).

Why a config module?  Exercise 2 (semantic communication) must run on *the same
cell* and be compared with this system, so all system parameters (Table 1 of the
problem statement) are defined once and imported by every module.

Set the environment variable  EX1_QUICK=1  to run a drastically reduced version of
every experiment (a few minutes) -- useful as a smoke test.  The default (full)
setting reproduces the figures/numbers reported in the paper.
"""
import os
import numpy as np

QUICK = os.environ.get("EX1_QUICK", "0") == "1"

# --------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------
ROOT = os.environ.get("EX2_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # EX2_ROOT: write somewhere else (e.g. quick runs)
DATA_DIR = os.path.join(ROOT, "data")          # corpus + trained models + LUTs
FIG_DIR = os.path.join(ROOT, "figures")        # every PNG/PDF figure
RES_DIR = os.path.join(ROOT, "results")        # JSON/CSV numbers quoted in the paper
for _d in (DATA_DIR, FIG_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

SEED = 2026                                    # global reproducibility seed

# --------------------------------------------------------------------------
# Table 1 of the problem statement -- simulation parameters
# --------------------------------------------------------------------------
U = 8                       # number of single-antenna users
N = 16                      # number of OFDMA resource blocks (RBs)
B_RB = 180e3                # RB bandwidth [Hz]
CELL_RADIUS_M = 500.0       # cell radius [m]
MIN_DIST_M = 35.0           # minimum BS-user distance [m]
PMAX_DBM_DEFAULT = 30.0     # BS transmit power budget [dBm]
PMAX_DBM_SWEEP = np.arange(15.0, 40.1, 2.5)   # 15 ... 40 dBm
PL_A_DB = 128.1             # path loss  128.1 + 37.6 log10(d/1km)  [dB]
PL_B_DB = 37.6
SHADOW_STD_DB = 8.0         # log-normal shadowing standard deviation [dB]
NOISE_PSD_DBM_HZ = -174.0   # thermal noise PSD [dBm/Hz]
NOISE_FIGURE_DB = 9.0       # receiver noise figure F [dB]
R_MIN = 1e6                 # minimum user rate [bit/s]

N_TRAIN_SENT = 20000        # >= 20 000 training sentences (Table 1)
N_TEST_SENT = 2000          # >= 2 000 test sentences      (Table 1)

# --------------------------------------------------------------------------
# Task 1 -- link-level simulation
# --------------------------------------------------------------------------
# (name, bits per symbol m, code rate)
MCS_TABLE = [
    dict(name="QPSK 1/2",   mod="QPSK",   m=2, rate="1/2", r=1 / 2),
    dict(name="QPSK 3/4",   mod="QPSK",   m=2, rate="3/4", r=3 / 4),
    dict(name="16-QAM 1/2", mod="16QAM",  m=4, rate="1/2", r=1 / 2),
    dict(name="16-QAM 3/4", mod="16QAM",  m=4, rate="3/4", r=3 / 4),
    dict(name="64-QAM 2/3", mod="64QAM",  m=6, rate="2/3", r=2 / 3),
    dict(name="64-QAM 3/4", mod="64QAM",  m=6, rate="3/4", r=3 / 4),
]
for _mc in MCS_TABLE:                       # spectral efficiency [information bit / symbol]
    _mc["eta"] = _mc["m"] * _mc["r"]

BLER_TARGET = 0.10          # AMC rule: highest-rate MCS with BLER <= 10 %

if QUICK:
    LINK_NFRAMES = 300                       # sentences per (MCS, SNR) point
    SNR_GRID_AWGN = np.arange(0.0, 30.1, 2.0)
    SNR_GRID_RAYL = np.arange(0.0, 36.1, 3.0)
    BER_VALID_BITS = 200_000
else:
    LINK_NFRAMES = N_TEST_SENT              # all 2000 test sentences per point
    SNR_GRID_AWGN = np.arange(0.0, 30.01, 0.5)   # 0.5 dB grid -> fine LUT
    SNR_GRID_RAYL = np.arange(0.0, 36.01, 1.0)
    BER_VALID_BITS = 2_000_000

# --------------------------------------------------------------------------
# Task 2 -- autoencoder
# --------------------------------------------------------------------------
AE_EBN0_GRID = np.arange(-2.0, 10.01, 1.0)   # dB
AE_TRAIN_ITERS = 400 if QUICK else 12000
AE_BATCH = 2048
AE_EVAL_MSGS = 200_000 if QUICK else 2_000_000

# --------------------------------------------------------------------------
# Task 3 -- resource allocation
# --------------------------------------------------------------------------
RA_TEST_DROPS = 40 if QUICK else 300       # random channel realisations per Pmax
RA_TRAIN_ITERS = 500 if QUICK else 12000
RA_BATCH = 256

# ==========================================================================
# EXERCISE 2 -- semantic communication (additions; everything above is Exercise 1, unchanged)
# ==========================================================================
K_LIST = [1, 2, 4, 8]                  # complex channel symbols per word (Table 1 of Exercise 2)
EPS_TH = 0.9                           # similarity threshold eps_th
S_MIN = 2000.0                         # minimum semantic rate per user [suts/s]
I_SUT = 1.0                            # semantic information per sentence [sut]
SEM_TRAIN_SNR = (-6.0, 18.0)           # AWGN training SNR range [dB]
MIN_WORD_COUNT = 3                     # a word enters the model vocabulary if it occurs >= 3 times in the train set
D_MODEL, N_HEADS, N_LAYERS, D_FF = 128, 4, 2, 512   # Transformer: width 128, 4 heads, 2 layers (Table 1 allows 2-3)
MAX_TOK = 64                           # longest sentence (tokens + <eos>) the model handles

if QUICK:
    SEM_TRAIN_STEPS = 150
    SEM_STEPS_BY_K = {}
    SEM_BATCH = 64
    SEM_EVAL_SENT = 300                # test sentences per (K, SNR) point
    SEM_SNR_GRID = np.arange(-10.0, 30.1, 4.0)
    RA2_DROPS = 30
    RA2_IMIT_TRAIN = 2000              # MILP-labelled drops for the imitation DNN
    RA2_IMIT_STEPS = 300
else:
    SEM_TRAIN_STEPS = 3000
    SEM_STEPS_BY_K = {2: 7000}          # K=2 converges more slowly (less redundancy) -> longer schedule, same batch size
    SEM_BATCH = 96
    SEM_EVAL_SENT = N_TEST_SENT        # all 2000 test sentences per point
    SEM_SNR_GRID = np.arange(-10.0, 30.1, 1.0)
    RA2_DROPS = 300                    # the SAME 300 drops as Exercise 1 (rng seed SEED+77)
    RA2_IMIT_TRAIN = 20000
    RA2_IMIT_STEPS = 6000

# conventional-link sweeps of Exercise 2 (character AND word Huffman; the grid starts at -6 dB so that the
# curves can be compared with the semantic system, which works down to -10 dB)
if QUICK:
    CONV_SNR_AWGN = np.arange(-6.0, 30.1, 3.0)
    CONV_SNR_RAYL = np.arange(-6.0, 36.1, 3.0)
else:
    CONV_SNR_AWGN = np.arange(-6.0, 30.01, 0.5)
    CONV_SNR_RAYL = np.arange(-6.0, 36.01, 1.0)
PLOT_MCS = ["QPSK 1/2", "16-QAM 1/2", "64-QAM 3/4"]     # the three MCS shown in the Task-1 / Task-2 plots
