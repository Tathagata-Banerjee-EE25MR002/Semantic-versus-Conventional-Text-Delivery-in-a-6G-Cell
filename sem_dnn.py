"""
sem_dnn.py -- Task 3: a deep neural network that IMITATES the optimal MILP allocation (supervised learning).

Exercise 1 trained its allocator without labels (unsupervised L2O) because the bit-rate MILP was slow.  Here the task
asks for imitation, so the network is trained with labels: for every random drop we solve the MILP once (offline)
and teach the network to reproduce the optimal user of every RB.

Input   per (user, RB) cell six features computed from the admissible semantic rate S~_{u,n} (the K decision is
        already contained in S~, see sem_ra.py):
          c0  S~ / 5 ksut/s                              absolute attractiveness of the pair
          c1  S~ / (best S~ on this RB over all users)   1 = this user is the best one on this RB
          c2  S~ / (mean S~ of this user over its RBs)   good RB for this user relative to its average
          c3  1[S~ > 0]                                  pair admissible (eps >= eps_th)
          c4  (sum_n S~_{u,n}) / Smin / 10               how well the user can be served at all (broadcast over the RBs)
          c5  (Pmax_dBm - 27.5)/12.5                     power budget (same for all cells)
Network the permutation-EQUIVARIANT network of Exercise 1 (EqLayer: every cell sees its own state, the mean over its
        user's RBs, the mean and max over the users on its RB and a global mean; 4 layers, 48 units, shared weights).
        Equivariance means: relabel users or RBs and the output is relabelled identically -- the network does not have
        to learn the same rule for every position, so 20 000 labelled drops are plenty.
Output  one logit per cell; for every RB a soft-max over the U users.
Loss    cross-entropy against the MILP's user of that RB (RBs the MILP leaves unused are ignored).
Test    hard arg-max per RB (exactly one user per RB) -- no solver, one forward pass.
"""
import os, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
import resource_alloc as ra
import sem_ra

torch.set_num_threads(int(os.environ.get("EX2_THREADS", "2")))
P_LO, P_HI = float(PMAX_DBM_SWEEP[0]), float(PMAX_DBM_SWEEP[-1])


def features(S, p_dbm):
    """S: (B,U,N) admissible rate [suts/s]; p_dbm: (B,)  ->  (B,U,N,6)."""
    s = S / 1e3
    best = s.max(1, keepdim=True).values
    c0 = s / 5.0
    c1 = s / (best + 1e-6)
    c2 = s / (s.mean(2, keepdim=True) + 1e-6)
    c3 = (s > 0).float()
    c4 = (s.sum(2, keepdim=True) * 1e3 / S_MIN / 10.0).expand_as(s)
    c5 = ((p_dbm - 27.5) / 12.5)[:, None, None].expand_as(s)
    return torch.stack([c0, c1, c2, c3, c4, c5], dim=-1)


class EqLayer(nn.Module):
    """h'_{un} = ReLU(W0 h_{un} + W1 mean_n h + W2 mean_u h + W3 max_u h + W4 mean_{un} h + b)   (same layer as Exercise 1)."""
    def __init__(self, din, dout):
        super().__init__()
        self.w0 = nn.Linear(din, dout)
        self.w1, self.w2, self.w3, self.w4 = (nn.Linear(din, dout, bias=False) for _ in range(4))

    def forward(self, h):                                                # (B,U,N,d)
        return torch.relu(self.w0(h) + self.w1(h.mean(2, keepdim=True)) + self.w2(h.mean(1, keepdim=True))
                          + self.w3(h.max(1, keepdim=True).values) + self.w4(h.mean((1, 2), keepdim=True)))


class ImitationNet(nn.Module):
    def __init__(self, hid=48, layers=4):
        super().__init__()
        dims = [6] + [hid] * layers
        self.layers = nn.ModuleList([EqLayer(dims[i], dims[i + 1]) for i in range(layers)])
        self.head = nn.Linear(hid, 1)

    def forward(self, S, p_dbm):
        h = features(S, p_dbm)
        for l in self.layers:
            h = l(h)
        return self.head(h).squeeze(-1)                                  # (B,U,N) logits


# ------------------------------------------------------------------------------------ labelled data
def make_dataset(n, sem, seed, path=None):
    """n random drops with random Pmax in [15,40] dBm; label = optimal MILP assignment of the adaptive-K rate matrix."""
    if path and os.path.exists(path):
        d = np.load(path); return d["S"], d["p"], d["lab"]
    rng = np.random.default_rng(seed)
    a = ra.generate_drops(rng, n)
    p_dbm = rng.uniform(P_LO, P_HI, n)
    gam = 10 * np.log10(np.maximum(a * (ra.dbm_to_w(p_dbm) / N)[:, None, None], 1e-30))
    S = sem.pair_rates(gam)["S"].astype(np.float32)
    t0 = time.time()
    lab = np.full((n, N), -100, np.int64)
    for d in range(n):
        X = sem_ra.milp_assign_one(S[d].astype(float))
        used = X.sum(0) > 0
        lab[d, used] = X.argmax(0)[used]
        if d % 2000 == 0:
            print(f"[3] MILP labels {d}/{n} ({time.time()-t0:.0f}s)", flush=True)
    if path:
        np.savez_compressed(path, S=S, p=p_dbm, lab=lab)
    return S, p_dbm, lab


def train_imitation(S, p, lab, steps=RA2_IMIT_STEPS, batch=128, seed=SEED, path=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    S_t, p_t, lab_t = torch.tensor(S), torch.tensor(p, dtype=torch.float32), torch.tensor(lab)
    net = ImitationNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)
    hist, t0 = [], time.time()
    for it in range(steps):
        idx = torch.tensor(rng.integers(0, len(S), batch))
        z = net(S_t[idx], p_t[idx])                                      # (B,U,N)
        loss = F.cross_entropy(z.permute(0, 2, 1).reshape(-1, U), lab_t[idx].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        if it % 200 == 0 or it == steps - 1:
            with torch.no_grad():
                m = lab_t[idx] >= 0
                acc = float((z.argmax(1)[m] == lab_t[idx][m]).float().mean())
            hist.append((it, float(loss.detach()), acc))
            if it % 1000 == 0 or it == steps - 1:
                print(f"[3] imitation step {it:5d} loss {float(loss):.3f} RB-accuracy {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    if path:
        torch.save(net.state_dict(), path)
    return net, hist


@torch.no_grad()
def dnn_assign(net, S3, p_dbm_value):
    """Inference: S3 (D,U,N) numpy -> hard assignment X (D,U,N): arg-max user of every RB (a pair with S~=0 is dropped)."""
    S = torch.tensor(S3, dtype=torch.float32)
    z = net(S, torch.full((S.shape[0],), float(p_dbm_value)))
    X = torch.zeros_like(z).scatter_(1, z.argmax(1, keepdim=True), 1.0).numpy().astype(np.float64)
    return X * (S3 > 0)


def repair(X, S3, smin=S_MIN):
    """Deficit repair (cheap post-processing of the raw arg-max output, O(U*N) per deficient user).

    The imitation network decides every RB separately, so nothing guarantees that every user reaches Smin.
    Repair: while a user u has S_u < Smin, move to u the RB n (owned by a user v that keeps S_v >= Smin
    after losing it, or an unused RB) with the largest S~[u,n] > 0; stop when no such move exists.
    S3: (D,U,N) rate matrix, X: (D,U,N) 0/1 assignment (modified copy returned)."""
    X = X.copy()
    for d in range(X.shape[0]):
        S = S3[d]
        for _ in range(S.shape[1]):                                         # at most N moves per drop
            Ru = (X[d] * S).sum(1)
            deficit = np.where(Ru < smin * (1 - 1e-9))[0]
            if len(deficit) == 0:
                break
            u = deficit[np.argmax(smin - Ru[deficit])]                     # most deficient user first
            owner = X[d].argmax(0) * (X[d].sum(0) > 0.5) - (X[d].sum(0) < 0.5)   # -1 = unused RB
            best, best_n = 0.0, -1
            for n in np.where((S[u] > 0) & (owner != u))[0]:
                v = owner[n]
                if v >= 0 and Ru[v] - S[v, n] < smin * (1 - 1e-9) and Ru[v] >= smin * (1 - 1e-9):
                    continue                                                # would push a satisfied user below Smin
                gain = S[u, n]
                if gain > best:
                    best, best_n = gain, n
            if best_n < 0:
                break
            X[d][:, best_n] = 0.0; X[d][u, best_n] = 1.0
    return X


def load(path):
    net = ImitationNet(); net.load_state_dict(torch.load(path)); net.eval(); return net
