"""
semantic_model.py -- DeepSC-style semantic transceiver (Task 2).

Block diagram (one model per K, trained end to end)
---------------------------------------------------
 words --embedding--> Transformer semantic ENCODER --> dense CHANNEL ENCODER --> K complex symbols / word
        --> power normalisation --> channel (AWGN, or Rayleigh + zero forcing) --> dense CHANNEL DECODER
        --> Transformer semantic DECODER --> soft-max over the vocabulary (one word per position)

* Semantic encoder : token embedding (width d = 128) + sinusoidal positions + 2 pre-LN Transformer
                     encoder layers, 4 heads.  It turns the words into context-aware "meaning" vectors.
* Channel encoder  : dense 128 -> 256 -> 2K, i.e. K complex symbols (I and Q) for EVERY word.
                     K is the number of channel uses per word, so a sentence of L words uses K*L symbols.
* Power normalisation: every sentence is scaled to unit AVERAGE symbol energy (Es = 1), the same
                     convention as the conventional system, so the SNR is Es/N0 = gamma of Exercise 1.
* Channel          : AWGN  y = x + n,  or Rayleigh  y = h x + n  (one complex gain h ~ CN(0,1) per
                     sentence = block fading, as in Exercise 1) followed by zero forcing  x_hat = y / h.
                     The decoder never sees h, only x_hat (so the noise is coloured by 1/|h|).
* Channel decoder  : dense 2K -> 256 -> 128.
* Semantic decoder : 2 Transformer layers (self-attention + feed-forward), then logits =
                     LayerNorm(h) E^T + b with the embedding matrix E of the encoder (weight tying).
* Loss             : cross-entropy of every transmitted position (words and <eos>), padding ignored.
                     A NEW random SNR ~ U[-6, 18] dB is drawn for every sentence of every batch.

Differences to DeepSC (Xie et al., 2021), all on purpose
--------------------------------------------------------
1. The semantic decoder is NON-autoregressive: position i of the output is decoded from the K symbols
   received for word i plus bidirectional self-attention over the whole sentence.  It needs one forward
   pass (not one per word), has no exposure bias and is ~10x cheaper to evaluate on a CPU.  The
   decoder is still a Transformer (self-attention + feed-forward layers) and still uses sentence context.
2. No mutual-information (MINE) regulariser; cross-entropy only, exactly as in the task text.
3. Weight tying between input embedding and output projection: rare words then get gradient from both
   ends, which matters with only ~320 k training words and a few thousand CPU training steps.
The receiver knows the burst length (number of symbols = K * (words + 1)), exactly like the conventional
receiver knows the frame length in Exercise 1; <eos> is transmitted anyway (costs K symbols).
"""
import os, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from tokens import PAD, EOS, UNK

torch.set_num_threads(int(os.environ.get("EX2_THREADS", "2")))


def sinusoid(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class SemanticCodec(nn.Module):
    def __init__(self, vocab_size, K, d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, ff=D_FF, drop=0.1):
        super().__init__()
        self.K, self.d, self.V = K, d, vocab_size
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, 0.0, d ** -0.5)
        self.register_buffer("pe", sinusoid(MAX_TOK + 1, d))
        mk = lambda: nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.sem_enc = nn.TransformerEncoder(mk(), layers, enable_nested_tensor=False)
        self.ch_enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 2 * K))
        self.ch_dec = nn.Sequential(nn.Linear(2 * K, 256), nn.ReLU(), nn.Linear(256, d))
        self.sem_dec = nn.TransformerEncoder(mk(), layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    # ------------------------------------------------------------------ transmitter
    def transmit(self, ids):
        """ids (B,T) -> power-normalised symbols (B,T,K,2) [last axis = I,Q] and the validity mask (B,T)."""
        mask = ids != PAD
        h = self.emb(ids) * math.sqrt(self.d) + self.pe[: ids.shape[1]]
        h = self.sem_enc(h, src_key_padding_mask=~mask)                  # semantic encoder
        x = self.ch_enc(h).view(ids.shape[0], ids.shape[1], self.K, 2)    # channel encoder -> K complex symbols
        x = x * mask[:, :, None, None]                                    # padded positions are NOT transmitted
        n_sym = mask.sum(1).clamp(min=1) * self.K                         # symbols actually sent per sentence
        energy = (x ** 2).sum((1, 2, 3)) / n_sym                          # mean |x|^2 per sentence
        x = x / torch.sqrt(energy + 1e-9)[:, None, None, None]            # power normalisation: Es = 1
        return x, mask

    # ------------------------------------------------------------------ channel
    @staticmethod
    def channel(x, snr_db, kind="awgn", fade_per_word=False):
        """x (B,T,K,2); snr_db (B,) or scalar = Es/N0.  Returns the (equalised) received symbols."""
        B = x.shape[0]
        snr = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device).expand(B)
        sigma = torch.sqrt(10 ** (-snr / 10) / 2)[:, None, None, None]      # std per real dimension
        if kind == "awgn":
            return x + sigma * torch.randn_like(x)
        shape = (B, x.shape[1], 1) if fade_per_word else (B, 1, 1)
        h = torch.complex(torch.randn(shape), torch.randn(shape)) / math.sqrt(2.0)   # h ~ CN(0,1), |h|^2 ~ Exp(1)
        xc = torch.view_as_complex(x.contiguous())
        n = torch.view_as_complex((sigma * torch.randn_like(x)).contiguous())
        y = h * xc + n                                                    # fading + noise
        return torch.view_as_real(y / h)                                  # zero-forcing equaliser

    # ------------------------------------------------------------------ receiver
    def receive(self, y, mask):
        h = self.ch_dec(y.flatten(2))                                     # channel decoder
        h = h + self.pe[: y.shape[1]]
        h = self.sem_dec(h, src_key_padding_mask=~mask)                   # semantic decoder
        return self.norm_out(h) @ self.emb.weight.T + self.bias           # logits over the vocabulary

    def forward(self, ids, snr_db, kind="awgn", fade_per_word=False):
        x, mask = self.transmit(ids)
        y = self.channel(x, snr_db, kind, fade_per_word)
        return self.receive(y, mask), mask


# ---------------------------------------------------------------------- data helpers
def pad_batch(seqs):
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def encode_set(vocab, sentences):
    return [vocab.encode(s)[:MAX_TOK] for s in sentences]


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------- training
def train_codec(K, vocab, train_sents, steps=SEM_TRAIN_STEPS, batch=SEM_BATCH, seed=SEED, lr=1.5e-3,
                path=None, log_every=100):
    """End-to-end training for one K: cross-entropy loss, a random SNR ~ U[-6,18] dB for every sentence."""
    torch.manual_seed(seed + K); rng = np.random.default_rng(seed + K)
    data = encode_set(vocab, train_sents)
    model = SemanticCodec(len(vocab), K)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(10, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps))))
    hist, t0 = [], time.time()
    model.train()
    for it in range(steps):
        idx = rng.integers(0, len(data), batch)
        ids = pad_batch([data[i] for i in idx])
        snr = torch.empty(batch).uniform_(*SEM_TRAIN_SNR)
        logits, mask = model(ids, snr, "awgn")
        loss = F.cross_entropy(logits[mask], ids[mask])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % log_every == 0 or it == steps - 1:
            with torch.no_grad():
                acc = float((logits.argmax(-1)[mask] == ids[mask]).float().mean())
            hist.append((it, float(loss.detach()), acc))
            print(f"[2] K={K} step {it:5d}/{steps} loss {float(loss):.3f} token-acc {acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    if path:
        torch.save(model.state_dict(), path)
    return model, hist


def load_codec(K, vocab, path):
    m = SemanticCodec(len(vocab), K)
    sd = {k: v for k, v in torch.load(path).items() if k != "pe"}      # 'pe' is a fixed sinusoid buffer (its length depends on MAX_TOK)
    m.load_state_dict(sd, strict=False); m.eval()
    return m


if __name__ == "__main__":
    # speed benchmark / smoke test:  python semantic_model.py
    from corpus import load_corpus
    from tokens import Vocab
    tr, te = load_corpus(); v = Vocab(tr)
    for K in (1, 8):
        t = time.time()
        m, h = train_codec(K, v, tr, steps=20, log_every=10)
        print(f"K={K}: {(time.time()-t)/20:.2f} s/step, params {count_params(m)/1e6:.2f} M")
