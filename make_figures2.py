"""
make_figures2.py -- every figure of the Exercise-2 paper and slides, drawn from the saved result files.
Nothing is simulated here, so the figures can be restyled in seconds.
    figures/paper/<name>.pdf   compact, 7 pt fonts, sized for IEEE columns (3.4 in) / page (7.1 in)
    figures/slide/<name>.png   large fonts for 16:9 slides
Colour logic (colour-blind safe, Okabe-Ito)
    semantic system : colour = K (1 sky, 2 green, 4 blue, 8 vermillion), thick solid lines
    conventional    : character Huffman = black, word Huffman = purple; MCS = line style / marker
                      (QPSK 1/2 dotted, 16-QAM 1/2 dashed, 64-QAM 3/4 solid-thin)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from config import *
from rate_tables import monotone

BLUE, GREEN, VERM, ORANGE, SKY, PURPLE, GREY, BLACK = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#56B4E9", "#CC79A7", "#7f7f7f", "#000000"
KCOL = {1: SKY, 2: GREEN, 4: BLUE, 8: VERM}
SRC_COL = {"char": BLACK, "word": PURPLE}
MCS_LS = {"QPSK 1/2": ":", "16-QAM 1/2": "--", "64-QAM 3/4": "-"}
KIND, SC = "paper", 1.0


def setup(kind):
    global KIND, SC
    KIND = kind
    SC = 1.0 if kind == "paper" else 1.75
    plt.rcParams.update({
        "font.size": 7 * SC, "axes.labelsize": 7.5 * SC, "legend.fontsize": 6 * SC if kind == "paper" else 8.5,
        "xtick.labelsize": 6.5 * SC, "ytick.labelsize": 6.5 * SC, "axes.titlesize": 7.5 * SC,
        "lines.linewidth": 1.1 * SC, "lines.markersize": 3.2 * SC, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def fig(w, h, **kw):
    return plt.subplots(figsize=(w * SC, h * SC), **kw)


def save(f, name, rect=(0, 0, 1, 1)):
    d = os.path.join(FIG_DIR, KIND)
    os.makedirs(d, exist_ok=True)
    f.tight_layout(pad=0.4, rect=rect)
    f.savefig(os.path.join(d, name + (".pdf" if KIND == "paper" else ".png")), bbox_inches="tight")
    plt.close(f)


def L(name):
    return json.load(open(os.path.join(RES_DIR, name)))


def fs(paper, slide):
    return paper * SC if KIND == "paper" else slide


# ------------------------------------------------------------------------------------------ Task 1
def fig_t1_pairs():
    s = L("task1_pair_summary.json")
    kinds = list(s.keys())
    f, ax = fig(3.4, 2.3)
    w = 0.26
    for i, (m, lab, c) in enumerate((("bleu", "BLEU", BLUE), ("wacc", "word accuracy", GREEN), ("sim", "similarity", VERM))):
        ax.bar(np.arange(len(kinds)) + (i - 1) * w, [s[k][m] for k in kinds], w, label=lab, color=c)
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels([k.replace(" ", "\n", 1) for k in kinds], fontsize=fs(5.8, 9))
    ax.set_ylim(0, 1.18); ax.set_ylabel("score (mean of 5 pairs)")
    ax.legend(ncol=3, frameon=False, loc="upper center")
    save(f, "t1_pairs")


def fig_t1_conv():
    sw = L("task1_conv_sweeps.json")["sweeps"]
    f, axs = fig(7.1, 2.4, ncols=3)
    for ax, key, ttl in zip(axs, ("bleu", "sim", "exact"), ("BLEU", "sentence similarity", "fraction reproduced exactly")):
        for src in ("char", "word"):
            for n in PLOT_MCS:
                r = sw[src]["awgn"][n]
                ax.plot(r["snr"], r[key], color=SRC_COL[src], ls=MCS_LS[n], lw=1.0 * SC)
        ax.set_xlabel("SNR $E_s/N_0$ [dB]"); ax.set_title(ttl); ax.set_xlim(-6, 26); ax.set_ylim(-0.02, 1.03)
    if KIND == "paper":
        hs = [Line2D([], [], color=SRC_COL[s], label=f"{s} Huffman") for s in SRC_COL] + \
             [Line2D([], [], color="k", ls=MCS_LS[n], label=n) for n in PLOT_MCS]
        axs[0].legend(handles=hs, frameon=False, loc="lower right", fontsize=5.3, ncol=1)
    else:
        hs = [Line2D([], [], color=SRC_COL[s], label=f"{s} Huffman") for s in SRC_COL] + \
             [Line2D([], [], color="k", ls=MCS_LS[n], label=n) for n in PLOT_MCS]
        axs[0].legend(handles=hs, frameon=False, loc="lower right", fontsize=8)
    axs[0].set_ylabel("score")
    save(f, "t1_conv")


# ------------------------------------------------------------------------------------------ Task 2
def _sem():
    return L("task2_sem_curves.json")


def fig_t2_vs_conv():
    sem, sw = _sem(), L("task1_conv_sweeps.json")["sweeps"]
    f, axs = fig(7.1, 2.5, ncols=3)
    for ax, key, ttl in zip(axs, ("bleu", "sim", "exact"), ("BLEU", "sentence similarity", "fraction reproduced exactly")):
        for src in ("char", "word"):
            for n in PLOT_MCS:
                r = sw[src]["awgn"][n]
                ax.plot(r["snr"], r[key], color=GREY if src == "char" else "#b9a0c8", ls=MCS_LS[n], lw=0.9 * SC)
        for K in K_LIST:
            r = sem["K"][str(K)]["awgn"]
            ax.plot(r["snr"], r[key], color=KCOL[K], lw=1.4 * SC, label=f"semantic K={K}")
        if key == "sim":
            ax.axhline(EPS_TH, color="k", lw=0.6, ls=":"); ax.text(-9.5, EPS_TH + 0.012, r"$\varepsilon_{th}$", fontsize=fs(6, 9))
        ax.set_xlabel("SNR $E_s/N_0$ [dB]"); ax.set_title(ttl); ax.set_xlim(-10, 26); ax.set_ylim(-0.02, 1.03)
    axs[0].set_ylabel("score")
    h = [Line2D([], [], color=KCOL[K], lw=1.4 * SC, label=f"semantic K={K}") for K in K_LIST]
    h += [Line2D([], [], color=GREY, lw=1, label="conv. char Huffman"), Line2D([], [], color="#b9a0c8", lw=1, label="conv. word Huffman")]
    axs[1].legend(handles=h, frameon=False, loc="lower right", fontsize=fs(5.3, 8))
    save(f, "t2_vs_conv")


def fig_t2_uses():
    sem, sw = _sem(), L("task1_conv_sweeps.json")["sweeps"]
    f, axs = fig(7.1, 2.3, ncols=3, sharey=True)
    for ax, g in zip(axs, (0.0, 6.0, 12.0)):
        for src in ("char", "word"):
            xs, ys = [], []
            for m in MCS_TABLE:
                r = sw[src]["awgn"][m["name"]]; i = int(np.argmin(np.abs(np.array(r["snr"]) - g)))
                xs.append(r["avg_symbols"]); ys.append(r["sim"][i])
            ax.plot(xs, ys, "o-", color=SRC_COL[src], mfc="none", lw=0.9 * SC, label=f"conv. {src} Huffman (6 MCS)")
        xs = [K * sem["L_bar_tx"] for K in K_LIST]
        ys = [sem["K"][str(K)]["awgn"]["sim"][int(np.argmin(np.abs(np.array(sem["snr_awgn"]) - g)))] for K in K_LIST]   # nearest grid point (quick grid is coarser)
        ax.plot(xs, ys, "s-", color=VERM, lw=1.4 * SC, label="semantic, K = 1,2,4,8")
        for K, x, y in zip(K_LIST, xs, ys):
            ax.annotate(f"K={K}", (x, y), textcoords="offset points", xytext=(2, -8), fontsize=fs(5.5, 8), color=VERM)
        ax.axhline(EPS_TH, color="k", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("channel uses per sentence"); ax.set_title(f"SNR = {g:.0f} dB"); ax.set_ylim(-0.02, 1.03)
        ax.set_xticks([10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["10", "20", "50", "100", "200", "500"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlim(12, 700)
    axs[0].set_ylabel("sentence similarity"); axs[2].legend(frameon=False, loc="lower right", fontsize=fs(5.3, 8))
    save(f, "t2_uses")


def fig_t2_lut():
    sem = _sem()
    f, ax = fig(3.4, 2.5)
    for K in K_LIST:
        r = sem["K"][str(K)]
        ax.plot(r["awgn"]["snr"], monotone(r["awgn"]["sim"]), color=KCOL[K], lw=1.3 * SC, label=f"K={K}, AWGN (LUT)")
        ax.plot(r["rayleigh"]["snr"], r["rayleigh"]["sim"], color=KCOL[K], ls="--", lw=0.9 * SC)
    ax.axhline(EPS_TH, color="k", lw=0.6, ls=":")
    ax.set_xlabel("SNR $\\gamma$ [dB]"); ax.set_ylabel("$\\varepsilon(K,\\gamma)$"); ax.set_xlim(-10, 30); ax.set_ylim(0, 1.03)
    h, _ = ax.get_legend_handles_labels(); h.append(Line2D([], [], color="k", ls="--", label="Rayleigh + ZF (dashed)"))
    ax.legend(handles=h, frameon=False, loc="lower right", fontsize=fs(5.5, 8))
    save(f, "t2_lut")


def fig_t2_mismatch():
    mm = L("task2_mismatch.json")
    sets = ["in-distribution", "unseen template", "unseen words"]
    i12 = mm["snr"].index(12.0)
    rows = [("semantic K=2", mm["semantic"]["2"], GREEN), ("semantic K=4", mm["semantic"]["4"], BLUE), ("semantic K=8", mm["semantic"]["8"], VERM),
            ("char Huffman\n16-QAM 1/2", mm["conv"]["char Huffman, 16-QAM 1/2"], BLACK),
            ("word Huffman\n16-QAM 1/2", mm["conv"]["word Huffman, 16-QAM 1/2"], PURPLE)]
    f, axs = fig(7.1, 2.2, ncols=2)
    for ax, key, ttl in zip(axs, ("sim", "exact"), ("sentence similarity at 12 dB", "fraction reproduced exactly at 12 dB")):
        w = 0.16
        for i, (lab, d, c) in enumerate(rows):
            ax.bar(np.arange(3) + (i - 2.0) * w, [d[s][key][i12] for s in sets], w, label=lab.replace("\n", " "), color=c)
        ax.set_xticks(range(3)); ax.set_xticklabels(sets); ax.set_title(ttl); ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0].legend(frameon=False, ncol=3, loc="upper center", fontsize=fs(5.5, 8), columnspacing=0.8, handlelength=1.0)
    save(f, "t2_mismatch")


def fig_t2_training():
    f, ax = fig(3.4, 2.2)
    for K in K_LIST:
        p = os.path.join(RES_DIR, f"sem_train_hist_K{K}.json")
        if os.path.exists(p):
            h = np.array(json.load(open(p))["hist"]); ax.plot(h[:, 0], h[:, 1], color=KCOL[K], label=f"K={K}")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss"); ax.set_yscale("log"); ax.legend(frameon=False, ncol=2)
    save(f, "t2_training")


# ------------------------------------------------------------------------------------------ Task 3
RA_STYLE = {"MILP (optimal)": dict(color=BLUE, ls="-", marker="o"), "Greedy": dict(color=GREEN, ls="--", marker="s"),
            "Fixed K=4 (MILP)": dict(color=ORANGE, ls="-.", marker="^"), "Bit-rate-optimal + K": dict(color=GREY, ls=":", marker="v"),
            "DNN (imitation)": dict(color=VERM, ls="--", marker="D"), "DNN + repair": dict(color=PURPLE, ls="-", marker="P")}


def fig_t3_main():
    r = L("task3_sem_results.json"); p = np.array(r["pmax_dbm"])
    f, axs = fig(7.1, 2.3, ncols=3)
    for ax, (k, yl, sc) in zip(axs, (("total_rate", "total semantic rate [ksuts/s]", 1e3), ("pct_served", "users with $S_u\\geq S_{min}$ [%]", 1), ("p5_rate", "cell-edge rate (5th pct.) [ksuts/s]", 1e3))):
        for m, st in RA_STYLE.items():
            ax.plot(p, np.array(r[m][k]) / sc, label=m, **st, mfc="none" if m == "DNN (imitation)" else None)
        ax.set_xlabel("$P_{max}$ [dBm]"); ax.set_ylabel(yl)
    axs[1].set_ylim(0, 102)
    axs[1].legend(frameon=False, loc="lower center", fontsize=fs(5.2, 8))
    save(f, "t3_main")


def fig_t3_time_k():
    r = L("task3_sem_results.json")
    f, axs = fig(7.1, 2.2, ncols=2, gridspec_kw=dict(width_ratios=[1, 1.15]))
    names = list(r["time_ms_per_drop"].keys()); vals = [r["time_ms_per_drop"][n] for n in names]
    ax = axs[0]
    ax.barh(range(len(names)), vals, color=[RA_STYLE[n]["color"] for n in names])
    ax.set_xscale("log"); ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace(" ", "\n", 1) if len(n) > 14 else n for n in names], fontsize=fs(5.5, 8))
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:.2f} ms", va="center", fontsize=fs(6, 9))
    ax.invert_yaxis(); ax.set_xlabel("time per drop at 30 dBm [ms]"); ax.set_xlim(right=max(vals) * 10)
    ax = axs[1]; p = np.array(r["pmax_dbm"]); bottom = np.zeros(len(p))
    for K in K_LIST:
        v = np.array([u[str(K)] for u in r["k_usage"]]) * 100
        ax.bar(p, v, 1.7, bottom=bottom, color=KCOL[K], label=f"K={K}"); bottom += v
    ax.set_xlabel("$P_{max}$ [dBm]"); ax.set_ylabel("RBs using this K [%]"); ax.set_ylim(0, 122); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=fs(5.5, 8))
    save(f, "t3_time_k")


# ------------------------------------------------------------------------------------------ Task 4
SYS_STYLE = {
    "Semantic, adaptive K": dict(color=VERM, ls="-", marker="o"),
    "Semantic, fixed K=4": dict(color=BLUE, ls="--", marker="s"),
    "Semantic, fixed K=8": dict(color=SKY, ls=":", marker="^"),
    "Conventional, char Huffman + AMC": dict(color=BLACK, ls="-", marker="v"),
    "Conventional, word Huffman + AMC": dict(color=PURPLE, ls="-", marker="D"),
    "Conventional, char Huffman + meaning-aware MCS": dict(color=GREY, ls="--", marker="x"),
    "Conventional, word Huffman + meaning-aware MCS": dict(color="#b9a0c8", ls="--", marker="P"),
}


def fig_t4_main():
    r = L("task4_compare.json"); p = np.array(r["pmax_dbm"]); sw = r["sweep"]
    f, axs = fig(7.1, 4.0, nrows=2, ncols=3)
    panels = [("total_rate", "total semantic rate [ksuts/s]", 1e3), ("pct_served", "users with $S_u\\geq S_{min}$ [%]", 1),
              ("p5_rate", "cell-edge (5th pct.) rate [ksuts/s]", 1e3), ("uses_per_sentence", "channel uses / sentence", 1),
              ("exact_frac", "sentences reproduced exactly", 1), ("mean_eps", "mean similarity of delivered sentences", 1)]
    for ax, (k, yl, sc) in zip(axs.ravel(), panels):
        for s, st in SYS_STYLE.items():
            ax.plot(p, np.array(sw[s][k]) / sc, label=s.replace("Conventional, ", "conv. ").replace("Semantic, ", "sem. "), **st, ms=2.6 * SC)
        ax.set_xlabel("$P_{max}$ [dBm]"); ax.set_ylabel(yl)
    axs[0, 1].set_ylim(0, 102)
    h, l = axs[0, 0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=fs(5.2, 8.5))
    save(f, "t4_main", rect=(0, 0.09, 1, 1))


def fig_t4_bw_rob():
    r = L("task4_compare.json")
    f, axs = fig(7.1, 2.3, ncols=2, gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]; names = [s for s in SYS_STYLE if s in r["bandwidth"]]
    vals = [r["bandwidth"][s]["mean_mhz"] for s in names]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]          # (quick mode with weak models can give inf/nan)
    ax.barh(range(len(names)), vals, color=[SYS_STYLE[s]["color"] for s in names])
    ax.set_yticks(range(len(names))); ax.set_yticklabels([s.replace("Conventional, ", "conv. ").replace("Semantic, ", "sem. ").replace(" + ", "\n+ ") for s in names], fontsize=fs(5, 7))
    for i, v in enumerate(vals):
        ax.text(v + 0.03, i, f"{v:.2f} MHz", va="center", fontsize=fs(5.5, 8.5))
    ax.invert_yaxis(); ax.set_xlabel("bandwidth to serve all 8 users [MHz]"); ax.set_xlim(right=max(max(vals), 0.1) * 1.35)
    ax = axs[1]; rb = r["robustness"]
    rows = [(f"sem. K={K}", rb["sem"][str(K)]) for K in K_LIST]
    for src in ("char", "word"):
        best = min((rb["conv"][f"{src}|{m['name']}"] for m in MCS_TABLE), key=lambda d: d["awgn"])
        rows.append((f"conv. {src} (best MCS)", best))
    x = np.arange(len(rows)); w = 0.38
    ya = [min(d["awgn"], 40) for _, d in rows]; yr = [min(d["rayleigh"], 40) for _, d in rows]
    ax.bar(x - w / 2, ya, w, color=BLUE, label="AWGN"); ax.bar(x + w / 2, yr, w, color=VERM, label="Rayleigh + ZF")
    for i, (_, d) in enumerate(rows):                     # similarity 0.9 never reached (e.g. K=1) -> hatched full-height bar
        if not np.isfinite(d["awgn"]):
            ax.bar([i - w / 2, i + w / 2], [40, 40], w, fill=False, hatch="///", edgecolor="0.4", lw=0.4)
            ax.text(i, 41, "never\nreached", ha="center", va="bottom", fontsize=fs(5, 8))
    ax.set_ylim(0, 48)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n, _ in rows], fontsize=fs(5.3, 8), rotation=0)
    ax.set_ylabel("SNR for similarity $\\geq$ 0.9 [dB]"); ax.legend(frameon=False, fontsize=fs(5.5, 8.5))
    save(f, "t4_bw_rob")


ALL = [fig_t1_pairs, fig_t1_conv, fig_t2_vs_conv, fig_t2_uses, fig_t2_lut, fig_t2_mismatch, fig_t2_training,
       fig_t3_main, fig_t3_time_k, fig_t4_main, fig_t4_bw_rob]


def main():
    for kind in ("paper", "slide"):
        setup(kind)
        for fn in ALL:
            try:
                fn()
            except FileNotFoundError as e:
                print("[figures] skipped", fn.__name__, "(missing results):", e.filename)
    print("[figures] written to", FIG_DIR)


if __name__ == "__main__":
    main()
