import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "report", "derived")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})
C_FIX, C_ADA, C_OLD, C_WARN = "#1b6ca8", "#d1495b", "#8d99ae", "#e08600"

main = pd.read_csv(os.path.join(DER, "main_table.csv"))
per = pd.read_csv(os.path.join(DER, "per_example_scores.csv"))
effect = pd.read_csv(os.path.join(DER, "sampling_effect.csv"))
BENCHES = ["gsm8k", "mbpp_plus", "schema_json"]
LBL = {"gsm8k": "GSM8K (exact match)", "mbpp_plus": "MBPP+ (pass@1)",
       "schema_json": "Schema JSON (exact match)"}
SHORT = {"gsm8k": "GSM8K", "mbpp_plus": "MBPP+", "schema_json": "Schema JSON"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

def klab(m): return m.split("_")[-1].replace("p", ".")
def nice(m): return ("k=" if m.startswith("fixed") else "δ=") + klab(m)

# The three configurations. The adaptive arm was never regenerated -- its rows are
# bit-identical between the two roots (verified across all 24) -- so it appears once.
GROUPS = [
    ("matched", "fixed",    "fixed, greedy",                       "#1b6ca8", "o", True),
    ("v2",      "fixed",    "fixed, entropy $T{=}0.1$ top-$p{=}0.9$", "#6a4c93", "D", True),
    ("matched", "adaptive", "adaptive, greedy",                    "#d1495b", "s", False),
]

def pareto(name, suptitle):
    """Accuracy on y, latency on x. Best is top-left.

    The fixed series are joined because they are ordered and monotone in $k$;
    the adaptive series is not, because accuracy shows no monotone trend in
    delta on any benchmark and a line would imply one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, b in zip(axes, BENCHES):
        for run, typ, label, colour, mk, connect in GROUPS:
            g = main[(main.run == run) & (main.type == typ) & (main.benchmark == b)]
            g = g.sort_values("seconds")
            yerr = np.vstack([g.corrected - g.ci_lo, g.ci_hi - g.corrected])
            ax.errorbar(g.seconds, g.corrected, yerr=yerr,
                        fmt=mk, linestyle="-" if connect else "none",
                        color=colour, ms=5, lw=1.4, capsize=2, elinewidth=0.8,
                        label=label if b == "gsm8k" else None)
            for _, r in g.iterrows():
                ax.annotate(klab(r.method), (r.seconds, r.corrected), fontsize=6,
                            xytext=(0, 7), textcoords="offset points", ha="center", color=colour)
        ax.set_title(LBL[b]); ax.set_xlabel("mean latency per task (s)")
        ax.set_ylim(-0.05, 1.15)
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
               ncol=3, fontsize=8.5)
    fig.suptitle(suptitle, y=1.04, fontsize=11, fontweight="bold")
    save(fig, name)

pareto("fig1_pareto_combined",
       "All three configurations: quality vs. wall-clock latency (best is top-left)")

# ------------------------------------------------- v2 scoring bug
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
g = main[(main.run == "v2") & (main.benchmark == "schema_json")].copy()
g["lab"] = g.method.map(nice)
x = np.arange(len(g)); w = 0.38
ax = axes[0]
ax.bar(x - w/2, g.score, w, color=C_OLD, label="as originally scored")
ax.bar(x + w/2, g.corrected, w, color=C_ADA, label="corrected (extraction-aware)")
ax.axvspan(-0.6, 3.5, color="#ffd166", alpha=0.25, zorder=0)
ax.text(1.45, 1.03, "fixed arm: 0.00 → up to 1.00", ha="center", fontsize=7.5, color="#8a5a00")
ax.set_xticks(x); ax.set_xticklabels(g.lab, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("exact match"); ax.set_ylim(0, 1.12)
ax.set_title("Schema JSON: the scoring bug inverts the result")
# Below the axes: the bars fill the plot area at both ends, so any in-axes
# placement collides with data.
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
ax = axes[1]
frac = per[(per.run == "v2") & (per.benchmark == "schema_json")].groupby("method").repaired.mean()
ax.bar(x, frac.reindex(g.method).values, 0.6,
       color=np.where(g.type == "fixed", C_FIX, C_ADA))
ax.set_xticks(x); ax.set_xticklabels(g.lab, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("fraction of completions"); ax.set_ylim(0, 1.08)
ax.set_title("Completions needing extraction repair")
fig.suptitle("Root cause: the fixed arm wraps JSON in a ```json fence; the scorer called json.loads() on it",
             y=1.06, fontsize=10)
save(fig, "fig2_scoring_bug")

# ------------------------------------------------- sampling effect
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
fx = effect[effect.method.str.startswith("fixed")]
for ax, b in zip(axes, BENCHES):
    g = fx[fx.benchmark == b].sort_values("method")
    xs = np.arange(len(g))
    ax.bar(xs - w/2, g.v2, w, color=C_OLD, label="group 2  (entropy, T=0.1, top_p=0.9)")
    ax.bar(xs + w/2, g.matched, w, color=C_FIX, label="group 1  (greedy, logp)")
    for xi, (_, r) in zip(xs, g.iterrows()):
        ax.annotate("", xy=(xi + w/2, r.matched), xytext=(xi - w/2, r.v2),
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.8, alpha=0.55))
    ax.set_xticks(xs); ax.set_xticklabels([nice(m) for m in g.method], fontsize=8)
    ax.set_title(SHORT[b]); ax.set_ylim(0, 1.12)
axes[0].set_ylabel("accuracy (fixed arm)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("The two recipes fail in opposite directions: greedy collapses at low $k$, "
             "nucleus degenerates at high $k$", y=1.05, fontsize=10.5, fontweight="bold")
save(fig, "fig3_sampling_effect")

# ------------------------------------------------- early EOS
M = main[main.run == "matched"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
ax = axes[0]
for b, ls, mk in (("gsm8k", "-", "o"), ("mbpp_plus", "--", "^")):
    for typ, c in (("fixed", C_FIX), ("adaptive", C_ADA)):
        g = M[(M.benchmark == b) & (M.type == typ)].sort_values("tokens_per_step")
        ax.plot(g.tokens_per_step, g.early_eos, mk + ls, color=c, ms=4, lw=1.2,
                label=f"{SHORT[b]}, {typ}")
ax.set_xlabel("tokens committed per step"); ax.set_ylabel("early-EOS rate")
ax.set_title("Early EOS is worst at $k\\!=\\!1$"); ax.legend(fontsize=7)
ax.annotate("fixed $k\\!=\\!1$: 19/100 MBPP+\ncompletions are empty",
            xy=(1.0, 0.36), xytext=(4.2, 0.50), fontsize=6.8, color=C_WARN,
            arrowprops=dict(arrowstyle="->", color=C_WARN, lw=0.9))

ax = axes[1]
g = M[M.benchmark == "gsm8k"]
ax.scatter(g.early_eos, g.corrected, c=np.where(g.type == "fixed", C_FIX, C_ADA), s=34)
for _, r in g.iterrows():
    ax.annotate(nice(r.method), (r.early_eos, r.corrected), fontsize=6,
                xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("early-EOS rate"); ax.set_ylabel("accuracy")
ax.set_title(f"GSM8K: accuracy tracks early EOS\n(r = {np.corrcoef(g.early_eos, g.corrected)[0,1]:.2f})")

ax = axes[2]
g = M[M.benchmark == "gsm8k"].sort_values(["type", "tokens_per_step"])
xs = np.arange(len(g))
ax.bar(xs - w/2, g.corrected, w, color=C_OLD, label="all completions")
ax.bar(xs + w/2, g.corrected_full, w,
       color=np.where(g.type == "fixed", C_FIX, C_ADA), label="full completions only")
ax.set_xticks(xs); ax.set_xticklabels([nice(m) for m in g.method], rotation=60,
                                      ha="right", fontsize=6.5)
ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.95)
ax.set_title("GSM8K: conditioning on\ncompletions that contain work")
ax.legend(fontsize=7, loc="upper left")
fig.suptitle("The matched comparison is dominated by early EOS, not by scheduling",
             y=1.06, fontsize=11, fontweight="bold")
save(fig, "fig4_eos_confound")

# ------------------------------------------------- cost
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
ax = axes[0]
for b, mk in zip(BENCHES, "os^"):
    g = M[(M.benchmark == b) & (M.type == "adaptive")].sort_values("tokens_per_step")
    ax.plot(g.tokens_per_step, g.step_cost_multiple, mk + "-", ms=4, lw=1.3, label=SHORT[b])
ax.axhline(1, color="k", ls="--", lw=1)
ax.text(4, 1.8, "cost of one fixed-arm step", fontsize=7)
ax.set_xlabel("tokens committed per step"); ax.set_ylabel("cost of one step (× fixed step)")
ax.set_title("Adaptive steps cost 6–20×"); ax.legend(fontsize=7.5)
ax = axes[1]
for b, mk in zip(BENCHES, "os^"):
    g = M[(M.benchmark == b) & (M.type == "adaptive")].sort_values("tokens_per_step")
    ax.plot(g.tokens_per_step, g.verifier_rows / g.steps, mk + "-", ms=4, lw=1.3, label=SHORT[b])
ax.set_xlabel("tokens committed per step"); ax.set_ylabel("verifier rows per step")
ax.set_title("Verifier work per step")
ax = axes[2]
for i, b in enumerate(BENCHES):
    gf = M[(M.benchmark == b) & (M.type == "fixed")]
    ga = M[(M.benchmark == b) & (M.type == "adaptive")]
    ax.scatter(gf.tokens_per_step, gf.throughput, color=C_FIX, marker="os^"[i], s=28,
               label="fixed" if i == 0 else None)
    ax.scatter(ga.tokens_per_step, ga.throughput, color=C_ADA, marker="os^"[i], s=28,
               label="adaptive" if i == 0 else None)
ax.set_yscale("log")
ax.set_xlabel("tokens committed per step"); ax.set_ylabel("output tokens / second")
ax.set_title("Realised throughput"); ax.legend(fontsize=7.5)
fig.suptitle("Verifier cost cancels the parallelism gain under both setups",
             y=1.05, fontsize=11, fontweight="bold")
save(fig, "fig5_cost")
