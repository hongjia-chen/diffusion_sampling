import pandas as pd, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "report", "derived")
main = pd.read_csv(os.path.join(DER, "main_table.csv"))
effect = pd.read_csv(os.path.join(DER, "sampling_effect.csv"))
LBL = {"gsm8k": "GSM8K (exact match)", "mbpp_plus": "MBPP+ (pass@1)",
       "schema_json": "Schema JSON (exact match)"}
BENCHES = list(LBL)

# The adaptive arm was never regenerated; its rows are bit-identical between the
# two roots, so it is listed once. Guard that, rather than assuming it.
a = main[(main.run == "v2") & (main.type == "adaptive")].set_index(["benchmark", "method"]).sort_index()
b = main[(main.run == "matched") & (main.type == "adaptive")].set_index(["benchmark", "method"]).sort_index()
cols = ["corrected", "seconds", "tokens_per_step", "steps", "early_eos", "corrected_full"]
assert a.index.equals(b.index) and np.allclose(a[cols], b[cols]), \
    "adaptive rows differ between roots; they can no longer be merged into one group"

GROUPS = [
    ("matched", "fixed",    "fixed, greedy"),
    ("v2",      "fixed",    "fixed, entropy $T\\!=\\!0.1$, top-$p\\!=\\!0.9$"),
    ("matched", "adaptive", "adaptive, greedy"),
]

def pm(m):
    if m.startswith("fixed_k_"):
        return "fixed $k\\!=\\!%s$" % m.split("_")[-1]
    return "adaptive $\\delta\\!=\\!%s$" % m.split("_")[-1].replace("p", ".")

rows = []
for bench in BENCHES:
    rows.append("\\multicolumn{7}{l}{\\textbf{%s}}\\\\[2pt]" % LBL[bench])
    for gi, (run, typ, label) in enumerate(GROUPS):
        g = main[(main.run == run) & (main.type == typ) & (main.benchmark == bench)]
        rows.append("\\multicolumn{7}{l}{\\itshape\\quad %s}\\\\[1pt]" % label)
        for _, r in g.iterrows():
            if r.early_eos == 0:
                eos = "---"
            elif r.early_eos >= 0.20:
                eos = "\\textcolor{badred}{%.2f}" % r.early_eos
            else:
                eos = "%.2f" % r.early_eos
            rows.append("\\quad\\quad %s & %.2f & %.2f & [%.2f,\\,%.2f] & %.1f & %s & %.2f \\\\" % (
                pm(r.method), r.tokens_per_step, r.corrected, r.ci_lo, r.ci_hi,
                r.seconds, eos, r.corrected_full))
        if gi < len(GROUPS) - 1:
            rows.append("\\addlinespace[3pt]")
    rows.append("\\midrule")

HEAD = ("Method & Tokens/step & Performance & 95\\% CI & Time (s) & Early EOS & Full only \\\\")
CAPTION = ("All three configurations on the same 100 tasks per benchmark. "
           "\\emph{Performance} is the primary metric with a 95\\% paired bootstrap "
           "interval; \\emph{Time} is mean wall-clock latency per task; \\emph{Early EOS} "
           "is the share of completions whose first EOS lands before token 16; "
           "\\emph{Full only} is accuracy over the remaining completions. Read "
           "\\emph{Full only} as a diagnostic, not a corrected score --- it conditions on "
           "an outcome of the method being measured (Section~\\ref{sec:eos}). The adaptive "
           "arm was run once and not regenerated, so groups 1 and 2 share its rows.")

tex = "\n".join([
    "\\begin{longtable}{lcccccc}",
    "\\caption{%s}\\label{tab:main}\\\\" % CAPTION,
    "\\toprule", HEAD, "\\midrule", "\\endfirsthead",
    "\\multicolumn{7}{l}{\\itshape Table \\thetable\\ (continued)}\\\\",
    "\\toprule", HEAD, "\\midrule", "\\endhead",
    "\\bottomrule", "\\endlastfoot",
    "\n".join(rows[:-1]),
    "\\end{longtable}",
]) + "\n"
open(os.path.join(ROOT, "report", "table_main_combined.tex"), "w").write(tex)

# ---- the cross-setup comparison, paired on identical tasks (unchanged) -----
fx = effect[effect.method.str.startswith("fixed")]
tab = ["\\begin{tabular}{lcccc}", "\\toprule",
       "Benchmark & $k\\!=\\!1$ & $k\\!=\\!2$ & $k\\!=\\!4$ & $k\\!=\\!8$ \\\\", "\\midrule"]
for bench in BENCHES:
    g = fx[fx.benchmark == bench].sort_values("method")
    cells = []
    for _, r in g.iterrows():
        cell = "%.2f\\,$\\to$\\,%.2f" % (r.v2, r.matched)
        if r.hi < 0:
            cell = "\\textcolor{badred}{%s}" % cell
        elif r.lo > 0:
            cell = "\\textbf{%s}" % cell
        cells.append(cell)
    tab.append("%s & %s \\\\" % (LBL[bench].split(" (")[0], " & ".join(cells)))
tab += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "report", "table_sampling.tex"), "w").write("\n".join(tab) + "\n")
print("wrote table_main_combined.tex, table_sampling.tex")
