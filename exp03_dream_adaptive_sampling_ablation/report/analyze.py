"""Analysis for the Dream scheduler comparison.

Reads both experiment roots and emits paired, per-task data:

  outputs/v2          fixed arm on the Dream/ADJUST recipe, adaptive arm greedy.
                      The two arms are not comparable to each other.
  outputs/v3_matched  both arms greedy. This is the scheduler comparison.

The same 100 tasks run in both roots under every method, so every comparison
here -- across methods within a root, and across roots within a method -- is
paired, and the intervals are paired bootstraps rather than independent ones.

Writes report/derived/{per_example_scores,main_table,sampling_effect}.csv and
report/derived/diagnostics.json.
"""
import json, os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dream_ablation.scoring import score_schema_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "report", "derived")
os.makedirs(DER, exist_ok=True)

ROOTS = {"v2": os.path.join("outputs", "v2"),
         "matched": os.path.join("outputs", "v3_matched")}
BENCHES = ["gsm8k", "mbpp_plus", "schema_json"]
ORDER = ["fixed_k_1", "fixed_k_2", "fixed_k_4", "fixed_k_8",
         "adaptive_delta_0p10", "adaptive_delta_0p15", "adaptive_delta_0p20",
         "adaptive_delta_0p30", "adaptive_delta_0p50", "adaptive_delta_0p75",
         "adaptive_delta_1p00", "adaptive_delta_1p50"]

schema_ref = {}
for path in glob.glob(os.path.join(ROOT, "data", "schema_json", "*.jsonl")):
    for line in open(path):
        row = json.loads(line)
        schema_ref[row.get("example_id") or row.get("id")] = row

# ---------------------------------------------------------------- per-example
rows = []
for run, rel in ROOTS.items():
    out = os.path.join(ROOT, rel)
    for bench in BENCHES:
        for method in ORDER:
            base = os.path.join(out, bench, method)
            scored = os.path.join(base, "records.scored.jsonl")
            plain = os.path.join(base, "records.jsonl")
            # records.jsonl holds the verdict the run originally wrote, which is
            # what "as reported" means in the scoring-bug figure. For schema_json
            # the corrected verdict is recomputed below, so reading the rescored
            # file here would collapse that comparison. MBPP+ has no choice:
            # pass@1 only exists in the scored file.
            path = scored if (os.path.exists(scored) and bench == "mbpp_plus") else plain
            if not os.path.exists(path):
                continue
            for line in open(path):
                r = json.loads(line)
                score = bool(r.get("score"))
                corrected, repaired = score, False
                if bench == "schema_json":
                    example = schema_ref.get(r["example_id"])
                    if example is not None:
                        result = score_schema_json(
                            r["raw_completion"], example["target"], example["schema"])
                        corrected = bool(result["score"])
                        repaired = result["json_repair"] is not None
                rows.append(dict(
                    run=run, benchmark=bench, method=method, example_id=r["example_id"],
                    score=score, corrected_score=corrected, repaired=repaired,
                    tokens_per_step=r.get("mean_committed_tokens_per_step"),
                    steps=r.get("denoising_steps"), wall=r.get("wall_time_seconds"),
                    out_tokens=r.get("output_token_count"),
                    early_eos=bool(r.get("early_eos")),
                    verifier_rows=r.get("verifier_rows", 0) or 0))
per = pd.DataFrame(rows)
per.to_csv(os.path.join(DER, "per_example_scores.csv"), index=False)

# ---------------------------------------------------------------- paired bootstrap
rng = np.random.default_rng(271828)
BOOT = 10000

def boot_ci(values):
    values = np.asarray(values, float)
    draws = values[rng.integers(0, len(values), size=(BOOT, len(values)))].mean(axis=1)
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)

def paired_ci(a, b):
    """CI on the mean of b - a over the same tasks, and the sign-consistency."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    draws = diff[rng.integers(0, len(diff), size=(BOOT, len(diff)))].mean(axis=1)
    return diff.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)

main = []
for run in ROOTS:
    sub_run = per[per.run == run]
    for bench in BENCHES:
        sub = sub_run[sub_run.benchmark == bench]
        for method in ORDER:
            g = sub[sub.method == method]
            if g.empty:
                continue
            lo, hi = boot_ci(g.corrected_score.values)
            ok = g[~g.early_eos]
            main.append(dict(
                run=run, benchmark=bench, method=method, n=len(g),
                score=g.score.mean(), corrected=g.corrected_score.mean(),
                ci_lo=lo, ci_hi=hi,
                tokens_per_step=g.tokens_per_step.mean(), steps=g.steps.mean(),
                seconds=g.wall.mean(), s_per_step=g.wall.mean() / g.steps.mean(),
                out_tokens=g.out_tokens.mean(), early_eos=g.early_eos.mean(),
                n_full=len(ok),
                corrected_full=ok.corrected_score.mean() if len(ok) else np.nan,
                verifier_rows=g.verifier_rows.mean(), repaired=g.repaired.mean()))
main = pd.DataFrame(main)
main["type"] = np.where(main.method.str.startswith("fixed"), "fixed", "adaptive")
main["throughput"] = main.out_tokens / main.seconds
base = main[main.type == "fixed"].groupby(["run", "benchmark"]).s_per_step.mean()
main["step_cost_multiple"] = [
    r.s_per_step / base[(r.run, r.benchmark)] for r in main.itertuples()]
main.to_csv(os.path.join(DER, "main_table.csv"), index=False)

# ---------------------------------------------------------------- sampling effect
# The fixed arm ran the same tasks under both recipes, so this is paired.
effect = []
for bench in BENCHES:
    for method in ORDER:
        a = per[(per.run == "v2") & (per.benchmark == bench) & (per.method == method)]
        b = per[(per.run == "matched") & (per.benchmark == bench) & (per.method == method)]
        if a.empty or b.empty:
            continue
        a = a.set_index("example_id").sort_index()
        b = b.set_index("example_id").sort_index()
        if not a.index.equals(b.index):
            raise SystemExit(f"task sets differ for {bench}/{method}")
        delta, lo, hi = paired_ci(a.corrected_score.values, b.corrected_score.values)
        effect.append(dict(
            benchmark=bench, method=method,
            v2=a.corrected_score.mean(), matched=b.corrected_score.mean(),
            delta=delta, lo=lo, hi=hi,
            identical=bool((a.corrected_score.values == b.corrected_score.values).all()),
            v2_eos=a.early_eos.mean(), matched_eos=b.early_eos.mean(),
            v2_seconds=a.wall.mean(), matched_seconds=b.wall.mean()))
effect = pd.DataFrame(effect)
effect.to_csv(os.path.join(DER, "sampling_effect.csv"), index=False)

# ---------------------------------------------------------------- diagnostics
m = main[main.run == "matched"]
diag = {"early_eos_confound": {}, "step_cost_multiple": {}, "gpu_timing_skew_pct": {}}
for bench in BENCHES:
    g = m[m.benchmark == bench]
    diag["early_eos_confound"][bench] = {
        "corr_eos_vs_accuracy": float(np.corrcoef(g.early_eos, g.corrected)[0, 1]),
        "unconditional_best": g.loc[g.corrected.idxmax(), "method"],
        "full_completion_best": g.loc[g.corrected_full.idxmax(), "method"],
    }
    f = g[g.type == "fixed"].s_per_step
    diag["gpu_timing_skew_pct"][bench] = float(100 * (f.max() - f.min()) / f.mean())
    diag["step_cost_multiple"][bench] = dict(
        zip(g[g.type == "adaptive"].method, g[g.type == "adaptive"].step_cost_multiple.round(2)))
json.dump(diag, open(os.path.join(DER, "diagnostics.json"), "w"), indent=2, default=float)

pd.set_option("display.width", 220)
print(m[["benchmark", "method", "corrected", "ci_lo", "ci_hi", "seconds",
         "tokens_per_step", "early_eos", "corrected_full", "n_full"]].round(3).to_string(index=False))
print()
print(json.dumps(diag["early_eos_confound"], indent=2))
