#!/usr/bin/env python3
"""Every table in the report, built from outputs/final/ only.

Ratios are aggregate (sum over tasks / sum over tasks), not means of per-task
ratios, so a short task cannot outvote a long one.
"""
from __future__ import annotations

import json
from statistics import fmean as st_mean

from common import (
    AUTOSPEC,
    BENCH_LABEL,
    BENCHMARKS,
    DELTA_VALUE,
    DELTAS,
    FIXED,
    METHODS,
    METRIC_LABEL,
    REPORT,
    fmt,
    mean,
    metadata,
    records,
    summary_rows,
)

TABLES = REPORT / "tables"

# The two schedulers the report analyses in depth, in the order they appear.
ANALYSED = [("3_adaptive_delta", m) for m in DELTAS] + [("4_autospec", AUTOSPEC)]


def method_label(method: str) -> str:
    if method == AUTOSPEC:
        return r"autospec $\rho$=auto"
    if method.startswith("adaptive_delta"):
        return rf"adaptive $\delta$={DELTA_VALUE[method]:g}"
    return f"fixed $k$={method.rsplit('_', 1)[1]}"


def agg(rows: list[dict], numer: str, denom: str) -> float:
    n = sum(float(r[numer]) for r in rows)
    d = sum(float(r[denom]) for r in rows)
    return n / d if d else float("nan")


def write(name: str, body: str) -> None:
    (TABLES / name).write_text(body, encoding="utf-8")
    print(f"wrote tables/{name}")


# --------------------------------------------------------------- headline
def table_headline() -> None:
    rows = summary_rows()
    head = r"benchmark & model & scheduler & canvas & score & tok/s \\"
    # longtable, because 56 arms do not fit on one page.
    lines = [
        r"\begin{longtable}{lllrrr}",
        r"\caption{Every arm in the report. Score is the benchmark's primary "
        r"metric (GSM8K exact match, MATH-500 accuracy, HumanEval+/MBPP+ pass@1) "
        r"over 100 tasks; tok/s is output tokens per second.}",
        r"\label{tab:headline}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for bench in BENCHMARKS:
        got = [r for r in rows if r["benchmark"] == bench]
        first = True
        for group in ("1_dream7b_baseline", "2_sdar_b64_baseline", "3_adaptive_delta", "4_autospec"):
            model = "Dream-7B" if group == "1_dream7b_baseline" else "SDAR-1.7B"
            for r in [x for x in got if x["group"] == group]:
                label = BENCH_LABEL[bench] if first else ""
                first = False
                lines.append(
                    f"{label} & {model} & {method_label(r['method'])} & {r['canvas']} & "
                    f"{r['score']:.2f} & {r['tokens_per_sec']:.1f} \\\\"
                )
        lines.append(r"\midrule")
    lines.pop()  # the \endfoot rule closes the table
    lines.append(r"\end{longtable}")
    write("headline.tex", "\n".join(lines))


# ------------------------------------------------- tokens decoded per step
BLOCK = 64
_TOKENIZER = None


def prompt_length(rendered: str) -> int:
    """Prompt tokens exactly as run_benchmark encodes them (no special tokens)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            str(REPORT.parent / "models" / "SDAR-1.7B-Chat-b64-local"), local_files_only=True
        )
    return len(_TOKENIZER(rendered, add_special_tokens=False)["input_ids"])


def forwards_at_64(group: str, rec: dict) -> int:
    """Sequential forward passes at a common batch size of 64, excluding block commits.

    adaptive: every step is one proposal forward, then ONE verification forward
    over all of its (at most 64) candidate prefixes -- 2 per step. The run itself
    used microbatch 4, which splits that verification into more passes; this
    counts what the same step costs at autospec's batch size.
    autospec: the run's own oracle_rounds_physical, which it executed at 64.
    """
    if group == "4_autospec":
        return int(rec["denoising_steps"])
    return 2 * int(rec["denoising_steps"])


def padding_blocks(rec: dict) -> int:
    """Fully masked blocks lying wholly after the first EOS in the canvas."""
    p = prompt_length(rec["rendered_prompt"])
    c = int(rec["canvas_token_count"])
    e = rec["eos_position"] if rec["eos_position"] is not None else c
    n_blocks = -(-(p + c) // BLOCK)
    return sum(1 for i in range(n_blocks) if i * BLOCK >= p and i * BLOCK > p + e)


def table_per_step() -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{steps at cap} & \multicolumn{3}{c}{per forward pass, batch 64} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
        r"benchmark & method & canvas & all & non-padding & forwards & canvas tok & output tok \\",
        r"\midrule",
    ]
    for bench in BENCHMARKS:
        first = True
        for group, method in ANALYSED:
            R = records(group, bench, method)
            md = metadata(group, bench, method)
            fwd = sum(forwards_at_64(group, r) for r in R)
            if method == AUTOSPEC:
                cap = content_cap = "--"
            else:
                steps = sum(int(r["denoising_steps"]) for r in R)
                at_cap = sum(int(r["commitment_histogram"][-1]) for r in R)
                not_padding = sum(
                    max(0, int(r["commitment_histogram"][-1]) - padding_blocks(r)) for r in R
                )
                cap = f"{100 * at_cap / steps:.1f}\\%"
                content_cap = f"{100 * not_padding / steps:.1f}\\%"
            label = BENCH_LABEL[bench] if first else ""
            first = False
            canvas_tok = sum(int(r["canvas_token_count"]) for r in R)
            out_tok = sum(int(r["output_token_count"]) for r in R)
            lines.append(
                f"{label} & {method_label(method)} & {md['continuation_length']} & {cap} & "
                f"{content_cap} & {fmt(fwd / len(R), 1)} & {fmt(canvas_tok / fwd, 2)} & "
                f"{fmt(out_tok / fwd, 2)} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("per_step.tex", "\n".join(lines))


# ------------------------------------------------------ latency bottlenecks
def table_bottlenecks() -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{as executed} & \multicolumn{3}{c}{at batch 64} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
        r"benchmark & method & s/task & forwards & ms/forward & forwards & rows & rows/forward \\",
        r"\midrule",
    ]
    for bench in BENCHMARKS:
        first = True
        for group, method in ANALYSED:
            R = records(group, bench, method)
            secs = mean(R, "wall_time_seconds")
            executed = st_mean([r["model_forward_passes"] - r["commit_passes"] for r in R])
            fwd64 = st_mean([forwards_at_64(group, r) for r in R])
            if method == AUTOSPEC:
                rows = mean(R, "verifier_rows")
            else:  # one proposal row per step plus the verifier rows
                rows = st_mean([r["denoising_steps"] + r["verifier_rows"] for r in R])
            label = BENCH_LABEL[bench] if first else ""
            first = False
            lines.append(
                f"{label} & {method_label(method)} & {fmt(secs, 1)} & {fmt(executed, 1)} & "
                f"{fmt(1000 * secs / executed, 0)} & {fmt(fwd64, 1)} & {fmt(rows, 0)} & "
                f"{fmt(rows / fwd64, 1)} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("bottlenecks.tex", "\n".join(lines))


# ----------------------------------------------- autospec speculation books
def table_speculation() -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"benchmark & blocks & singleton & spec.\ acc & spec.\ rej & acc/block & fallback & max pooled \\",
        r"\midrule",
    ]
    for bench in BENCHMARKS:
        R = records("4_autospec", bench, AUTOSPEC)
        blocks = mean(R, "blocks_opened")
        single = mean(R, "singleton_blocks")
        acc = mean(R, "speculations_accepted")
        rej = mean(R, "speculations_rejected")
        # acc/block is the run's own `speculation_accept_rate`: accepted
        # speculations per block opened, not per speculation attempted.
        lines.append(
            f"{BENCH_LABEL[bench]} & {fmt(blocks, 1)} & {fmt(100 * single / blocks, 1)}\\% & "
            f"{fmt(acc, 1)} & {fmt(rej, 1)} & {fmt(acc / blocks, 3)} & "
            f"{fmt(mean(R, 'fallback_draws'), 1)} & {fmt(mean(R, 'max_pooled_request_rows'), 0)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("speculation.tex", "\n".join(lines))


# ------------------------------------------------------------- key numbers
def macros() -> None:
    """Numbers quoted in the prose, so the text cannot drift from the data."""
    out = {}
    for bench in BENCHMARKS:
        Ra = records("3_adaptive_delta", bench, "adaptive_delta_0p50")
        Rs = records("4_autospec", bench, AUTOSPEC)
        fa = st_mean([forwards_at_64("3_adaptive_delta", r) for r in Ra])
        fs = st_mean([forwards_at_64("4_autospec", r) for r in Rs])
        out[f"{bench}_fwd64_ratio_vs_delta0p5"] = fs / fa
        out[f"{bench}_autospec_canvas_per_fwd"] = sum(r["canvas_token_count"] for r in Rs) / sum(
            forwards_at_64("4_autospec", r) for r in Rs)
        blocks = st_mean([-(-(prompt_length(r["rendered_prompt"]) + r["canvas_token_count"]) // BLOCK)
                          - prompt_length(r["rendered_prompt"]) // BLOCK for r in Rs])
        out[f"{bench}_autospec_blocks_per_task"] = blocks
        out[f"{bench}_autospec_fwd_per_block"] = fs / blocks
    for group, method in ANALYSED[:-1]:
        for bench in BENCHMARKS:
            R = records(group, bench, method)
            out[f"{bench}_{method}_canvas_per_fwd"] = sum(r["canvas_token_count"] for r in R) / sum(
                forwards_at_64(group, r) for r in R)
    (TABLES / "key_numbers.json").write_text(json.dumps(out, indent=1))
    print("wrote tables/key_numbers.json")
    for k, v in out.items():
        print(f"   {k:48s} {v:8.2f}")


if __name__ == "__main__":
    TABLES.mkdir(exist_ok=True)
    table_headline()
    table_per_step()
    table_bottlenecks()
    table_speculation()
    macros()
