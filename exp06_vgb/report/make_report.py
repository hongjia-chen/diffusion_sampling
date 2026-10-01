#!/usr/bin/env python3
"""Remasking with an aligned verifier: report/report.html.

exp05's page (../exp05_adaptive_re/report/remask_report.html) rebuilt for the
exp06 runs, all read from finished outputs:

1. Accuracy against tokens/s per benchmark: SDAR-8B fixed:k (outputs/sdar8b),
   SDAR-1.7B fixed:k (exp05 B1), cross8b and cross8b_swap with remasking on and
   off, and VGB Track A.
2. Remasking on against off, task by task at the same delta, for exp05 (Dream
   verifies), cross8b (8B verifies) and the swap (1.7B verifies).
3. What kind of edit it makes: the first edit of every edited run, classified
   the same way for all three (scripts/analyze_cross8b.py), and every edit
   that flipped a score in cross8b and the swap.
3b. What kind of error each fixed and adaptive arm makes, and what remasking
   does to each kind (outputs/error_taxonomy.json, from
   scripts/error_taxonomy.py and the hand labels in report/data/).
4. A decoding player for the tasks in report/trajectories.py: 8B fixed:1,
   1.7B fixed:1, cross8b delta=0.1 and delta=1 (each with remasking on or
   off), and VGB Track A's three arms (forward only, lambda=0.5, lambda=1),
   all from traced re-runs that must reproduce the recorded runs.

    PYTHONPATH=../harness/vendor_env/tf452:. python report/make_report.py
"""
from __future__ import annotations

import ast
import difflib
import json
import math
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parent
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(REPORT))

import analyze_cross8b as ac  # noqa: E402
import trajectories as tj  # noqa: E402

EXP05 = PROJECT.parent / "exp05_adaptive_re"
TEMPLATE = REPORT / "report.template.html"
OUTPUT = REPORT / "report.html"
BENCHMARKS = ["gsm8k", "mbpp_plus", "humaneval_plus"]
BENCH_LABEL = {"gsm8k": "GSM8K", "mbpp_plus": "MBPP+", "humaneval_plus": "HumanEval+"}
METRIC = {"gsm8k": "exact match", "mbpp_plus": "pass@1", "humaneval_plus": "pass@1"}
DELTAS = {"0p10": "0.1", "1p00": "1"}
STOPS = {151643, 151645}
TABLE_ROWS = 12  # remask inspector: this many least-probable re-scored tokens
DIRECT = 0.9     # a flip is the edit's own when the answers otherwise align this closely

EXPERIMENTS = {
    # key: (label, who proposes -> who verifies, css token, records root)
    "exp05": ("exp05", "1.7B → Dream-7B", "--ctx", EXP05),
    "cross8b": ("cross8b", "1.7B → 8B", "--s3", PROJECT / "outputs/cross8b"),
    "swap": ("swap", "8B → 1.7B", "--s4", PROJECT / "outputs/cross8b_swap"),
}


def summary(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else list(data.values())


def tip(bench: str, score: float, tps: float, sec: float, setting: str) -> list[list[str]]:
    return [["setting", setting], [METRIC[bench], f"{score:.2f}"], ["tokens/s", f"{tps:.1f}"],
            ["s/task", f"{sec:.1f}"]]


# ------------------------------------------------------------------ frontiers
def frontiers() -> list[dict]:
    b8 = summary(PROJECT / "outputs/sdar8b/comparison/summary.json")
    b17 = summary(EXP05 / "outputs/baseline/sdar/comparison/summary.json")
    cross = summary(PROJECT / "outputs/cross8b/comparison/summary.json")
    swap = summary(PROJECT / "outputs/cross8b_swap/comparison/summary.json")
    vgb = summary(PROJECT / "outputs/vgb_trackA/comparison/summary.json")
    vgb_label = {"vgb_fwdonly_sl8x8_T1": ("forward only", "fwd"),
                 "vgb_lam0p5_interior_mom1_sl8x8_T1": ("λ=0.5", "λ0.5"),
                 "vgb_lam1_interior_mom1_sl8x8_T1": ("λ=1", "λ1")}
    out = []
    for bench in BENCHMARKS:
        series = []

        def fixed(rows, key, name, css, shape, dash, label_at):
            pts = []
            for r in (r for r in rows if r["benchmark"] == bench):
                k = r["method"].rsplit("_", 1)[1]
                y, x, sec = r["primary_score"], r["tokens_per_second"], r["seconds_per_task"]
                pts.append({"x": x, "y": y, "sec": sec, "method": f"fixed k={k}",
                            "label": f"k={k}" if k in ("1", "8") else None, "labelAt": label_at,
                            "tip": tip(bench, y, x, sec, f"fixed k={k}")})
            series.append({"key": key, "name": name, "css": css, "shape": shape, "line": True,
                           "dash": dash, "points": pts})

        fixed(b8, "b8", "SDAR-8B fixed:k", "--s1", "circle", False, [0, -10, "middle"])
        fixed(b17, "b17", "SDAR-1.7B fixed:k", "--s2", "square", True, [0, 16, "middle"])
        for rows, key, name, css, shape in ((cross, "cross8b", "cross8b · 1.7B → 8B", "--s3", "triangle"),
                                            (swap, "swap", "swap · 8B → 1.7B", "--s4", "diamond")):
            pts = []
            for r in (r for r in rows if r["benchmark"] == bench):
                d = DELTAS[r["method"].rsplit("_", 1)[1]]
                on = "remask" in r["method"]
                y, x, sec = r["primary_score"], r["tokens_per_second"], r["seconds_per_task"]
                pts.append({"x": x, "y": y, "sec": sec, "method": f"δ={d} · remasking {'on' if on else 'off'}",
                            "hollow": not on, "label": f"δ{d}" if on else None, "labelAt": [8, 4, "start"],
                            "tip": tip(bench, y, x, sec, f"δ={d}, remasking {'on' if on else 'off'}")})
            series.append({"key": key, "name": name, "css": css, "shape": shape, "line": False, "points": pts})
        pts = []
        for r in (r for r in vgb if r["benchmark"] == bench):
            long, short = vgb_label[r["method"]]
            y, x, sec = r["primary_score"], r["tokens_per_second"], r["seconds_per_task"]
            pts.append({"x": x, "y": y, "sec": sec, "method": long, "label": short, "labelAt": [0, -10, "middle"],
                        "tip": tip(bench, y, x, sec, f"VGB {long}, 512 moves/block")})
        series.append({"key": "vgb", "name": "VGB Track A · 8B/1.7B ratio", "css": "--ctx", "shape": "plus",
                       "line": True, "dash": True, "points": pts})
        out.append({"bench": bench, "label": BENCH_LABEL[bench], "metric": METRIC[bench], "series": series})
    return out


# ------------------------------------------------------------------ paired impact
def token_ids(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def edits_of(record: dict) -> list[dict]:
    edits = record.get("remask_edits") or []
    return ast.literal_eval(edits) if isinstance(edits, str) else edits


def impact(tok) -> tuple[list[dict], list[dict]]:
    """Per benchmark: rows of (experiment, delta) on/off, plus every score flip."""
    b8 = {r["benchmark"]: r["primary_score"] for r in summary(PROJECT / "outputs/sdar8b/comparison/summary.json")
          if r["method"] == "fixed_k_1"}
    b17 = {r["benchmark"]: r["primary_score"] for r in summary(EXP05 / "outputs/baseline/sdar/comparison/summary.json")
           if r["method"] == "fixed_k_1"}
    charts, flips = [], []
    for bench in BENCHMARKS:
        rows = []
        for exp, (label, roles, css, root) in EXPERIMENTS.items():
            for delta, dl in DELTAS.items():
                remask, plain = ac.arms(exp, bench, delta, root)
                n = len(remask)
                on = [bool(remask[k]["score"]) for k in remask]
                off = [bool(plain[k]["score"]) for k in remask]
                fixed = sum(a and not b for a, b in zip(on, off))
                broke = sum(b and not a for a, b in zip(on, off))
                edited = [k for k in remask if int(remask[k].get("tokens_changed_by_remask", 0)) > 0]
                rows.append({
                    "exp": exp, "label": label, "roles": roles, "css": css, "delta": dl,
                    "on": sum(on) / n, "off": sum(off) / n, "fixed": fixed, "broke": broke,
                    "p": ac.mcnemar(fixed, broke), "edited": len(edited),
                    "edits": sum(int(remask[k]["tokens_changed_by_remask"]) for k in edited),
                    "changed": sum(remask[k]["raw_completion"] != plain[k]["raw_completion"] for k in remask),
                })
                if exp == "exp05":
                    continue
                for k in remask:
                    a, b = bool(remask[k]["score"]), bool(plain[k]["score"])
                    if a == b:
                        continue
                    new, old = token_ids(tok, remask[k]["raw_completion"]), token_ids(tok, plain[k]["raw_completion"])
                    ratio = difflib.SequenceMatcher(None, new, old, autojunk=False).ratio()
                    first = ac.first_edit(tok, bench, remask[k], plain[k])
                    flips.append({
                        "exp": exp, "bench": bench, "benchLabel": BENCH_LABEL[bench], "id": k, "delta": dl,
                        "outcome": "fix" if a else "break",
                        "edits": [[tok.decode([e["old_token"]]), tok.decode([e["new_token"]]), e["canvas_index"]]
                                  for e in edits_of(remask[k])],
                        "firstKind": fine_kind(first) if first else "none",
                        "direct": ratio >= DIRECT, "similarity": round(ratio, 3),
                        "on": remask[k]["raw_completion"], "off": plain[k]["raw_completion"],
                    })
        charts.append({"bench": bench, "label": BENCH_LABEL[bench], "metric": METRIC[bench], "rows": rows,
                       "ref8b": b8[bench], "ref17": b17[bench]})
    return charts, flips


# ------------------------------------------------------------------ edit kinds
KINDS = [
    # key, label, css
    ("eol", "line ending (\\n ↔ \\r\\n)", "--k-eol"),
    ("ws", "other whitespace", "--k-ws"),
    ("word", "word / identifier", "--s1"),
    ("num", "number", "--s2"),
    ("op", "operator / punctuation", "--s3"),
    ("end", "answer ends elsewhere", "--k-end"),
]


def fine_kind(first: dict) -> str:
    base = first["kind"]
    if base == "whitespace only":
        return "eol" if "\r" in (first["old"] or "") + (first["new"] or "") else "ws"
    return {"content: number": "num", "content: word/identifier": "word",
            "content: operator/punctuation": "op"}.get(base, "end")


def taxonomy(tok) -> dict:
    rows = []
    for bench in BENCHMARKS:
        for exp, (label, roles, css, root) in EXPERIMENTS.items():
            counts = {k: 0 for k, _, _ in KINDS}
            samples: dict[str, list[str]] = {k: [] for k, _, _ in KINDS}
            pos0 = total = 0
            for delta in DELTAS:
                remask, plain = ac.arms(exp, bench, delta, root)
                for k in remask:
                    if int(remask[k].get("tokens_changed_by_remask", 0)) == 0:
                        continue
                    first = ac.first_edit(tok, bench, remask[k], plain[k])
                    if first is None:
                        continue
                    kind = fine_kind(first)
                    counts[kind] += 1
                    total += 1
                    pos0 += first["index"] == 0
                    if len(samples[kind]) < 3:
                        samples[kind].append(f"{first['old']!r} → {first['new']!r}")
            rows.append({"bench": bench, "benchLabel": BENCH_LABEL[bench], "exp": exp, "label": label,
                         "roles": roles, "total": total, "pos0": pos0, "counts": counts, "samples": samples})
    return {"kinds": [{"key": k, "label": l, "css": c} for k, l, c in KINDS], "rows": rows}


# ------------------------------------------------------------------ error kinds
ERROR_CLASSES = [
    # key, label, css, one-line definition
    ("unfinished", "ran out of canvas", "--e-unf", "cut off by the canvas before the answer or the function was complete"),
    ("syntax", "ended early / broken", "--e-syn", "code that does not parse as written, stops before its return, is a skeleton, "
     "or degenerates into repetition; a GSM8K answer that stops with no result"),
    ("interface", "wrong interface", "--e-int", "code with no function of the prompt's name, or one that rejects the prompt's arguments"),
    ("prompt", "misread the prompt", "--e-prm", "ignores or misreads a stated quantity or condition, or the prompt's own example"),
    ("careless", "careless slip", "--e-car", "a local slip in a right plan: arithmetic, a copied number, off by one, a wrong variable, "
     "a missing return branch; for code, often one line away from another arm's correct solution"),
    ("logic", "wrong plan", "--e-log", "wrong reasoning, algorithm or formula, or code that never returns"),
    ("edge", "misses edge cases", "--e-edg", "code that passes the base tests but fails EvalPlus's extra ones"),
    ("harness", "scorer artifact", "--e-har", "not the model's error: HumanEval code that calls a helper the prompt defines, "
     "correct once the helper is added"),
]
ERROR_ARMS = [
    # key in outputs/error_taxonomy.json, label, group, css, remask pair (on arm) or None
    ("b8", "SDAR-8B fixed k=1", "fixed", "--s1"),
    ("b17", "SDAR-1.7B fixed k=1", "fixed", "--s2"),
    ("cross_off_0p10", "cross8b δ=0.1", "off", "--s3"),
    ("cross_off_1p00", "cross8b δ=1", "off", "--s3"),
    ("swap_off_0p10", "swap δ=0.1", "off", "--s4"),
    ("swap_off_1p00", "swap δ=1", "off", "--s4"),
    ("cross_on_0p10", "cross8b δ=0.1", "on", "--s3"),
    ("cross_on_1p00", "cross8b δ=1", "on", "--s3"),
    ("swap_on_0p10", "swap δ=0.1", "on", "--s4"),
    ("swap_on_1p00", "swap δ=1", "on", "--s4"),
]
ERROR_PAIRS = {"cross8b": [("cross_on_0p10", "cross_off_0p10"), ("cross_on_1p00", "cross_off_1p00")],
               "swap": [("swap_on_0p10", "swap_off_0p10"), ("swap_on_1p00", "swap_off_1p00")]}


def errors(flips: list[dict]) -> dict:
    """Error classes per arm (scripts/error_taxonomy.py), and what remasking does to each.

    Also stamps each score flip with its error class with remasking off and on.
    """
    data = json.loads((PROJECT / "outputs/error_taxonomy.json").read_text())
    rows = {(r["arm"], r["bench"], r["id"]): r for r in data["rows"]}
    counts = {b: {a: {} for a, *_ in ERROR_ARMS} for b in BENCHMARKS}
    failures = []
    for (arm, bench, eid), r in rows.items():
        if r["score"] or arm not in counts[bench]:
            continue
        counts[bench][arm][r["cls"]] = counts[bench][arm].get(r["cls"], 0) + 1
        failures.append({"b": bench, "id": eid, "arm": arm, "cls": r["cls"], "why": r["why"],
                         "hand": bench == "gsm8k" or "rule" in r})
    # Paired transitions, remasking off -> on, pooled over delta. "ok" is a correct answer.
    trans = {e: {b: {} for b in BENCHMARKS} for e in ERROR_PAIRS}
    for exp, pairs in ERROR_PAIRS.items():
        for on, off in pairs:
            for bench in BENCHMARKS:
                for (arm, b, eid), r in rows.items():
                    if arm != off or b != bench:
                        continue
                    ro = rows[(on, bench, eid)]
                    src = "ok" if r["score"] else r["cls"]
                    dst = "ok" if ro["score"] else ro["cls"]
                    cell = trans[exp][bench].setdefault(src, {"to": {}, "n": 0, "edited": 0})
                    cell["n"] += 1
                    cell["to"][dst] = cell["to"].get(dst, 0) + 1
                    cell["edited"] += int(ro.get("remasks", 0) > 0)
    arm_of = {("cross8b", "0.1"): ("cross_on_0p10", "cross_off_0p10"), ("cross8b", "1"): ("cross_on_1p00", "cross_off_1p00"),
              ("swap", "0.1"): ("swap_on_0p10", "swap_off_0p10"), ("swap", "1"): ("swap_on_1p00", "swap_off_1p00")}
    for f in flips:
        on, off = arm_of[(f["exp"], f["delta"])]
        for side, arm in (("errOn", on), ("errOff", off)):
            r = rows[(arm, f["bench"], f["id"])]
            f[side] = None if r["score"] else r["cls"]
            f[side + "Why"] = None if r["score"] else r["why"]
    return {
        "classes": [{"key": k, "label": l, "css": c, "desc": d} for k, l, c, d in ERROR_CLASSES],
        "arms": [{"key": k, "label": l, "group": g, "css": c} for k, l, g, c in ERROR_ARMS],
        "counts": counts, "transitions": trans, "failures": failures,
    }


# ------------------------------------------------------------------ player
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(PROJECT.parent / "models" / "SDAR-1.7B-Chat-b64-local"),
                                         local_files_only=True)


ARM_META = {
    "sdar8b": ("SDAR-8B · fixed k=1", "the verifier alone · one token per step", "--s1"),
    "sdar1p7b": ("SDAR-1.7B · fixed k=1", "the proposer alone · one token per step", "--s2"),
    "remask_0p10": ("cross8b · δ=0.1", "1.7B proposes, 8B verifies · remasking on", "--s3"),
    "remask_1p00": ("cross8b · δ=1", "1.7B proposes, 8B verifies · remasking on", "--s3"),
    "plain_0p10": ("cross8b · δ=0.1", "1.7B proposes, 8B verifies · remasking off", "--s3"),
    "plain_1p00": ("cross8b · δ=1", "1.7B proposes, 8B verifies · remasking off", "--s3"),
}


def build_arm(arm: str, bench: str, eid: str, tok) -> dict:
    d = tj.rerun_dir(arm, bench)
    rec = tj.records(d)[eid]
    recorded = tj.records(tj.ARMS[arm][2](bench))[eid]
    if rec["raw_completion"] != recorded["raw_completion"]:
        raise SystemExit(f"{bench} {eid} {arm}: traced re-run does not reproduce the recorded answer")
    traces = [t for t in map(json.loads, (d / "decision_traces.jsonl").open()) if t["example_id"] == eid]
    fixed = arm.startswith("sdar")
    prompt_len = len(tok(rec["rendered_prompt"], add_special_tokens=False)["input_ids"])
    block = 64
    stop = rec["eos_position"]
    special = set(tok.all_special_ids)
    pool: list[list] = []
    pool_index: dict[int, int] = {}

    def piece(token_id: int) -> int:
        if token_id not in pool_index:
            pool_index[token_id] = len(pool)
            pool.append([tok.decode([token_id]), token_id in special])
        return pool_index[token_id]

    def canvas_pos(t: dict, p: int) -> int:
        # blockwise traces hold offsets inside the block; cross traces hold canvas indices
        return t["block_index"] * block + p - prompt_len if fixed else p

    committed = [canvas_pos(t, p) for t in traces if not t.get("remask") for p in t["committed_positions"]]
    end = stop if stop is not None else max(committed)
    hist: list[list] = [[] for _ in range(end + 1)]
    frames, keys, edits = [], [], []
    for t in traces:
        if not t.get("remask"):
            k = t["accepted_prefix_length"]
            ids = t["committed_token_ids"] if "committed_token_ids" in t else t["candidate_token_ids"][:k]
            shown, toks = [], []
            for p, token_id in zip(t["committed_positions"], ids):
                pos = canvas_pos(t, p)
                if pos <= end:
                    idx = piece(token_id)
                    hist[pos].append([len(frames), idx])
                    shown.append(pos)
                    toks.append(idx)
            cands = [canvas_pos(t, p) for p in t["candidate_positions"]]
            frame = {"t": "s", "s": t["step_index"], "b": t["block_index"], "c": len(cands), "k": k,
                     "p": shown, "tok": toks}
            if fixed:
                # the proposer's own ranking: committed candidate first, runners-up after
                frame["top"] = [[cands[j], tok.decode([t["candidate_token_ids"][j]]),
                                 round(t["candidate_confidences"][j], 4)] for j in range(min(6, len(cands)))]
            else:
                errs = t["error_by_prefix_nats"]
                frame["e"] = [round(e, 4) for e in errs]
                frame["n"] = ([cands[k], tok.decode([t["candidate_token_ids"][k]]),
                               round(errs[k], 4) if k < len(errs) else None] if k < len(cands) else None)
            frames.append(frame)
            keys.append([t["step_index"], 0])
            continue
        checked, lps, args = t["checked_positions"], t["log_probs"], t["argmax"]
        survivors = set(t["survivors"])
        weights = {}
        if survivors:
            zs = [(p, -lps[checked.index(p)]) for p in survivors]
            top = max(z for _, z in zs)
            total = sum(math.exp(z - top) for _, z in zs)
            weights = {p: math.exp(z - top) / total for p, z in zs}
        rows = []
        for pos, lp, arg, clp in zip(checked, lps, args, t["commit_time_log_probs"]):
            tok_idx = hist[pos][-1][1] if pos <= end and hist[pos] else None
            text = pool[tok_idx][0] if tok_idx is not None else "(past the stop)"
            rows.append([pos, text, round(lp, 5), tok.decode([arg]),
                         None if clp is None or (isinstance(clp, float) and math.isnan(clp)) else round(clp, 5),
                         pos in t["bottom"], pos in survivors, round(weights[pos], 4) if pos in weights else None])
        rows.sort(key=lambda r: r[2])
        chosen = t["chosen"]
        frame = {"t": "r", "s": t["step_index"], "b": t["block_index"], "trigger": t["trigger"],
                 "rows": rows[:TABLE_ROWS], "total": len(rows), "bottom": t["bottom"], "chosen": chosen,
                 "survivors": len(survivors), "chosenWeight": weights.get(chosen),
                 "old": tok.decode([t["old_token"]]) if chosen is not None else None,
                 "new": tok.decode([t["new_token"]]) if chosen is not None else None}
        if chosen is not None:
            edits.append([chosen, t["old_token"], t["new_token"]])
            if chosen <= end:  # an edit past the final stop changes nothing in the answer
                hist[chosen].append([len(frames), piece(t["new_token"])])
        frames.append(frame)
        keys.append([t["step_index"], 1])

    final_ids = []
    for pos in range(end + 1):
        if not hist[pos]:
            raise SystemExit(f"{bench} {eid} {arm}: position {pos} never committed before the stop")
        final_ids.append(hist[pos][-1][1])
    by_idx = {v: k for k, v in pool_index.items()}
    scored = [by_idx[i] for i in (final_ids[:stop] if stop is not None else final_ids)]
    if tok.decode(scored, skip_special_tokens=True) != rec["raw_completion"]:
        raise SystemExit(f"{bench} {eid} {arm}: reconstructed canvas does not decode to the recorded answer")
    title, subtitle, css = ARM_META[arm]
    steps = sum(1 for f in frames if f["t"] == "s")
    if steps != recorded["denoising_steps"]:
        raise SystemExit(f"{bench} {eid} {arm}: {steps} traced steps, recorded {recorded['denoising_steps']}")
    return {
        "key": arm, "title": title, "subtitle": subtitle, "css": css, "fixed": fixed,
        "correct": bool(recorded["score"]), "failure": recorded.get("failure_type"),
        "stop": stop,
        "stopToken": tok.convert_ids_to_tokens(rec["stop_token_id"]) if rec["stop_token_id"] is not None else None,
        "steps": steps, "remaskEvents": sum(1 for f in frames if f["t"] == "r"), "edits": len(edits),
        "outTokens": rec["output_token_count"], "completion": rec["raw_completion"],
        "block": block, "promptLen": prompt_len, "pool": pool, "hist": hist, "frames": frames, "keys": keys,
        "rmMarks": [e[0] for e in edits], "finalIds": final_ids, "_edits": edits,
    }


VGB_META = {
    "vgb_fwd": ("VGB · forward only", "8B via the 1.7B shortlist, T=1 · no remasking"),
    "vgb_lam0p5": ("VGB · λ=0.5", "8B via the 1.7B shortlist, T=1 · remask ∝ ρ^0.5, momentum"),
    "vgb_lam1": ("VGB · λ=1", "8B via the 1.7B shortlist, T=1 · remask ∝ ρ, momentum"),
}


def build_vgb_arm(arm: str, bench: str, eid: str, tok) -> dict:
    """One VGB Track A arm from its traced re-run: frames are the moves that change the canvas.

    A forward move unmasks one position; a backward move remasks one (its hist
    entry is -1); stays and switches change nothing, so they are counted onto
    the next move. Forced fills (budget) ride on the block's last trace row.
    """
    d = tj.rerun_dir(arm, bench)
    rec = tj.records(d)[eid]
    recorded = tj.records(tj.ARMS[arm][2](bench))[eid]
    if rec["raw_completion"] != recorded["raw_completion"]:
        raise SystemExit(f"{bench} {eid} {arm}: traced re-run does not reproduce Track A's answer")
    traces = [t for t in map(json.loads, (d / "decision_traces.jsonl").open()) if t["example_id"] == eid]
    prompt_len = len(tok(rec["rendered_prompt"], add_special_tokens=False)["input_ids"])
    block = 64
    stop = rec["eos_position"]
    special = set(tok.all_special_ids)
    pool: list[list] = []
    pool_index: dict[int, int] = {}

    def piece(token_id: int) -> int:
        if token_id not in pool_index:
            pool_index[token_id] = len(pool)
            pool.append([tok.decode([token_id]), token_id in special])
        return pool_index[token_id]

    def canvas(t: dict, p: int) -> int:
        return t["block_index"] * block + p - prompt_len

    unmasked = [canvas(t, t["position"]) for t in traces if t["move"] == "forward"]
    unmasked += [canvas(t, p) for t in traces for p, _ in t.get("forced", [])]
    end = stop if stop is not None else max(unmasked)
    hist: list[list] = [[] for _ in range(end + 1)]
    frames, keys = [], []
    pending = {"stays": 0, "switches": 0}
    stats = {"forward": 0, "remasks": 0, "stays": 0, "switches": 0, "forced": 0}

    def p_forward(t: dict) -> float:
        lf, lb = t["log_forward"], t["log_backward"]
        if lb is None or lb == -math.inf:
            return 1.0
        if lf == -math.inf:
            return 0.0
        return 1.0 / (1.0 + math.exp(lb - lf))

    def add(frame: dict) -> None:
        frame.update(stays=pending["stays"], switches=pending["switches"])
        pending.update(stays=0, switches=0)
        keys.append([len(frames), 0])
        frames.append(frame)

    for t in traces:
        move = t["move"]
        if move in ("stay", "switch"):
            counter = "stays" if move == "stay" else "switches"
            pending[counter] += 1
            stats[counter] += 1
            continue
        pos = canvas(t, t["position"])
        base = {"b": t["block_index"], "k": t["k"], "dir": t["direction"], "pf": round(p_forward(t), 4)}
        if move == "forward":
            stats["forward"] += 1
            shown = pos <= end
            if shown:
                hist[pos].append([len(frames), piece(t["token"])])
            add({"t": "s", "p": [pos] if shown else [], "tok": [pool_index[t["token"]]] if shown else [],
                 "pos": pos, "token": tok.decode([t["token"]]), **base,
                 "pb": round(t["log_big"], 4), "ps": None if t.get("log_small") is None else round(t["log_small"], 4),
                 "cands": t.get("candidates"),
                 "share": None if t.get("pick_share") is None else round(t["pick_share"], 4),
                 "top": [[canvas(t, p), tok.decode([a]), round(lb, 4), round(ls, 4), c, round(s, 4)]
                         for p, a, lb, ls, c, s in t.get("top", [])]})
        else:
            stats["remasks"] += 1
            if pos <= end:
                hist[pos].append([len(frames), -1])
            add({"t": "b", "p": [pos] if pos <= end else [], "pos": pos, "token": tok.decode([t["token"]]), **base,
                 "rho": round(t["log_rho"], 4), "share": round(t["pick_share"], 4), "parents": t["parents"],
                 "top": [[canvas(t, p), tok.decode([a]), round(lr, 4), round(s, 4)] for p, a, lr, s in t["top"]]})
        for p, a in t.get("forced", []):
            stats["forced"] += 1
            fpos = canvas(t, p)
            if fpos <= end:
                hist[fpos].append([len(frames), piece(a)])
            add({"t": "s", "p": [fpos] if fpos <= end else [], "tok": [pool_index[a]] if fpos <= end else [],
                 "pos": fpos, "token": tok.decode([a]), "b": t["block_index"], "forced": True})

    final_ids = []
    for pos in range(end + 1):
        if not hist[pos] or hist[pos][-1][1] == -1:
            raise SystemExit(f"{bench} {eid} {arm}: position {pos} is not filled at the end")
        final_ids.append(hist[pos][-1][1])
    by_idx = {v: k for k, v in pool_index.items()}
    scored = [by_idx[i] for i in (final_ids[:stop] if stop is not None else final_ids)]
    if tok.decode(scored, skip_special_tokens=True) != rec["raw_completion"]:
        raise SystemExit(f"{bench} {eid} {arm}: reconstructed canvas does not decode to the recorded answer")
    totals = rec["vgb"] if isinstance(rec["vgb"], dict) else ast.literal_eval(rec["vgb"])
    if (stats["forward"], stats["remasks"]) != (totals["forward_moves"], totals["backward_moves"]):
        raise SystemExit(f"{bench} {eid} {arm}: traced moves {stats} disagree with the record {totals}")
    # positions the chain remasked and then refilled, and how many ended on a different token
    refilled = sum(1 for h in hist if any(i == -1 for _, i in h))
    changed = sum(1 for h in hist if any(i == -1 for _, i in h)
                  and [i for _, i in h if i != -1][0] != h[-1][1])
    title, subtitle = VGB_META[arm]
    return {
        "key": arm, "title": title, "subtitle": subtitle, "css": "--ctx", "fixed": False, "vgb": True,
        "correct": bool(recorded["score"]), "failure": recorded.get("failure_type"),
        "stop": stop,
        "stopToken": tok.convert_ids_to_tokens(rec["stop_token_id"]) if rec["stop_token_id"] is not None else None,
        "steps": len(frames), "moves": stats, "refilled": refilled, "changed": changed,
        "remaskEvents": 0, "edits": stats["remasks"],
        "outTokens": rec["output_token_count"], "completion": rec["raw_completion"],
        "wall": round(recorded["wall_time_seconds"]),
        "block": block, "promptLen": prompt_len, "pool": pool, "hist": hist, "frames": frames, "keys": keys,
        "rmMarks": [], "xMarks": [], "divergence": None,
    }


def build_examples(tok) -> list[dict]:
    out = []
    for category, bench, eid, short, note in tj.SELECTION:
        arms = {a: (build_vgb_arm if a in tj.VGB_ARMS else build_arm)(a, bench, eid, tok) for a in tj.ARMS}
        for delta in ("0p10", "1p00"):
            on, off = arms[f"remask_{delta}"], arms[f"plain_{delta}"]
            on_text = [on["pool"][i][0] for i in on["finalIds"]]
            off_text = [off["pool"][i][0] for i in off["finalIds"]]
            divergence = next((i for i, (a, b) in enumerate(zip(off_text, on_text)) if a != b),
                              None if len(off_text) == len(on_text) else min(len(off_text), len(on_text)))
            on["divergence"] = off["divergence"] = divergence
            # tokens remasking replaced, marked where the run without it still has the old token
            off["xMarks"] = [pos for pos, old, _ in on["_edits"]
                             if pos < len(off["hist"]) and off["hist"][pos]
                             and off["pool"][off["hist"][pos][-1][1]][0] == tok.decode([old])]
            on["xMarks"] = []
        for key, arm in arms.items():
            arm.setdefault("divergence", None)
            arm.setdefault("xMarks", [])
            if key not in tj.VGB_ARMS:
                del arm["finalIds"], arm["_edits"]
        prompt = tj.records(tj.ARMS["sdar8b"][2](bench))[eid]["raw_prompt"]
        gold = None
        if bench == "gsm8k":
            gold = next(json.loads(l)["target"] for l in (tj.DATA / bench / "test.jsonl").open()
                        if json.loads(l)["example_id"] == eid)
        out.append({
            "category": category, "categoryLabel": tj.CATEGORY_LABEL[category], "bench": bench,
            "benchLabel": BENCH_LABEL[bench], "id": eid, "short": short, "note": note, "prompt": prompt,
            "gold": gold, "tau": {"0.1": math.log1p(0.1), "1": math.log1p(1.0)},
            "arms": [arms[a] for a in ("sdar8b", "sdar1p7b", "remask_0p10", "remask_1p00", "plain_0p10", "plain_1p00",
                                       *tj.VGB_ARMS)],
        })
        print(f"{bench:15s} {eid:16s} " + "  ".join(
            f"{a['key']}:{a['steps']}{'mv' if a.get('vgb') else 'st'}/{a['edits']}ed{'✓' if a['correct'] else '✗'}"
            for a in out[-1]["arms"]))
    return out


# ------------------------------------------------------------------ page
def main() -> None:
    if not tj.verify():
        raise SystemExit("a traced re-run no longer matches its recorded run; not building")
    tok = tokenizer()
    fr = frontiers()
    charts, flips = impact(tok)
    errs = errors(flips)
    tax = taxonomy(tok)
    examples = build_examples(tok)

    def pooled(exp: str) -> tuple[int, int]:
        rows = [r for c in charts for r in c["rows"] if r["exp"] == exp]
        return sum(r["fixed"] for r in rows), sum(r["broke"] for r in rows)

    (f5, b5), (f8, b8), (fs, bs) = pooled("exp05"), pooled("cross8b"), pooled("swap")
    direct = {(e, o): sum(f["direct"] for f in flips if f["exp"] == e and f["outcome"] == o)
              for e in ("cross8b", "swap") for o in ("fix", "break")}
    payload = {
        "pooled": {"exp05": [f5, b5], "cross8b": [f8, b8], "swap": [fs, bs]},
        "direct": {f"{e}:{o}": v for (e, o), v in direct.items()},
        "directThreshold": DIRECT,
        "frontiers": fr, "impact": charts, "flips": flips, "taxonomy": tax, "errors": errs, "examples": examples,
    }
    # A token that is part of a multi-byte character decodes alone to U+FFFD (the T=1 VGB
    # shortlist samples such fragments). Show it as a visible box; the legend says what it is.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    data = data.replace("\ufffd", "\u25af")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"pooled fix/break: exp05 {f5}/{b5}, cross8b {f8}/{b8}, swap {fs}/{bs}; direct {direct}")
    print(f"wrote {OUTPUT} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
