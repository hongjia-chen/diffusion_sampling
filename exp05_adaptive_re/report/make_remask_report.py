#!/usr/bin/env python3
"""Remasking diagnosis page: report/remask_report.html.

Three parts, all read from finished runs:

1. Accuracy against tokens/s per benchmark: Dream-7B fixed:k (exp04), SDAR
   fixed:k (exp05 B1), and the SDAR -> Dream remask sweep (outputs/cross).
2. The paired remask ablation: every task at every delta with remasking on
   (outputs/cross) and off (report/data/no_remask, re-run by
   remask_diagnosis.py; tasks remasking never edited are identical by
   construction and reused).
3. A two-panel player per example in report/examples.json, built from traced
   re-runs (report/data/traced) that must reproduce the recorded answers.

The page also reports exp05's paired results for the four exp04 trajectory
tasks in each of the three shared benchmarks.

    PYTHONPATH=vendor_env/tf452:. python report/make_remask_report.py
"""
from __future__ import annotations

import csv
import json
import math
from math import comb
from pathlib import Path

import remask_diagnosis as rd

PROJECT = rd.PROJECT
REPORT = PROJECT / "report"
TEMPLATE = REPORT / "remask_report.template.html"
OUTPUT = REPORT / "remask_report.html"
EXP04_SUMMARY = PROJECT.parent / "exp04_autospeculation" / "outputs" / "final" / "summary.csv"
B1_SUMMARY = PROJECT / "outputs" / "baseline" / "sdar" / "comparison" / "summary.json"
CROSS_SUMMARY = PROJECT / "outputs" / "cross" / "comparison" / "summary.json"
EXP04_PLAYER = PROJECT.parent / "exp04_autospeculation" / "report" / "trajectory_player.html"

BENCH_LABEL = {"gsm8k": "GSM8K", "humaneval_plus": "HumanEval+", "mbpp_plus": "MBPP+"}
METRIC = {"gsm8k": "exact match", "humaneval_plus": "pass@1", "mbpp_plus": "pass@1"}
DELTA_LABEL = {"0p10": "0.1", "0p30": "0.3", "1p00": "1", "3p00": "3", "10p00": "10"}
STOPS = {151643, 151645}
TABLE_ROWS = 12  # remask inspector: this many least-probable re-scored tokens


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else list(data.values())


def exp04_results() -> list[dict]:
    """Exp05 outcomes for the four exp04 trajectory tasks in each benchmark."""
    html = EXP04_PLAYER.read_text(encoding="utf-8")
    marker, end_marker = "const DATA = ", ";\nconst TAU_LOW"
    start = html.index(marker) + len(marker)
    end = html.index(end_marker, start)
    selected = [task for task in json.loads(html[start:end])["tasks"] if task["bench"] in BENCH_LABEL]

    out = []
    for task in selected:
        runs = []
        for delta in rd.DELTAS:
            pair = next(p for p in rd.paired(task["bench"], delta) if p["id"] == task["id"])
            on, off = pair["with"], pair["without"]
            if off is None:
                raise SystemExit(f"{task['bench']} {delta} {task['id']}: no-remask result is missing")
            runs.append({
                "delta": DELTA_LABEL[delta],
                "onScore": bool(on["score"]), "offScore": bool(off["score"]),
                "onFailure": on.get("failure_type"), "offFailure": off.get("failure_type"),
                "onCompletion": on["raw_completion"], "offCompletion": off["raw_completion"],
                "onTokens": on["output_token_count"], "offTokens": off["output_token_count"],
                "edits": on["tokens_changed_by_remask"],
            })
        out.append({
            "bench": task["bench"], "benchLabel": task["benchLabel"], "id": task["id"],
            "categoryLabel": task["categoryLabel"], "note": task["note"],
            "prompt": task["prompt"], "runs": runs,
        })
    counts = {bench: sum(t["bench"] == bench for t in out) for bench in BENCH_LABEL}
    if any(n != 4 for n in counts.values()):
        raise SystemExit(f"expected four exp04 tasks per exp05 benchmark, found {counts}")
    return out


# ------------------------------------------------------------------ frontiers
def frontiers() -> list[dict]:
    dream = [r for r in csv.DictReader(EXP04_SUMMARY.open()) if r["group"] == "1_dream7b_baseline"]
    b1, cross = load_list(B1_SUMMARY), load_list(CROSS_SUMMARY)
    out = []
    for bench in rd.BENCHMARKS:
        def tip(score, tps, sec, setting):
            return [["setting", setting], [METRIC[bench], f"{score:.2f}"], ["tokens/s", f"{tps:.1f}"], ["s/task", f"{sec:.1f}"]]

        series = []
        pts = []
        for r in (r for r in dream if r["benchmark"] == bench):
            k = r["method"].rsplit("_", 1)[1]
            y, x, sec = float(r["score"]), float(r["tokens_per_sec"]), float(r["sec_per_task"])
            pts.append({"x": x, "y": y, "sec": sec, "method": f"fixed k={k}",
                        "label": f"k={k}" if k in ("1", "8") else None, "labelAt": [-7, 3, "end"],
                        "tip": tip(y, x, sec, f"fixed k={k} (exp04 recipe)")})
        series.append({"key": "dream", "name": "Dream-7B fixed:k (exp04)", "css": "--s1", "shape": "circle", "line": True, "points": pts})
        pts = []
        for r in (r for r in b1 if r["benchmark"] == bench):
            k = r["method"].rsplit("_", 1)[1]
            y, x, sec = r["primary_score"], r["tokens_per_second"], r["seconds_per_task"]
            pts.append({"x": x, "y": y, "sec": sec, "method": f"fixed k={k}",
                        "label": f"k={k}" if k in ("1", "8") else None, "labelAt": [0, 14, "middle"],
                        "tip": tip(y, x, sec, f"fixed k={k}")})
        series.append({"key": "sdar", "name": "SDAR-1.7B fixed:k", "css": "--s2", "shape": "square", "line": True, "dash": True, "points": pts})
        pts = []
        for r in (r for r in cross if r["benchmark"] == bench):
            d = DELTA_LABEL[r["method"].rsplit("_", 1)[1]]
            y, x, sec = r["primary_score"], r["tokens_per_second"], r["seconds_per_task"]
            # the sweep's points sit within ~1 tok/s of each other: label its ends only
            pts.append({"x": x, "y": y, "sec": sec, "method": f"δ={d}",
                        "label": f"δ={d}" if d in ("0.1", "10") else None, "labelAt": [7, 3, "start"],
                        "tip": tip(y, x, sec, f"δ={d}, remasking on, no veto")})
        series.append({"key": "cross", "name": "SDAR → Dream, remasking", "css": "--s3", "shape": "triangle", "line": False, "points": pts})
        out.append({"bench": bench, "label": BENCH_LABEL[bench], "metric": METRIC[bench], "series": series})
    return out


# ------------------------------------------------------------------ impact
def impact() -> list[dict]:
    out = []
    for bench in rd.BENCHMARKS:
        rows, with_pts, without_pts = [], [], []
        for delta in rd.DELTAS:
            pairs = rd.paired(bench, delta)
            if any(p["without"] is None for p in pairs):
                raise SystemExit(f"{bench} {delta}: counterfactual incomplete; run remask_diagnosis.py verify")
            w = [bool(p["with"]["score"]) for p in pairs]
            wo = [bool(p["without"]["score"]) for p in pairs]
            fixed = sum(a and not b for a, b in zip(w, wo))
            broke = sum(b and not a for a, b in zip(w, wo))
            n = len(pairs)
            row = {
                "delta": DELTA_LABEL[delta], "with": sum(w) / n, "without": sum(wo) / n,
                "fixed": fixed, "broke": broke, "p": mcnemar(fixed, broke),
                "changed": sum(p["with"]["raw_completion"] != p["without"]["raw_completion"] for p in pairs),
                "editsPerTask": sum(p["with"]["tokens_changed_by_remask"] for p in pairs) / n,
                "tokWith": sum(p["with"]["output_token_count"] for p in pairs) / n,
                "tokWithout": sum(p["without"]["output_token_count"] for p in pairs) / n,
                "synWith": sum(p["with"].get("failure_type") == "syntax_error" for p in pairs),
                "synWithout": sum(p["without"].get("failure_type") == "syntax_error" for p in pairs),
            }
            rows.append(row)
            tip_rows = [["with / without", f"{row['with']:.2f} / {row['without']:.2f}"],
                        ["fixed / broke", f"{fixed} / {broke}"], ["McNemar p", f"{row['p']:.3f}"]]
            with_pts.append({"x": row["delta"], "y": row["with"], "method": f"δ={row['delta']}", "sec": None,
                             "label": None, "tip": [["δ", row["delta"]], [METRIC[bench], f"{row['with']:.2f}"]] + tip_rows})
            without_pts.append({"x": row["delta"], "y": row["without"], "method": f"δ={row['delta']}", "sec": None,
                                "label": None, "tip": [["δ", row["delta"]], [METRIC[bench], f"{row['without']:.2f}"]] + tip_rows})
        # end labels, pushed apart vertically so equal values cannot collide
        upper = with_pts[-1]["y"] >= without_pts[-1]["y"]
        with_pts[-1]["label"], with_pts[-1]["labelAt"] = "on", [8, -5 if upper else 11, "start"]
        without_pts[-1]["label"], without_pts[-1]["labelAt"] = "off", [8, 11 if upper else -5, "start"]
        out.append({
            "bench": bench, "label": BENCH_LABEL[bench], "metric": METRIC[bench],
            "categories": [DELTA_LABEL[d] for d in rd.DELTAS], "rows": rows,
            "series": [
                {"key": "on", "name": "remasking on", "css": "--s3", "shape": "triangle", "line": True, "points": with_pts},
                {"key": "off", "name": "remasking off", "css": "--s4", "shape": "diamond", "line": True, "dash": True, "points": without_pts},
            ],
        })
    return out


# ------------------------------------------------------------------ edit taxonomy
def taxonomy(tok) -> dict:
    """Every remask edit in the sweep's traced tasks (the first 5 per arm)."""
    import collections
    kinds = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)
    passes = skipped = at_zero = 0
    tasks = set()
    for path in sorted((PROJECT / "outputs" / "cross").glob("*/xadaptive*/decision_traces.jsonl")):
        for t in map(json.loads, path.open()):
            if not t.get("remask"):
                continue
            tasks.add((path.parts[-3], path.parts[-2], t["example_id"]))
            passes += 1
            if t["chosen"] is None:
                skipped += 1
                continue
            old_id, new_id = t["old_token"], t["new_token"]
            old_s, new_s = tok.decode([old_id]), tok.decode([new_id])
            if old_id in STOPS and new_id in STOPS:
                kind = "stop token → the other stop token"
            elif old_id in STOPS:
                kind = "stop → content (the answer keeps going)"
            elif new_id in STOPS:
                kind = "content → stop (the answer ends here)"
            elif old_s.strip() == new_s.strip() and old_s != new_s:
                kind = "same text, different whitespace"
            else:
                kind = "different content"
            kinds[kind] += 1
            at_zero += t["chosen"] == 0
            if len(samples[kind]) < 3:
                samples[kind].append(f"{old_s!r} → {new_s!r} @{t['chosen']}")
    edits = sum(kinds.values())
    css = {"different content": "--s3", "same text, different whitespace": "--s4",
           "stop → content (the answer keeps going)": "--s1", "content → stop (the answer ends here)": "--s2",
           "stop token → the other stop token": "--rm"}
    rows = [{"kind": k, "count": c, "share": c / edits, "css": css.get(k, "--rm"),
             "examples": "   ".join(samples[k])} for k, c in kinds.most_common()]
    return {
        "rows": rows, "edits": edits, "tasks": len(tasks),
        "lede": (
            f"Every remask edit in the tasks the sweep traced ({len(tasks)} tasks, {passes} passes). "
            f"<b>{skipped} of {passes} passes ({skipped / passes:.0%}) make no edit at all</b> — none of the four "
            "least probable tokens is one Dream would change."
        ),
        "note": (
            f"{at_zero} of {edits} edits ({at_zero / edits:.0%}) land on the answer's very first token, where SDAR "
            "writes a leading space (\u2019 The\u2019) and Dream prefers the form without it. That edit changes no "
            "content, but every later block is conditioned on it, so the rest of the answer can be re-routed: in "
            "GSM8K gsm8k-test-278 below it turns a 256-token runaway into 52 tokens."
        ),
    }


# ------------------------------------------------------------------ examples
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(PROJECT.parent / "models" / "SDAR-1.7B-Chat-b64-local"),
                                         trust_remote_code=True, local_files_only=True)


def traced(arm: str, bench: str, delta: str, eid: str) -> tuple[dict, list[dict]]:
    label = ("xadaptive_remask_noveto_delta_" if arm == "remask" else "xadaptive_noveto_delta_") + delta
    d = rd.TRACED / arm / bench / label
    rec = next(r for r in map(json.loads, (d / "records.jsonl").open()) if r["example_id"] == eid)
    traces = [t for t in map(json.loads, (d / "decision_traces.jsonl").open()) if t["example_id"] == eid]
    return rec, traces


def build_arm(arm: str, bench: str, delta: str, eid: str, recorded: dict, tok) -> dict:
    rec, traces = traced(arm, bench, delta, eid)
    if rec["raw_completion"] != recorded["raw_completion"]:
        raise SystemExit(f"{bench} {delta} {eid} {arm}: traced re-run does not reproduce the recorded answer")
    stop = rec["eos_position"]
    special = set(tok.all_special_ids)
    pool: list[list] = []
    pool_index: dict[int, int] = {}

    def piece(token_id: int) -> int:
        if token_id not in pool_index:
            pool_index[token_id] = len(pool)
            pool.append([tok.decode([token_id]), token_id in special])
        return pool_index[token_id]

    # last position shown: the stop token, or the last committed position
    committed = [p for t in traces if not t.get("remask") for p in t["committed_positions"]]
    end = stop if stop is not None else max(committed)
    hist: list[list] = [[] for _ in range(end + 1)]
    frames, keys = [], []
    edits = []
    for t in traces:
        if not t.get("remask"):
            shown, toks = [], []
            for pos, token_id in zip(t["committed_positions"], t["committed_token_ids"]):
                if pos <= end:
                    idx = piece(token_id)
                    hist[pos].append([len(frames), idx])
                    shown.append(pos)
                    toks.append(idx)
            k, cands = t["accepted_prefix_length"], t["candidate_positions"]
            errs = t["error_by_prefix_nats"]
            nxt = None
            if k < len(cands):
                nxt = [cands[k], tok.decode([t["candidate_token_ids"][k]]), round(errs[k], 4) if k < len(errs) else None]
            frames.append({"t": "s", "s": t["step_index"], "b": t["block_index"], "c": len(cands), "k": k,
                           "p": shown, "tok": toks, "e": [round(e, 4) for e in errs], "n": nxt})
            keys.append((t["step_index"], 0))
            continue
        checked, lps, args = t["checked_positions"], t["log_probs"], t["argmax"]
        commit_lp = t["commit_time_log_probs"]
        survivors = set(t["survivors"])
        weights = {}
        if survivors:
            zs = [(p, -lps[checked.index(p)]) for p in survivors]
            top = max(z for _, z in zs)
            total = sum(math.exp(z - top) for _, z in zs)
            weights = {p: math.exp(z - top) / total for p, z in zs}
        rows = []
        for pos, lp, arg, clp in zip(checked, lps, args, commit_lp):
            # the committed token at this position, as of this pass
            tok_id = hist[pos][-1][1] if pos <= end and hist[pos] else None
            token_text = pool[tok_id][0] if tok_id is not None else "(past the stop)"
            rows.append([pos, token_text, round(lp, 5), tok.decode([arg]),
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
            edits.append(chosen)
            # an edit past the final stop changes nothing in the answer; the
            # inspector still lists it, the canvas does not show it
            if chosen <= end:
                hist[chosen].append([len(frames), piece(t["new_token"])])
        frames.append(frame)
        keys.append((t["step_index"], 1))

    final_ids = []
    for pos in range(end + 1):
        if not hist[pos]:
            raise SystemExit(f"{eid} {arm}: position {pos} never committed before the stop")
        final_ids.append(hist[pos][-1][1])
    token_ids = {v: k for k, v in pool_index.items()}
    scored_ids = [token_ids[i] for i in (final_ids[:stop] if stop is not None else final_ids)]
    decoded = tok.decode(scored_ids, skip_special_tokens=True)
    if decoded != rec["raw_completion"]:
        raise SystemExit(f"{bench} {eid} {arm}: reconstructed canvas does not decode to the recorded answer")
    prompt_len = len(tok(rec["rendered_prompt"], add_special_tokens=False)["input_ids"])
    return {
        "key": arm, "title": "Remasking on" if arm == "remask" else "Remasking off",
        "subtitle": f"SDAR proposes · Dream verifies · δ={DELTA_LABEL[delta]}",
        "css": "--s3" if arm == "remask" else "--s4",
        "correct": bool(recorded.get("score")), "failure": recorded.get("failure_type"),
        "stop": stop, "stopToken": tok.convert_ids_to_tokens(rec["stop_token_id"]) if rec["stop_token_id"] is not None else None,
        "steps": sum(1 for f in frames if f["t"] == "s"),
        "remaskEvents": sum(1 for f in frames if f["t"] == "r"), "edits": len(edits),
        "outTokens": rec["output_token_count"], "completion": rec["raw_completion"],
        "block": 64, "promptLen": prompt_len,
        "pool": pool, "hist": hist, "frames": frames, "keys": keys,
        "rmMarks": edits if arm == "remask" else [], "finalIds": final_ids,
    }


def build_examples(tok) -> list[dict]:
    chosen = json.loads(rd.EXAMPLES.read_text())
    out = []
    for ex in chosen:
        bench, delta, eid = ex["bench"], ex["delta"], ex["id"]
        pair = next(p for p in rd.paired(bench, delta) if p["id"] == eid)
        off = build_arm("no_remask", bench, delta, eid, pair["without"], tok)
        on = build_arm("remask", bench, delta, eid, pair["with"], tok)
        # tokens remasking replaced, marked on the run without it too
        off["xMarks"] = [p for p in on["rmMarks"] if p < len(off["hist"])]
        on["xMarks"] = []
        # first position where the final canvases differ
        off_ids = [off["pool"][i][0] for i in off["finalIds"]]
        on_ids = [on["pool"][i][0] for i in on["finalIds"]]
        divergence = next((i for i, (a, b) in enumerate(zip(off_ids, on_ids)) if a != b),
                          None if len(off_ids) == len(on_ids) else min(len(off_ids), len(on_ids)))
        keys = sorted(set(off["keys"]) | set(on["keys"]))
        labels = ["start"] + [f"step {s + 1}" if sub == 0 else f"remask pass after step {s + 1}" for s, sub in keys]
        for arm in (off, on):
            applied = [0]
            for key in keys:
                applied.append(sum(1 for k in arm["keys"] if k <= key))
            arm["applied"] = applied
            del arm["keys"], arm["finalIds"]
        remask_g = [g for g, key in enumerate(keys, start=1) if key[1] == 1]
        first_edit = next((g for g, key in enumerate(keys, start=1)
                           if key[1] == 1 and any(f["t"] == "r" and f["chosen"] is not None and (f["s"], 1) == key
                                                  for f in on["frames"])), None)
        prompt = rd.dataset_rows(bench)[eid]["prompt"]
        out.append({
            "category": ex["category"], "categoryLabel": ex["categoryLabel"], "bench": bench,
            "benchLabel": BENCH_LABEL[bench], "id": eid, "delta": DELTA_LABEL[delta],
            "tau": math.log1p(float(DELTA_LABEL[delta])), "short": ex["short"], "note": ex["note"],
            "prompt": prompt, "arms": [off, on], "G": len(keys), "gLabels": labels,
            "remaskG": remask_g, "firstEditG": first_edit, "divergence": divergence,
        })
        print(f"{bench:15s} {eid:18s} δ={DELTA_LABEL[delta]:4s} off:{off['steps']}st on:{on['steps']}st "
              f"+{on['remaskEvents']} passes/{on['edits']} edits · divergence at {divergence}")
    return out


def main() -> None:
    tok = tokenizer()
    fr = frontiers()
    im = impact()
    tax = taxonomy(tok)
    examples = build_examples(tok)
    lede = (
        "SDAR-1.7B proposes and Dream-7B verifies with the Dream-only ratio (no veto). After each block, and "
        "at the stop token, Dream re-scores every committed token with that one token masked. Among the "
        "4 least probable, it replaces one it would change. This page asks what that <b>remasking</b> pass "
        "buys: first against the baselines, then paired, task by task, against the same runs with remasking off."
    )
    frontier_caveat = (
        "Dream-7B points are exp04's runs, unchanged. They use the Dream/ADJUST recipe (T=0.1, top-p 0.9) and "
        "decode the full canvas without early stop, with MBPP+ at canvas 512. So their scores are comparable, "
        "but their tokens/s is not strictly comparable with the early-stopped SDAR and cross arms. SDAR fixed:k "
        "and the cross arms share the exp05 rules: early stop, the {<|im_end|>, <|endoftext|>} stop set and IDs "
        "below 151665. The cross arms ran with each model on its own card; the SDAR arms ran one process per card."
    )
    rows = [r for group in im for r in group["rows"]]
    fixed = sum(r["fixed"] for r in rows)
    broke = sum(r["broke"] for r in rows)
    deltas = [r["with"] - r["without"] for r in rows]
    better = sum(d > 0 for d in deltas)
    syn_off = sum(r["synWithout"] for r in rows)
    syn_on = sum(r["synWith"] for r in rows)
    impact_lede = (
        "Same task, same δ, remasking on against off. Tasks remasking never edited decode identically by "
        "construction; the re-run confirmed that on unchanged control tasks. <b>fixed</b> counts tasks right "
        "only with remasking, <b>broke</b> tasks right only without it, and p is an exact McNemar test on "
        "those two counts.<br>"
        f"Over the {len(rows)} arms: <b>{better} of {len(rows)} improve</b>, mean "
        f"<b>{sum(deltas) / len(deltas):+.3f}</b> (range {min(deltas):+.2f} to {max(deltas):+.2f}); "
        f"<b>{fixed}</b> task runs fixed against <b>{broke}</b> broken, out of {len(rows) * 100}. No single arm "
        "reaches p &lt; 0.05 on its own. Pooling the five δ values is not a valid test — they share the same "
        "tasks — but the direction is consistent. Remasking also removes syntax errors on the code benchmarks "
        f"rather than causing them: <b>{syn_off} → {syn_on}</b> across the ten code arms, at "
        f"<b>{sum(r['editsPerTask'] for r in rows) / len(rows):.2f}</b> edits per task."
    )
    payload = {"lede": lede, "frontierCaveat": frontier_caveat, "impactLede": impact_lede,
               "frontiers": fr, "impact": im, "taxonomy": tax,
               "exp04Results": exp04_results(), "examples": examples}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
