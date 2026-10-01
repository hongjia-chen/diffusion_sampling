"""Style or logic? The remask edits of exp05 (Dream-7B verifier) vs cross8b (SDAR-8B).

cross8b_swap (DESIGN.md §10a) is the role-swapped ablation: SDAR-8B proposes,
SDAR-1.7B verifies and remasks.

DESIGN.md §10. For each benchmark and delta, pairs every task's remask run with
its no-remask counterfactual (same delta) and reports:

* how often the remask pass edits at all, and the fix/break table;
* the FIRST edit of every edited run, classified the same way for both
  experiments: decoding is deterministic, so the first token where the two
  completions differ is the first remask edit. exp05 records keep no edit log,
  so this is the only like-for-like view;
* for cross8b, every edit from its `remask_edits` log, as a cross-check.

    PYTHONPATH=. python scripts/analyze_cross8b.py [--cross8b outputs/cross8b]
        [--swap outputs/cross8b_swap] [--json out.json]
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXP05 = PROJECT.parent / "exp05_adaptive_re"
TOKENIZER = PROJECT.parent / "models" / "SDAR-1.7B-Chat-b64-local"
BENCHMARKS = ("gsm8k", "mbpp_plus", "humaneval_plus")
DELTAS = ("0p10", "1p00")
STOPS = {151643, 151645}
CODE = {"mbpp_plus", "humaneval_plus"}


def load(path: Path) -> dict[str, dict]:
    scored = path.with_name("records.scored.jsonl")
    path = scored if scored.exists() else path
    if not path.exists():
        return {}
    return {row["example_id"]: row for row in map(json.loads, path.open())}


def arms(experiment: str, bench: str, delta: str, root: Path) -> tuple[dict, dict]:
    if experiment == "exp05":
        remask = load(EXP05 / "outputs/cross" / bench / f"xadaptive_remask_noveto_delta_{delta}" / "records.jsonl")
        # exp05 re-ran only the edited tasks; an unedited task decodes identically.
        edited = load(EXP05 / "report/data/no_remask/out" / bench / f"xadaptive_noveto_delta_{delta}" / "records.jsonl")
        plain = {k: edited.get(k, row) for k, row in remask.items()}
    else:
        remask = load(root / bench / f"xadaptive_remask_noveto_delta_{delta}" / "records.jsonl")
        plain = load(root / bench / f"xadaptive_noveto_delta_{delta}" / "records.jsonl")
    shared = [k for k in remask if k in plain]
    return {k: remask[k] for k in shared}, {k: plain[k] for k in shared}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def in_prose(text: str, offset: int, parsed: str | None) -> bool:
    """Code benchmarks: does the character at `offset` sit outside the code?"""
    if "```" in text:
        return text[:offset].count("```") % 2 == 0
    line = text[text.rfind("\n", 0, offset) + 1 : text.find("\n", offset) if "\n" in text[offset:] else None]
    return bool(line.strip()) and parsed is not None and line.strip() not in parsed


def kind(old: str | None, new: str | None, old_id: int | None, new_id: int | None) -> str:
    if old_id in STOPS or new_id in STOPS or old is None or new is None:
        if old_id in STOPS and new_id in STOPS:
            return "stop swap"
        return "answer end moved"
    if old.strip() == new.strip():
        return "whitespace only"
    if any(ch.isdigit() for ch in old + new):
        return "content: number"
    if any(ch.isalpha() for ch in old + new):
        return "content: word/identifier"
    return "content: operator/punctuation"


def first_edit(tok, bench: str, remask: dict, plain: dict) -> dict | None:
    new_ids = tok.encode(remask["raw_completion"], add_special_tokens=False)
    old_ids = tok.encode(plain["raw_completion"], add_special_tokens=False)
    i = next((j for j, (a, b) in enumerate(zip(old_ids, new_ids)) if a != b), min(len(old_ids), len(new_ids)))
    if i == len(old_ids) == len(new_ids):
        return None  # the edit left the scored text unchanged (e.g. past the stop)
    old_id = old_ids[i] if i < len(old_ids) else None
    new_id = new_ids[i] if i < len(new_ids) else None
    old = tok.decode([old_id]) if old_id is not None else None
    new = tok.decode([new_id]) if new_id is not None else None
    offset = len(tok.decode(old_ids[:i]))
    return {
        "index": i, "old": old, "new": new, "kind": kind(old, new, old_id, new_id),
        "prose": bench in CODE and in_prose(plain["raw_completion"], offset, plain.get("parsed_output")),
    }


def summarize(tok, experiment: str, bench: str, delta: str, root: Path) -> dict | None:
    remask, plain = arms(experiment, bench, delta, root)
    if not remask:
        return None
    edited = [k for k, r in remask.items() if int(r.get("tokens_changed_by_remask", 0)) > 0]
    fixed = sum(bool(remask[k]["score"]) and not bool(plain[k]["score"]) for k in remask)
    broke = sum(not bool(remask[k]["score"]) and bool(plain[k]["score"]) for k in remask)
    firsts = [f for k in edited if (f := first_edit(tok, bench, remask[k], plain[k])) is not None]
    kinds = collections.Counter(f["kind"] for f in firsts)
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for f in firsts:
        if len(examples[f["kind"]]) < 4:
            examples[f["kind"]].append(f"{f['old']!r}→{f['new']!r}@{f['index']}")
    out = {
        "experiment": experiment, "benchmark": bench, "delta": delta, "tasks": len(remask),
        "score_remask": sum(bool(r["score"]) for r in remask.values()),
        "score_plain": sum(bool(r["score"]) for r in plain.values()),
        "edited_tasks": len(edited),
        "edits": sum(int(remask[k]["tokens_changed_by_remask"]) for k in edited),
        "remask_passes": sum(int(r.get("remask_passes", 0)) for r in remask.values()),
        "skipped_passes": sum(int(r.get("remask_skipped", 0)) for r in remask.values()),
        "fixed": fixed, "broke": broke, "mcnemar_p": mcnemar(fixed, broke),
        "first_edits": len(firsts),
        "first_at_position_0": sum(f["index"] == 0 for f in firsts),
        "first_in_prose": sum(f["prose"] for f in firsts),
        "first_kinds": dict(kinds.most_common()),
        "first_examples": dict(examples),
    }
    if experiment != "exp05":
        logged = []
        for k in edited:
            edits = remask[k].get("remask_edits") or []
            logged += ast.literal_eval(edits) if isinstance(edits, str) else edits
        out["all_edit_kinds"] = dict(collections.Counter(
            kind(tok.decode([e["old_token"]]), tok.decode([e["new_token"]]), e["old_token"], e["new_token"])
            for e in logged).most_common())
        out["all_edits_at_position_0"] = sum(e["canvas_index"] == 0 for e in logged)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cross8b", type=Path, default=PROJECT / "outputs/cross8b")
    parser.add_argument("--swap", type=Path, default=PROJECT / "outputs/cross8b_swap")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    roots = {"exp05": EXP05, "cross8b": args.cross8b, "swap": args.swap}
    rows = [row for bench in BENCHMARKS for delta in DELTAS for experiment in roots
            if (row := summarize(tok, experiment, bench, delta, roots[experiment])) is not None]
    for row in rows:
        head = (f"{row['benchmark']:14s} δ={row['delta']} {row['experiment']:7s} n={row['tasks']:3d} "
                f"score {row['score_plain']}→{row['score_remask']}  edited {row['edited_tasks']:3d} "
                f"({row['edits']} edits, {row['skipped_passes']}/{row['remask_passes']} passes skipped)  "
                f"fix {row['fixed']} / break {row['broke']} (p={row['mcnemar_p']:.2f})")
        print(head)
        n = max(row["first_edits"], 1)
        print(f"    first edits {row['first_edits']}: pos0 {row['first_at_position_0'] / n:.0%}, "
              f"prose {row['first_in_prose'] / n:.0%}  " +
              ", ".join(f"{k} {v / n:.0%}" for k, v in row["first_kinds"].items()))
        for k, ex in row["first_examples"].items():
            print(f"      {k}: {'   '.join(ex)}")
        if "all_edit_kinds" in row:
            total = max(sum(row["all_edit_kinds"].values()), 1)
            print(f"    all logged edits {total}: pos0 {row['all_edits_at_position_0'] / total:.0%}  " +
                  ", ".join(f"{k} {v / total:.0%}" for k, v in row["all_edit_kinds"].items()))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
