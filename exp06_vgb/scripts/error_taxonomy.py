#!/usr/bin/env python3
"""What kind of error each fixed and adaptive decoding arm makes.

Every failed task of every arm gets one error class:

  unfinished   ran out of canvas before the answer (GSM8K) or the function (code) was complete
  syntax       ended early or wrote broken code: the function as written does not
               parse (judged on the raw completion -- the scorer's sanitizer drops
               broken lines, which turns a syntax error into a function missing its
               return), stops before its return or has no body, is a skeleton that
               never does the task, or degenerates into repetition; for GSM8K, stops
               with no answer
  interface    code that misreads the prompt's interface: no function with the
               entry point's name, or one that rejects the prompt's arguments
  prompt       misreads the task: ignores or misreads a stated quantity or condition
               (GSM8K), or a clause of the spec or the prompt's example (code)
  careless     a local slip in a right plan: an arithmetic or copying slip (GSM8K);
               for code, one line away from a correct solution another arm wrote, a
               typo-like NameError/AttributeError, or a hand-labelled slip
               (off by one, a wrong variable, a missing return branch)
  logic        a wrong plan: wrong reasoning (GSM8K), a wrong algorithm or formula,
               or code that never returns (timeout)
  edge         code that passes the base tests but fails EvalPlus's extra ones
  harness      not the model's error: HumanEval code that calls a helper the prompt
               defines, correct once that helper is added

Code classes come first from rules (the base tests are re-run in a subprocess
with EvalPlus's input handling to get the first failure's exception); what the
rules leave as `logic` is then hand-labelled in report/data/error_labels_code.json
(prompt / logic / careless / syntax, with the reason). GSM8K classes are all hand
labels, in report/data/error_labels_gsm8k.json; `dump` prints the texts. Each hand
label names the error in one line, so the classes can be audited.

    python scripts/error_taxonomy.py build   # writes outputs/error_taxonomy.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXP05 = PROJECT.parent / "exp05_adaptive_re"
EVALPLUS = Path.home() / ".cache/evalplus"
EVALPLUS_PYTHON = str(Path.home() / "miniconda3/envs/diffusion/bin/python")  # has evalplus 0.3.1
LABELS = PROJECT / "report/data/error_labels_gsm8k.json"
CODE_LABELS = PROJECT / "report/data/error_labels_code.json"
OUTPUT = PROJECT / "outputs/error_taxonomy.json"
BENCHMARKS = ("gsm8k", "mbpp_plus", "humaneval_plus")
CODE = {"mbpp_plus", "humaneval_plus"}

CLASSES = ["unfinished", "syntax", "interface", "prompt", "careless", "logic", "edge", "harness"]

# key: (label, group, records directory)
ARMS = {
    "b8": ("SDAR-8B fixed k=1", "baseline", PROJECT / "outputs/sdar8b/{b}/fixed_k_1"),
    "b17": ("SDAR-1.7B fixed k=1", "baseline", EXP05 / "outputs/baseline/sdar/{b}/fixed_k_1"),
    "cross_off_0p10": ("cross8b δ=0.1, remask off", "cross8b", PROJECT / "outputs/cross8b/{b}/xadaptive_noveto_delta_0p10"),
    "cross_on_0p10": ("cross8b δ=0.1, remask on", "cross8b", PROJECT / "outputs/cross8b/{b}/xadaptive_remask_noveto_delta_0p10"),
    "cross_off_1p00": ("cross8b δ=1, remask off", "cross8b", PROJECT / "outputs/cross8b/{b}/xadaptive_noveto_delta_1p00"),
    "cross_on_1p00": ("cross8b δ=1, remask on", "cross8b", PROJECT / "outputs/cross8b/{b}/xadaptive_remask_noveto_delta_1p00"),
    "swap_off_0p10": ("swap δ=0.1, remask off", "swap", PROJECT / "outputs/cross8b_swap/{b}/xadaptive_noveto_delta_0p10"),
    "swap_on_0p10": ("swap δ=0.1, remask on", "swap", PROJECT / "outputs/cross8b_swap/{b}/xadaptive_remask_noveto_delta_0p10"),
    "swap_off_1p00": ("swap δ=1, remask off", "swap", PROJECT / "outputs/cross8b_swap/{b}/xadaptive_noveto_delta_1p00"),
    "swap_on_1p00": ("swap δ=1, remask on", "swap", PROJECT / "outputs/cross8b_swap/{b}/xadaptive_remask_noveto_delta_1p00"),
}
# (remask on, its counterfactual): the same proposer, the same task, remasking off.
PAIRS = [("cross_on_0p10", "cross_off_0p10"), ("cross_on_1p00", "cross_off_1p00"),
         ("swap_on_0p10", "swap_off_0p10"), ("swap_on_1p00", "swap_off_1p00")]


def records(arm: str, bench: str) -> dict[str, dict]:
    folder = Path(str(ARMS[arm][2]).format(b=bench))
    path = folder / "records.scored.jsonl"
    if not path.exists():
        path = folder / "records.jsonl"
    return {r["example_id"]: r for r in map(json.loads, path.open())}


def solutions(arm: str, bench: str) -> dict[str, str]:
    """The exact text EvalPlus executed for each task."""
    folder = Path(str(ARMS[arm][2]).format(b=bench))
    return {r["task_id"]: r["solution"] for r in map(json.loads, (folder / "evalplus_samples.jsonl").open())}


def problems(bench: str) -> dict[str, dict]:
    name = "MbppPlus-v0.2.0.jsonl" if bench == "mbpp_plus" else "HumanEvalPlus-v0.1.10.jsonl"
    return {p["task_id"]: p for p in map(json.loads, (EVALPLUS / name).open())}


# ------------------------------------------------------------------ code: re-run the base tests
TYPO_ERRORS = {"NameError", "UnboundLocalError", "AttributeError", "ImportError", "ModuleNotFoundError"}
RUNNER = r"""
import json, math, sys
sys.setrecursionlimit(10000)
problem, solution, mode = json.loads(sys.stdin.read())
if mode == "mbpp":  # EvalPlus turns some MBPP inputs back into tuples
    from evalplus.data.mbpp import mbpp_deserialize_inputs
    problem["base_input"] = mbpp_deserialize_inputs(problem["task_id"], problem["base_input"])
def result(**kw):
    print(json.dumps(kw)); sys.exit(0)
env = {}
try:
    exec(compile(solution, "<solution>", "exec"), env)
except SyntaxError as e:
    result(stage="compile", error="SyntaxError", detail=str(e))
except BaseException as e:
    result(stage="define", error=type(e).__name__, detail=str(e)[:200])
fn = env.get(problem["entry_point"])
if fn is None:
    result(stage="define", error="MissingEntryPoint", detail=problem["entry_point"])
ref = {}
exec(problem["prompt"] + problem["canonical_solution"], ref)
expected_fn = ref[problem["entry_point"]]
import copy
def same(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=max(atol, 1e-6))
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(same(x, y, atol) for x, y in zip(a, b))
    return a == b
for i, args in enumerate(problem["base_input"]):
    want = expected_fn(*copy.deepcopy(args))
    try:
        got = fn(*copy.deepcopy(args))
    except TypeError as e:
        msg = str(e)
        arity = "positional argument" in msg or "takes" in msg or "missing" in msg and "required" in msg
        result(stage="call", error="ArityError" if arity and i == 0 else "TypeError", detail=msg[:200], test=i)
    except BaseException as e:
        result(stage="call", error=type(e).__name__, detail=str(e)[:200], test=i)
    if not same(got, want, problem.get("atol", 0) or 0):
        result(stage="check", error="WrongAnswer", detail=repr(got)[:120] + " != " + repr(want)[:120], test=i)
result(stage="pass", error=None)
"""


def run_base(job: tuple[dict, str]) -> dict:
    import subprocess

    problem, solution = job
    fields = ("task_id", "entry_point", "prompt", "canonical_solution", "base_input", "atol")
    payload = json.dumps([{k: problem[k] for k in fields}, solution,
                          "mbpp" if problem["task_id"].startswith("Mbpp/") else "humaneval"])
    try:
        out = subprocess.run([EVALPLUS_PYTHON, "-c", RUNNER], input=payload, capture_output=True, text=True,
                             timeout=20, cwd="/tmp")
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"stage": "call", "error": "Timeout"}
    except (json.JSONDecodeError, IndexError):
        return {"stage": "call", "error": "Crashed", "detail": out.stderr[-200:]}


def code_lines(text: str) -> list[str]:
    """Non-blank, non-comment lines with the indentation stripped."""
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def lines_to_fix(wrong: str, correct: list[str]) -> int | None:
    """Fewest lines to change to reach a correct solution another arm wrote for this task."""
    best = None
    ours = code_lines(wrong)
    for other in correct:
        ops = difflib.SequenceMatcher(None, ours, code_lines(other), autojunk=False).get_opcodes()
        changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
        best = changed if best is None else min(best, changed)
    return best


def raw_function_error(raw: str, entry_point: str) -> str | None:
    """Does the function the model wrote parse, before the scorer's sanitizer touches it?

    The sanitizer drops lines it cannot parse, so a function with one broken
    line is scored as a function missing that line (often its return). Read the
    `def` block straight from the completion instead: the `def` line and every
    blank or indented line after it, up to a code fence or unindented text.
    """
    import ast
    import textwrap

    lines = raw.replace("\r\n", "\n").split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(rf"\s*def {re.escape(entry_point)}\b", ln)), None)
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for ln in lines[start + 1:]:
        if ln.strip().startswith("```") or (ln.strip() and len(ln) - len(ln.lstrip()) <= indent):
            break
        block.append(ln)
    try:
        ast.parse(textwrap.dedent("\n".join(block)))
    except SyntaxError as e:
        bad = block[e.lineno - 1].strip() if e.lineno and e.lineno <= len(block) else ""
        return f"{e.msg}: {bad[:60]!r}" if bad else e.msg
    return None


def stopped_early(solution: str, entry_point: str) -> str | None:
    """A function that ends before doing its job: no body past the docstring, or no return."""
    import ast

    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == entry_point), None)
    if fn is None:
        return None
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if not body or all(isinstance(n, ast.Pass) for n in body):
        return "stops after the signature and docstring: no body"
    if not any(isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)):
        return "stops before the return: the function never returns a value"
    return None


def prompt_helper(run: dict, prompt: str) -> str | None:
    """The name a NameError reports, when the prompt itself defines it."""
    m = re.search(r"name '(\w+)'", run.get("detail") or "")
    if run.get("error") == "NameError" and m and re.search(rf"^def {m.group(1)}\b", prompt, re.M):
        return m.group(1)
    return None


def classify_code(record: dict, run: dict, fix: int | None, early: str | None,
                  raw_error: str | None = None) -> tuple[str, str]:
    failure = record.get("failure_type")
    if run.get("helper"):
        if run.get("error") is None:
            return "harness", f"correct once the prompt's helper {run['helper']}() is added; the scorer drops it"
        cls, why = classify_code(record, {**run, "helper": None}, fix, early, raw_error)
        return cls, why + f" (re-run with the prompt's {run['helper']}() added)"
    if failure == "plus_fail":
        return "edge", "passes the base tests, fails EvalPlus's extra ones"
    error = run.get("error")
    if error == "SyntaxError" or failure == "syntax_error":
        if record.get("maximum_length_termination"):
            return "unfinished", "cut off by the canvas mid-code"
        return "syntax", f"does not parse: {(run.get('detail') or record.get('syntax_error') or '')[:90]}"
    if raw_error:
        if record.get("maximum_length_termination"):
            return "unfinished", f"cut off by the canvas mid-function ({raw_error})"
        return "syntax", f"the function as written does not parse ({raw_error}); the sanitizer dropped the broken lines"
    if error == "MissingEntryPoint":
        return "interface", f"no function named {run.get('detail')}"
    if error == "ArityError":
        return "interface", f"rejects the prompt's arguments: {run.get('detail', '')[:90]}"
    if early and (error == "WrongAnswer" and (run.get("detail") or "").startswith("None") or "no body" in early):
        return "syntax", early
    if fix is not None and fix <= 1:
        return "careless", f"one line away from a correct solution another arm wrote ({error or 'wrong answer'})"
    if error in TYPO_ERRORS:
        return "careless", f"{error}: {run.get('detail', '')[:90]}"
    if failure == "timeout" or error == "Timeout":
        return "logic", "never returns (timeout)"
    if error == "WrongAnswer":
        return "logic", "runs, returns wrong answers"
    if error is None:
        return "logic", "passes our base re-run, fails EvalPlus's"
    return "logic", f"crashes: {error}: {run.get('detail', '')[:80]}"


# ------------------------------------------------------------------ GSM8K
ARITH = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?(?:\s*[-+*/×x÷]\s*\d[\d,]*(?:\.\d+)?)+)\s*=\s*\$?(\d[\d,]*(?:\.\d+)?)")


def arithmetic_slips(text: str) -> list[str]:
    """`a op b = c` lines whose c is wrong, in plain or LaTeX-light notation."""
    clean = text.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("$", "")
    slips = []
    for m in ARITH.finditer(clean):
        expr = m.group(1).replace(",", "").replace("×", "*").replace("÷", "/")
        expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr)
        try:
            value = eval(expr, {"__builtins__": {}})  # digits and operators only
        except Exception:
            continue
        claimed = float(m.group(2).replace(",", ""))
        if abs(value - claimed) > 1e-6 * max(1, abs(value)) and abs(round(value, 2) - claimed) > 1e-6:
            slips.append(f"{m.group(1).strip()} = {m.group(2)} (is {value:g})")
    return slips


def gold(bench: str) -> dict[str, dict]:
    return {r["example_id"]: r for r in map(json.loads, (PROJECT.parent / "data" / bench / "test.jsonl").open())}


def dump(args) -> None:
    """Print every GSM8K failure that needs a hand label."""
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    data = gold("gsm8k")
    for arm in ARMS:
        for eid, r in records(arm, "gsm8k").items():
            if r["score"] or f"{arm}|{eid}" in labels:
                continue
            q = data[eid]
            print(f"### {arm}|{eid}  gold={q.get('answer', q.get('target'))!s:.40}  got={r.get('parsed_output')}"
                  f"  maxlen={r.get('maximum_length_termination')}  slips={arithmetic_slips(r['raw_completion'])}")
            if args.questions:
                print("Q:", q["question"])
            print(r["raw_completion"].strip()[: args.chars])
            print()


# ------------------------------------------------------------------ build
def build(args) -> None:
    labels = json.loads(LABELS.read_text())
    code_labels = json.loads(CODE_LABELS.read_text())
    out: dict = {"classes": CLASSES, "arms": {k: {"label": v[0], "group": v[1]} for k, v in ARMS.items()},
                 "rows": []}
    for bench in BENCHMARKS:
        recs = {arm: records(arm, bench) for arm in ARMS}
        if bench in CODE:
            probs = problems(bench)
            sols = {arm: solutions(arm, bench) for arm in ARMS}
            correct: dict[str, list[str]] = {}
            for arm in ARMS:
                for eid, r in recs[arm].items():
                    if r["score"]:
                        correct.setdefault(eid, []).append(sols[arm][eid])
            jobs = [(arm, eid) for arm in ARMS for eid, r in recs[arm].items()
                    if not r["score"] and r.get("failure_type") != "plus_fail"]
            with mp.Pool(args.workers) as pool:
                runs = dict(zip(jobs, pool.map(run_base, [(probs[e], sols[a][e]) for a, e in jobs])))
                # HumanEval prompts define helpers the scorer does not carry over:
                # re-run those with the prompt prepended to find the real error.
                redo = [(k, h) for k, run in runs.items() if (h := prompt_helper(run, probs[k[1]]["prompt"]))]
                again = pool.map(run_base, [(probs[e], probs[e]["prompt"] + "\n" + sols[a][e]) for (a, e), _ in redo])
                for ((key, helper), run) in zip(redo, again):
                    runs[key] = {**run, "helper": helper}
        for arm in ARMS:
            for eid, r in recs[arm].items():
                row = {"arm": arm, "bench": bench, "id": eid, "score": bool(r["score"])}
                if not r["score"]:
                    if bench in CODE:
                        run = runs.get((arm, eid), {})
                        fix = lines_to_fix(sols[arm][eid], correct.get(eid, []))
                        early = stopped_early(sols[arm][eid], probs[eid]["entry_point"])
                        raw_error = raw_function_error(r["raw_completion"], probs[eid]["entry_point"])
                        row["cls"], row["why"] = classify_code(r, run, fix, early, raw_error)
                        if row["cls"] == "logic":  # the rules cannot tell a misread spec from a wrong algorithm
                            label = code_labels[f"{arm}|{eid}"]
                            row["rule"] = row["why"]
                            row["cls"], row["why"] = label["cls"], label["why"]
                        row["error"] = run.get("error")
                        row["fix_lines"] = fix
                    else:
                        label = labels[f"{arm}|{eid}"]
                        row["cls"], row["why"] = label["cls"], label["why"]
                if "_on_" in arm:
                    row["remasks"] = int(r.get("tokens_changed_by_remask") or 0)
                out["rows"].append(row)
    OUTPUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUTPUT}: {sum(not r['score'] for r in out['rows'])} failures classified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--chars", type=int, default=1400)
    d.add_argument("--questions", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    {"dump": dump, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
