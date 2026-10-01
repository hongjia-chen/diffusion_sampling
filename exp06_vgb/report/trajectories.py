#!/usr/bin/env python3
"""Decoding-path examples for report.html: selection, traced re-runs, verification.

Every example shows four panels, stepped in lockstep:

  SDAR-8B fixed k=1     outputs/sdar8b                       (the verifier alone)
  SDAR-1.7B fixed k=1   ../exp05_adaptive_re/outputs/baseline (the proposer alone)
  cross8b delta=0.1     outputs/cross8b, remasking on         (1.7B proposes, 8B verifies)
  cross8b delta=1       outputs/cross8b, remasking on

and, for the two cross8b panels, the same arm with remasking off (the paired
counterfactual) as a toggle.

Three examples per benchmark, one per category, hand-picked among the tasks
that pass the category's filter for a short answer and a single, readable edit:

  all_fail     all four panels wrong.
  remask_hurt  a fixed k=1 baseline right and a remask arm wrong; remasking
               caused it (its no-remask counterfactual is right) wherever such
               a task exists. cross8b MBPP+ has none (11 fixes, 0 breaks), so
               its example has both remask arms wrong without the edit being
               the cause.
  remask_fix   remasking turned a wrong answer right (remask arm right, its
               counterfactual wrong), at both delta where possible. No GSM8K
               task is fixed at both.

The benchmark runs traced only their first 5 tasks (the baselines none), so
the traces come from re-running just these tasks with each arm's recorded
flags. Seeds are per (seed, method, task), so a subset decodes as the full run
did; `verify` refuses any re-run whose completion, EOS position or step count
differs from the recorded run. The 1.7B baseline re-runs with exp05's own code,
which produced it.

    python report/trajectories.py plan      # subsets + data/traced/launch.sh
    bash report/data/traced/launch.sh       # ~20 min on cards 0+1
    python report/trajectories.py plan vgb  # data/traced/launch_vgb.sh: VGB Track A's three arms
    bash report/data/traced/launch_vgb.sh   # ~2.7 h on cards 0+1 (the runner replays each traced task)
    python report/trajectories.py verify

The VGB arms are sampled (T = 1), so their re-run is exact only because the
sampler's generator is seeded per (seed, method, task) and tracing draws no
random numbers (tests/test_vgb.py::test_tracing_does_not_change_the_walk);
`verify` also checks each block's list of remasks against Track A.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXP05 = PROJECT.parent / "exp05_adaptive_re"
REPORT = PROJECT / "report"
TRACED = REPORT / "data" / "traced"
DATA = PROJECT.parent / "data"
CANVAS = {"gsm8k": 256, "mbpp_plus": 256, "humaneval_plus": 512}
BENCHMARKS = ["gsm8k", "mbpp_plus", "humaneval_plus"]

# (category, benchmark, example_id, short label, what happens)
SELECTION = [
    ("all_fail", "gsm8k", "gsm8k-test-255", "four different wrong answers",
     "four different wrong answers (176, 220, 20, 16; the answer is 192). The δ=1 edit turns "
     "“among the ten twenty stalls” into “twenty twenty”: a wording repair that leaves the arithmetic alone"),
    ("remask_hurt", "gsm8k", "gsm8k-test-581", "8B edits 1 x 60 = 90 to = 60",
     "both baselines are right. At δ=1 the 1.7B drafts “1 x 60 = 90”, dropping the + 30 but keeping the "
     "right total. The 8B “corrects” the arithmetic to = 60 and the answer becomes 185"),
    ("remask_fix", "gsm8k", "gsm8k-test-1248", "a wording edit, then a re-generated fix",
     "without remasking, δ=0.1 multiplies 375 × 30. The first edit only rewords For → However, "
     "and the re-generated last line computes 30 × 25 = 750. Wording changed, and regeneration did the fixing"),
    ("all_fail", "mbpp_plus", "Mbpp/791", "every panel fails differently",
     "every panel fails differently. Remasking renames t → tup at both δ; at δ=0.1 it also swaps "
     "tuple for filter, still wrong"),
    ("remask_hurt", "mbpp_plus", "Mbpp/166", "the edit renames; return stays missing",
     "the 1.7B alone is right. Both δ lose the final return count, with or without remasking; the block's "
     "one edit renames lst → arr, the 8B's own habit, and the missing return stays missing"),
    ("remask_fix", "mbpp_plus", "Mbpp/581", "4 → 2 in the lateral area",
     "both baselines are wrong. At both δ the 8B replaces 4 → 2 in the lateral area, a real logic fix "
     "that neither model makes on its own (the other edit is the usual :\\n → :\\r\\n)"),
    ("all_fail", "humaneval_plus", "HumanEval/126", "> → >=, still wrong",
     "every panel ignores the duplicates rule. Remasking makes a logic edit at both δ, > → >=, which "
     "rejects every repeat instead of the triples only: still wrong"),
    ("remask_hurt", "humaneval_plus", "HumanEval/88", "a dropped ( unbalances the if",
     "the 8B alone is right, and δ=1 is right without remasking. Its one edit drops a redundant-looking "
     "( from if ((…) % 2 == 0):, which leaves the parentheses unbalanced"),
    ("remask_fix", "humaneval_plus", "HumanEval/110", "== → + makes the condition right",
     "the same one-token edit at both δ: len(even1) == len(even2) becomes len(even1) + len(even2) >= len(lst1), "
     "which is the correct condition"),
]

CATEGORY_LABEL = {
    "all_fail": "Every panel wrong",
    "remask_hurt": "A baseline right, remasking wrong",
    "remask_fix": "Remasking fixed it",
}

# arm key -> (runner, method flag, recorded run directory for one benchmark)
ARMS = {
    "sdar8b": ("sdar8b", "fixed:1", lambda b: PROJECT / "outputs/sdar8b" / b / "fixed_k_1"),
    "sdar1p7b": ("sdar1p7b", "fixed:1", lambda b: EXP05 / "outputs/baseline/sdar" / b / "fixed_k_1"),
    "remask_0p10": ("cross", "xadaptive-remask:0.10",
                    lambda b: PROJECT / "outputs/cross8b" / b / "xadaptive_remask_noveto_delta_0p10"),
    "remask_1p00": ("cross", "xadaptive-remask:1.00",
                    lambda b: PROJECT / "outputs/cross8b" / b / "xadaptive_remask_noveto_delta_1p00"),
    "plain_0p10": ("cross", "xadaptive:0.10",
                   lambda b: PROJECT / "outputs/cross8b" / b / "xadaptive_noveto_delta_0p10"),
    "plain_1p00": ("cross", "xadaptive:1.00",
                   lambda b: PROJECT / "outputs/cross8b" / b / "xadaptive_noveto_delta_1p00"),
    # VGB Track A (launch_vgb.sh): T = 1, 8x8 shortlist, interior rule, momentum
    "vgb_fwd": ("vgb", "vgb:fwdonly:sl8x8",
                lambda b: PROJECT / "outputs/vgb_trackA" / b / "vgb_fwdonly_sl8x8_T1"),
    "vgb_lam0p5": ("vgb", "vgb:lam0.5:mom:sl8x8",
                   lambda b: PROJECT / "outputs/vgb_trackA" / b / "vgb_lam0p5_interior_mom1_sl8x8_T1"),
    "vgb_lam1": ("vgb", "vgb:lam1:mom:sl8x8",
                 lambda b: PROJECT / "outputs/vgb_trackA" / b / "vgb_lam1_interior_mom1_sl8x8_T1"),
}
VGB_ARMS = ("vgb_fwd", "vgb_lam0p5", "vgb_lam1")


def records(directory: Path) -> dict[str, dict]:
    scored = directory / "records.scored.jsonl"
    path = scored if scored.exists() else directory / "records.jsonl"
    return {r["example_id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def rerun_dir(arm: str, bench: str) -> Path:
    return TRACED / ARMS[arm][0] / bench / ARMS[arm][2](bench).name


def plan(which: str = "base") -> None:
    python = Path.home() / "miniconda3/envs/diffusion/bin/python"
    overlay = EXP05 / "vendor_env" / "tf452"
    common = ["--family", "sdar", "--block-length", "64", "--dtype", "bfloat16", "--seed", "271828",
              "--timing-repeats", "1", "--require-lean-logits", "--max-candidates", "64",
              "--verifier-microbatch-size", "4", "--local-files-only", "--vocab-limit", "151665",
              "--overwrite"]
    jobs: list[tuple[Path, list[str]]] = []
    for bench in BENCHMARKS:
        wanted = [eid for _, b, eid, _, _ in SELECTION if b == bench]
        order = list(records(ARMS["remask_0p10"][2](bench)))
        rows = {r["example_id"]: r for r in map(json.loads, (DATA / bench / "test.jsonl").open())}
        subset = TRACED / "subsets" / f"{bench}.jsonl"
        subset.parent.mkdir(parents=True, exist_ok=True)
        subset.write_text("".join(json.dumps(rows[i]) + "\n" for i in order if i in wanted), encoding="utf-8")
        n = str(len(wanted))
        head = ["--dataset", str(subset), "--continuation-length", str(CANVAS[bench]), "--trace-limit", n]
        models = PROJECT.parent / "models"
        if which == "vgb":
            # launch_vgb.sh's arguments exactly, with the subset and tracing on
            jobs.append((PROJECT, ["scripts/run_benchmark.py", *head,
                                   "--output-root", str(TRACED / "vgb" / bench),
                                   "--family", "sdar", "--block-length", "64", "--dtype", "bfloat16",
                                   "--seed", "271828", "--timing-repeats", "1", "--require-lean-logits",
                                   "--local-files-only", "--vocab-limit", "151665", "--overwrite",
                                   "--model", str(models / "SDAR-8B-Chat-b64-local"), "--devices", "cuda:0", "cuda:1",
                                   "--verifier-model", str(models / "SDAR-1.7B-Chat-b64-local"),
                                   "--verifier-family", "sdar", "--verifier-device", "cuda:1",
                                   "--vgb-budget", "512", "--vgb-row-chunk", "16",
                                   "--methods", *[ARMS[a][1] for a in VGB_ARMS]]))
            continue
        jobs.append((PROJECT, ["scripts/run_benchmark.py", *head, *common,
                               "--output-root", str(TRACED / "sdar8b" / bench), "--methods", "fixed:1",
                               "--model", str(models / "SDAR-8B-Chat-b64-local"), "--devices", "cuda:0", "cuda:1",
                               "--fixed-temperature", "0.0", "--fixed-confidence", "logp"]))
        jobs.append((EXP05, ["scripts/run_benchmark.py", *head, *common,
                             "--output-root", str(TRACED / "sdar1p7b" / bench), "--methods", "fixed:1",
                             "--model", str(models / "SDAR-1.7B-Chat-b64-local"), "--device", "cuda:0",
                             "--fixed-temperature", "0.0", "--fixed-confidence", "logp"]))
        # launch_cross8b.sh, ROLES=1p7b_proposes
        methods = [ARMS[a][1] for a in ("remask_0p10", "remask_1p00", "plain_0p10", "plain_1p00")]
        jobs.append((PROJECT, ["scripts/run_benchmark.py", *head, *common,
                               "--output-root", str(TRACED / "cross" / bench), "--methods", *methods,
                               "--model", str(models / "SDAR-1.7B-Chat-b64-local"), "--device", "cuda:1",
                               "--verifier-model", str(models / "SDAR-8B-Chat-b64-local"),
                               "--verifier-family", "sdar", "--verifier-devices", "cuda:0", "cuda:1",
                               "--adaptive-temperature", "0.0", "--adaptive-confidence", "logp"]))
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by report/trajectories.py plan: one sequential queue on one card pair.",
        "set -uo pipefail",
        'PAIR="${PAIR:-0,1}"',
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
        f"mkdir -p {shlex.quote(str(TRACED / 'logs'))}",
    ]
    prefix = "vgb" if which == "vgb" else "job"
    for index, (cwd, cmd) in enumerate(jobs):
        log = shlex.quote(str(TRACED / "logs" / f"{prefix}{index}.log"))
        lines.append(
            f"(cd {shlex.quote(str(cwd))} && PYTHONPATH={shlex.quote(str(overlay))}:{shlex.quote(str(cwd))} "
            f'CUDA_VISIBLE_DEVICES="$PAIR" EXPERIMENT_PHYSICAL_GPU_ID="${{PAIR/,/+}}" '
            f"{shlex.quote(str(python))} -u {' '.join(shlex.quote(c) for c in cmd)}) > {log} 2>&1 "
            f"|| echo 'FAILED {prefix}{index}'"
        )
    lines.append("echo all jobs finished")
    out = TRACED / ("launch_vgb.sh" if which == "vgb" else "launch.sh")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(jobs)} jobs)")


def verify() -> bool:
    ok = True
    for _, bench, eid, _, _ in SELECTION:
        for arm, (_, _, recorded_dir) in ARMS.items():
            d = rerun_dir(arm, bench)
            try:
                rerun = records(d)[eid]
            except (FileNotFoundError, KeyError):
                print(f"MISSING  {bench:15s} {eid:16s} {arm}")
                ok = False
                continue
            original = records(recorded_dir(bench))[eid]
            diffs = [key for key in ("raw_completion", "eos_position", "denoising_steps")
                     if rerun[key] != original[key]]
            if rerun.get("remask_edits", []) != original.get("remask_edits", []):
                diffs.append("remask_edits")
            if arm in VGB_ARMS and [b["remasked"] for b in rerun["vgb_blocks"]] != [
                    b["remasked"] for b in original["vgb_blocks"]]:
                diffs.append("vgb remasks")
            steps = sum(1 for t in map(json.loads, (d / "decision_traces.jsonl").open())
                        if t["example_id"] == eid and not t.get("remask"))
            if steps != rerun["denoising_steps"]:
                diffs.append(f"traces={steps}")
            print(f"{'OK      ' if not diffs else 'DIVERGED'} {bench:15s} {eid:16s} {arm:12s} {diffs or ''}")
            ok &= not diffs
    return ok


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "plan":
        plan(sys.argv[2] if len(sys.argv) > 2 else "base")
    elif command == "verify":
        sys.exit(0 if verify() else 1)
    else:
        sys.exit(__doc__)
