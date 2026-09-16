"""End-to-end cover for --trace-limit and for resuming a partially done method.

Tracing costs a whole extra generation per task, so long runs only trace a
prefix of each method. That makes two things easy to get wrong: writing traces
for a task that was never traced, and the resume path discarding untraced
records (which would make a restarted run redo the same tasks forever).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_benchmark
from dream_ablation.scheduler import GenerationConfig, GenerationResult


class FakeTokenizer:
    name_or_path = "fake/model"


class FakeConfig:
    _commit_hash = "deadbeef"
    max_position_embeddings = 4096


class FakeModel:
    config = FakeConfig()

    def parameters(self):
        import torch

        yield torch.zeros(1, dtype=torch.bfloat16)


class FakeProvider:
    tokenizer = FakeTokenizer()
    model = FakeModel()

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cpu")

    def render_prompt(self, prompt: str, *, use_chat_template: bool = True) -> str:
        return prompt

    def encode_prompt(self, rendered: str):
        import torch

        return torch.zeros((1, 4), dtype=torch.long)

    def decode(self, token_ids: list[int]) -> str:
        return "42"


def fake_generate(provider, prompt_ids, config, *, capture_traces=False):
    """Three deterministic denoising steps; traces only when asked."""
    traces = []
    if capture_traces:
        traces = [{"step_index": i, "accepted_prefix_length": 1} for i in range(3)]
    return GenerationResult(
        continuation_token_ids=[1, 2, 3],
        traces=traces,
        denoising_steps=3,
        proposer_passes=3,
        verifier_rows=0,
        verifier_forward_passes=0,
        commitment_histogram=[3, 0, 0, 0, 0, 0, 0, 0],
        eos_position=None,
        maximum_length_termination=True,
    )


def rows(count: int) -> list[dict]:
    return [
        {"benchmark": "gsm8k", "example_id": f"t{i}", "prompt": "q", "target": "42"}
        for i in range(count)
    ]


@pytest.fixture()
def patched(monkeypatch):
    monkeypatch.setattr(run_benchmark, "generate", fake_generate)
    monkeypatch.setattr(run_benchmark, "score_record", lambda row, completion: {})
    monkeypatch.setattr(run_benchmark, "synchronize", lambda device: None)
    monkeypatch.setattr(run_benchmark, "reset_peak_memory", lambda device: None)
    monkeypatch.setattr(run_benchmark, "peak_memory", lambda device: 0)
    return FakeProvider()


def run(provider, output_root: Path, *, trace_limit: int, count: int = 5) -> Path:
    config = GenerationConfig(method="fixed", continuation_length=8, fixed_k=1)
    run_benchmark.run_method(
        provider, rows(count), config, output_root,
        timing_repeats=1, trace_limit=trace_limit,
        use_chat_template=False, overwrite=False,
    )
    return output_root / config.label


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sample_selection_is_paired_reproducible_and_not_a_prefix():
    source = rows(20)
    first = run_benchmark.select_sample(source, 6, seed=12345)
    second = run_benchmark.select_sample(source, 6, seed=12345)
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert [row["example_id"] for row in first] != [f"t{i}" for i in range(6)]
    positions = [source.index(row) for row in first]
    assert positions == sorted(positions)


def test_untraced_tasks_do_not_crash_or_write_traces(patched, tmp_path):
    """Regression: the trace loop used to dereference a None result."""
    method_dir = run(patched, tmp_path, trace_limit=2, count=5)
    records = read(method_dir / "records.jsonl")
    traces = read(method_dir / "decision_traces.jsonl")
    assert len(records) == 5
    assert [r["traced"] for r in records] == [True, True, False, False, False]
    # Three steps of traces for each of the two traced tasks, and nothing else.
    assert len(traces) == 6
    assert {t["example_id"] for t in traces} == {"t0", "t1"}


def test_trace_limit_zero_writes_no_traces(patched, tmp_path):
    method_dir = run(patched, tmp_path, trace_limit=0, count=3)
    assert len(read(method_dir / "records.jsonl")) == 3
    assert read(method_dir / "decision_traces.jsonl") == []


def test_resume_keeps_untraced_records(patched, tmp_path):
    """A restarted run must not redo tasks that were finished but untraced."""
    method_dir = run(patched, tmp_path, trace_limit=2, count=5)
    first = read(method_dir / "records.jsonl")
    assert len(first) == 5

    # Second pass over the same directory: everything is already complete, so
    # no task should be re-executed and no record duplicated.
    calls = []

    def counting_generate(*args, **kwargs):
        calls.append(1)
        return fake_generate(*args, **kwargs)

    run_benchmark.generate = counting_generate
    try:
        run(patched, tmp_path, trace_limit=2, count=5)
    finally:
        run_benchmark.generate = fake_generate

    second = read(method_dir / "records.jsonl")
    assert len(second) == 5, "resume duplicated or dropped records"
    assert [r["example_id"] for r in second] == [r["example_id"] for r in first]
    # Only the untimed warmup may run, and only if something were pending.
    assert not calls, "resume re-executed already-completed tasks"


def test_resume_redoes_a_task_whose_traces_were_truncated(patched, tmp_path):
    """A kill mid-task leaves a record with partial traces; that must be redone."""
    method_dir = run(patched, tmp_path, trace_limit=2, count=5)
    traces_path = method_dir / "decision_traces.jsonl"
    kept = [t for t in read(traces_path) if not (t["example_id"] == "t1" and t["step_index"] == 2)]
    traces_path.write_text("".join(json.dumps(t) + "\n" for t in kept))

    completed = run_benchmark.repair_partial_artifacts(method_dir / "records.jsonl", traces_path)
    assert "t1" not in completed, "task with truncated traces should be redone"
    assert {"t0", "t2", "t3", "t4"} <= completed
