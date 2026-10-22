"""The contention guard: who counts as "other", waiting, and watching a run."""
from __future__ import annotations

import os
import threading

from sdar_harness.contention import (
    ContentionGuard,
    NullGuard,
    other_processes,
    parse_compute_apps,
)

OURS, THEIRS = "GPU-aaaa", "GPU-bbbb"


def apps(*rows: tuple[int, str, int]) -> str:
    return "".join(f"{pid}, {uuid}, {mib}\n" for pid, uuid, mib in rows)


def test_parse_compute_apps_reads_pid_card_and_memory() -> None:
    text = "1120510, GPU-34a4, 3378\n1184768, GPU-34a4, [N/A]\n\n"
    assert parse_compute_apps(text) == [(1120510, "GPU-34a4", 3378), (1184768, "GPU-34a4", None)]


def test_only_other_processes_on_our_cards_count() -> None:
    text = apps((10, OURS, 8000), (11, OURS, 3000), (12, THEIRS, 5000))
    seen = other_processes([OURS], self_pid=10, query=lambda *args: text)
    assert [p.pid for p in seen] == [11]
    assert seen[0].gpu == OURS and seen[0].used_mib == 3000


def test_wait_until_clear_polls_until_the_cards_are_free() -> None:
    answers = iter([apps((11, OURS, 1)), apps((11, OURS, 1)), ""])
    slept: list[float] = []
    clock = iter(range(100))
    guard = ContentionGuard(
        [OURS], poll_seconds=30, self_pid=10, query=lambda *args: next(answers),
        sleep=slept.append, clock=lambda: float(next(clock)), log=lambda message: None,
    )
    guard.wait_until_clear()
    assert slept == [30, 30]


def test_wait_until_clear_does_not_block_when_nvidia_smi_fails() -> None:
    def broken(*args: str) -> str:
        raise OSError("no nvidia-smi")

    guard = ContentionGuard([OURS], self_pid=10, query=broken, sleep=lambda s: None)
    assert guard.probe() is None
    assert guard.wait_until_clear() >= 0


def test_watch_catches_a_process_that_comes_and_goes_mid_run() -> None:
    # Clear at entry and at exit; another process shows up only in between,
    # which only the sampler can see.
    calls = {"n": 0}
    sampled = threading.Event()

    def query(*args: str) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            sampled.set()
            return apps((11, OURS, 1))
        return ""

    guard = ContentionGuard([OURS], sample_seconds=0.01, self_pid=10, query=query)
    with guard.watch() as watch:
        assert sampled.wait(5)
    assert watch.contended and list(watch.processes) == [11]
    record = watch.record()
    assert record["contention_processes"][0]["pid"] == 11
    assert len(record["host_load_1min"]) == 2


def test_clean_watch_records_no_processes() -> None:
    guard = ContentionGuard([OURS], sample_seconds=0.01, self_pid=os.getpid(), query=lambda *a: "")
    with guard.watch() as watch:
        pass
    assert not watch.contended and watch.record()["contention_processes"] == []


def test_null_guard_never_waits_or_flags() -> None:
    guard = NullGuard()
    assert guard.wait_until_clear() == 0.0
    with guard.watch() as watch:
        pass
    assert not watch.contended and guard.settings() == {"mode": "off"}
