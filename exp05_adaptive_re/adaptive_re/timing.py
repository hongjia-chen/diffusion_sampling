"""Per-stage timing that does not perturb what it measures.

CUDA is asynchronous, so wall-clock around a launch measures the *launch*, not
the work. `torch.cuda.synchronize()` would give true numbers but stalls the
pipeline at every stage boundary -- perturbing exactly the latency under study,
which is the one thing a latency experiment cannot afford. CUDA events are
recorded on the stream and read back once, at the end, so the in-band cost is a
pair of event records per stage occurrence.

Off by default. `StageTimer(device)` with `enabled=False` yields a null context
manager and records nothing, so the headline arms run byte-for-byte as before
and only an explicitly profiled run pays anything.

The totals are wall-clock spans on the stream, so stages that overlap would
double-count. Nothing here overlaps: the scheduler proposes, then verifies,
then the block loop commits, strictly in order.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch


class StageTimer:
    """Accumulates elapsed time per named stage.

    Usage::

        timer = StageTimer(device, enabled=True)
        with timer("proposer"):
            ...
        timer.totals()      # {"proposer": seconds}

    `totals()` synchronizes once. Call it after generation, never inside it.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        enabled: bool = False,
        sync_devices: list[torch.device] | None = None,
    ):
        self.enabled = bool(enabled)
        self.device = device
        # Work spread over several cards (the cross-model sampler) cannot be
        # timed with one device's events. With `sync_devices`, every stage
        # boundary synchronizes all of them and is timed on the host clock:
        # exact, at the cost of the stalls the event path avoids -- acceptable
        # because profiling is always its own generation, never a timed repeat.
        self.sync_devices = [d for d in (sync_devices or []) if d.type == "cuda"]
        self._cuda = (
            bool(enabled) and not self.sync_devices
            and device is not None and device.type == "cuda"
        )
        # stage -> list of (start_event, end_event) on CUDA, or accumulated
        # seconds on CPU.
        self._events: dict[str, list[tuple[Any, Any]]] = {}
        self._seconds: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._counts[stage] = self._counts.get(stage, 0) + 1
        if self._cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.setdefault(stage, []).append((start, end))
        else:
            for sync_device in self.sync_devices:
                torch.cuda.synchronize(sync_device)
            began = time.perf_counter()
            try:
                yield
            finally:
                for sync_device in self.sync_devices:
                    torch.cuda.synchronize(sync_device)
                self._seconds[stage] = (
                    self._seconds.get(stage, 0.0) + time.perf_counter() - began
                )

    def merge(self, other: "StageTimer") -> None:
        """Fold another timer's records in. Used to roll blocks up into a run."""
        if not self.enabled or not other.enabled:
            return
        for stage, pairs in other._events.items():
            self._events.setdefault(stage, []).extend(pairs)
        for stage, value in other._seconds.items():
            self._seconds[stage] = self._seconds.get(stage, 0.0) + value
        for stage, value in other._counts.items():
            self._counts[stage] = self._counts.get(stage, 0) + value

    def totals(self) -> dict[str, float]:
        """Seconds per stage. Synchronizes once when CUDA events were used."""
        if not self.enabled:
            return {}
        out = dict(self._seconds)
        if self._events:
            torch.cuda.synchronize(self.device)
            for stage, pairs in self._events.items():
                # elapsed_time is milliseconds.
                total = sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
                out[stage] = out.get(stage, 0.0) + total
        return out

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


NULL_TIMER = StageTimer(None, enabled=False)
