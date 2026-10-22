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

Stages (block samplers; every one is optional and none nests in another):
  prefill          the prompt's whole blocks into the KV cache (both models)
  propose_forward  the proposer's model call: body plus output head
  propose_rank     log-softmax, token choice, confidence sort
  verify_forward   one verifier model call per group of rows (MI only)
  verify_test      handing candidates to the verifier, building its probe
                   rows, the log-ratio prefix test (MI only)
  decide           writing the accepted tokens into the block
  stop_check       the early-stop test after each step
  commit           a finished block's K/V into the cache (both models)
  remask           the remask pass (xadaptive-remask only)
What no stage covers -- the Python loop -- is the summary's "residual". The
SDAR-only adaptive sampler still times its verifier as one "verifier" stage.
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
        # Work spread over several cards (the cross-model sampler, or one model
        # pipeline-split over two) cannot be timed with one device's events. With `sync_devices`, every stage
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
    def __call__(self, stage: str, count: bool = True) -> Iterator[None]:
        """Time one span of `stage`. `count=False` adds the time without
        counting a call, for a stage timed in pieces (the proposer's forward
        and its output head) that is still one call."""
        if not self.enabled:
            yield
            return
        if count:
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


class SpanClock:
    """Wall time of one span of GPU work (the prompt prefill), always on.

    Unlike `StageTimer` this runs inside the timed repeats, so it must not
    synchronize: it records a CUDA event pair on every card the span touches
    and reads them only in `seconds()`, after the generation has finished.
    The span is the slowest card's, start to end; the events are recorded at
    the same host instant on streams that are idle or busy with the same
    request, so their starts line up. Host clock on CPU.
    """

    def __init__(self, devices: list[torch.device]):
        self.devices = list(dict.fromkeys(d for d in devices if d.type == "cuda"))
        self._pairs: list[tuple[Any, Any]] = []
        self._host: list[float] = []

    def start(self) -> None:
        if self.devices:
            self._pairs = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in self.devices
            ]
            for device, (begin, _) in zip(self.devices, self._pairs):
                begin.record(torch.cuda.current_stream(device))
        else:
            self._host = [time.perf_counter()]

    def stop(self) -> None:
        if self.devices:
            for device, (_, end) in zip(self.devices, self._pairs):
                end.record(torch.cuda.current_stream(device))
        else:
            self._host.append(time.perf_counter())

    def seconds(self) -> float | None:
        """Elapsed seconds; waits for the end events, so call it after generation."""
        if self._pairs:
            for _, end in self._pairs:
                end.synchronize()
            return max(begin.elapsed_time(end) for begin, end in self._pairs) / 1000.0
        if len(self._host) == 2:
            return self._host[1] - self._host[0]
        return None
