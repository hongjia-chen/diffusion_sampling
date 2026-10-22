"""Keep other processes' GPU work out of the timed runs.

The host's cards are shared. A process that lands on a card we time on slows
every forward (x1.2-1.5 on the SDAR-8B-b32 baselines, 2026-10-14), and nothing
in a record said so: the audit had to infer it from the timings. The guard makes
it explicit. Before each timed run it waits until no other process holds any of
our cards. While the run is timed, a sampler polls the cards every few seconds.
A run that saw another process is discarded and redone once the cards are
clear. Each record keeps what the guard saw, plus the host's load average, the
other way an eager-mode decoder gets slowed (CPU contention).

"Other" means any process but this one, whoever owns it: a second run of ours
on the same card contends just as much.
"""

from __future__ import annotations

import os
import pwd
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

Query = Callable[..., str]


def nvidia_smi(*args: str) -> str:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30,
    ).stdout


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    user: str
    gpu: str              # card UUID
    used_mib: int | None


def parse_compute_apps(text: str) -> list[tuple[int, str, int | None]]:
    """`--query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits`."""
    apps: list[tuple[int, str, int | None]] = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        memory = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        apps.append((int(parts[0]), parts[1], memory))
    return apps


def process_owner(pid: int) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (FileNotFoundError, KeyError, PermissionError):
        return "unknown"


def other_processes(
    uuids: list[str], *, self_pid: int, query: Query = nvidia_smi,
) -> list[GpuProcess]:
    """Every process but `self_pid` that holds a context on one of `uuids`."""
    wanted = set(uuids)
    text = query(
        "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits",
    )
    return [
        GpuProcess(pid=pid, user=process_owner(pid), gpu=uuid, used_mib=memory)
        for pid, uuid, memory in parse_compute_apps(text)
        if uuid in wanted and pid != self_pid
    ]


def describe(processes: list[GpuProcess], uuids: list[str]) -> str:
    names = {uuid: f"card {index}" for index, uuid in enumerate(uuids)}
    return ", ".join(
        f"{names.get(p.gpu, p.gpu)} pid {p.pid} ({p.user}"
        + (f", {p.used_mib} MiB)" if p.used_mib is not None else ")")
        for p in processes
    )


@dataclass
class Watch:
    """What one timed run saw: processes (by pid) and the host's 1-min load."""

    processes: dict[int, GpuProcess] = field(default_factory=dict)
    load_before: float | None = None
    load_after: float | None = None
    probe_failures: int = 0

    @property
    def contended(self) -> bool:
        return bool(self.processes)

    def record(self) -> dict:
        return {
            "contention_processes": [asdict(p) for p in self.processes.values()],
            "host_load_1min": [self.load_before, self.load_after],
            "contention_probe_failures": self.probe_failures,
        }


class ContentionGuard:
    def __init__(
        self,
        uuids: list[str],
        *,
        poll_seconds: float = 30.0,
        sample_seconds: float = 5.0,
        max_retries: int = 10,
        report_every_seconds: float = 600.0,
        self_pid: int | None = None,
        query: Query = nvidia_smi,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = lambda message: print(message, flush=True),
    ):
        self.uuids = uuids
        self.poll_seconds = poll_seconds
        self.sample_seconds = sample_seconds
        self.max_retries = max_retries
        self.report_every_seconds = report_every_seconds
        self.self_pid = os.getpid() if self_pid is None else self_pid
        self._query = query
        self._sleep = sleep
        self._clock = clock
        self._log = log

    def settings(self) -> dict:
        return {
            "mode": "wait", "poll_seconds": self.poll_seconds,
            "sample_seconds": self.sample_seconds, "max_retries": self.max_retries,
            "cards": self.uuids,
        }

    def probe(self) -> list[GpuProcess] | None:
        """Other processes on our cards now, or None if nvidia-smi failed."""
        try:
            return other_processes(self.uuids, self_pid=self.self_pid, query=self._query)
        except (subprocess.SubprocessError, OSError, ValueError):
            return None

    def wait_until_clear(self) -> float:
        """Block until no other process holds our cards; return seconds waited.

        A failing nvidia-smi does not block: the run goes on, and the watch
        around the timed run counts the failure in the record.
        """
        began = self._clock()
        last_report = None
        while True:
            seen = self.probe()
            if not seen:
                waited = self._clock() - began
                if last_report is not None:
                    self._log(f"[guard] cards clear after {waited / 60:.1f} min")
                return waited
            now = self._clock()
            if last_report is None or now - last_report >= self.report_every_seconds:
                self._log(
                    f"[guard] waiting ({(now - began) / 60:.0f} min so far): "
                    f"{describe(seen, self.uuids)}"
                )
                last_report = now
            self._sleep(self.poll_seconds)

    @contextmanager
    def watch(self) -> Iterator[Watch]:
        """Sample our cards around and during the block. Probes at entry and exit
        run before the caller starts its clock and after it stops it."""
        result = Watch(load_before=os.getloadavg()[0])

        def note(seen: list[GpuProcess] | None) -> None:
            if seen is None:
                result.probe_failures += 1
                return
            for process in seen:
                result.processes.setdefault(process.pid, process)

        note(self.probe())
        stop = threading.Event()

        def sampler() -> None:
            while not stop.wait(self.sample_seconds):
                note(self.probe())

        thread = threading.Thread(target=sampler, name="contention-sampler", daemon=True)
        thread.start()
        try:
            yield result
        finally:
            stop.set()
            thread.join()
            note(self.probe())
            result.load_after = os.getloadavg()[0]


class NullGuard:
    """--contention-guard off: never waits, never samples."""

    max_retries = 0

    def settings(self) -> dict:
        return {"mode": "off"}

    def wait_until_clear(self) -> float:
        return 0.0

    @contextmanager
    def watch(self) -> Iterator[Watch]:
        result = Watch(load_before=os.getloadavg()[0])
        yield result
        result.load_after = os.getloadavg()[0]
