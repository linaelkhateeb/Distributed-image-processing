"""The coordinator's brain: jobs, the task queue, workers, results and failure handling.

All state lives here, guarded by one lock, so the gRPC server (which handles calls on many
threads at once) stays thin and the logic can be unit-tested without any networking.

Scheduling is PULL-based: tiles wait in a FIFO queue and each idle worker takes the next one when
it calls ``request_task``. A fast worker simply comes back more often and ends up doing more
tiles, which balances the load without the coordinator knowing anything about worker speed.

Fault tolerance:
- Failure detector: every worker sends heartbeats. A worker silent for longer than
  ``heartbeat_timeout`` is *suspected* dead (in an asynchronous network a slow worker and a dead
  one look the same, so this is a suspicion, not a certainty). All tiles it holds go back to the
  front of the queue for other workers.
- Lease: a tile held for longer than ``lease_timeout`` is re-queued even if its worker still sends
  heartbeats (worker stuck or the tile lost).
- Idempotent tiles: a tile keeps the same task id when re-sent. The first result to arrive is
  stored; any later copy (e.g. from a worker wrongly suspected dead) is ignored. Processing a tile
  twice gives the same pixels, so this is always safe (at-least-once delivery).
- Retry limit: a tile handed out ``max_attempts`` times without a result fails its job, so one bad
  tile can't loop forever.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from common.filters import FILTERS, get_filter
from common.tiling import Tile, merge, split
from coordinator.job import Job, JobState

log = logging.getLogger("coordinator")


@dataclass
class WorkerInfo:
    worker_id: str
    hostname: str
    last_heartbeat: float  # monotonic clock
    tiles_done: int = 0
    alive: bool = True  # False = suspected dead by the failure detector


@dataclass
class Assignment:
    job_id: str
    tile_index: int
    worker_id: str
    assigned_at: float  # monotonic clock


class Scheduler:
    def __init__(
        self,
        heartbeat_interval: float = 2.0,
        heartbeat_timeout: float = 6.0,
        lease_timeout: float = 30.0,
        max_attempts: int = 3,
        check_interval: Optional[float] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.lease_timeout = lease_timeout
        self.max_attempts = max_attempts
        self.check_interval = check_interval or min(1.0, heartbeat_timeout / 4)
        self.clock = clock  # injectable so tests can fake the passage of time
        self._lock = threading.Lock()
        self.jobs: Dict[str, Job] = {}
        self.workers: Dict[str, WorkerInfo] = {}
        self._pending: Deque[Tuple[str, int]] = deque()  # (job_id, tile_index)
        self._assigned: Dict[str, Assignment] = {}  # task_id -> current holder
        self._monitor_stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ monitor

    def start_monitor(self) -> None:
        """Run the failure detector in a background thread."""
        if self._monitor is not None:
            return
        self._monitor = threading.Thread(target=self._monitor_loop, name="failure-detector",
                                         daemon=True)
        self._monitor.start()

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=5)

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self.check_interval):
            try:
                self.check_failures()
            except Exception:  # pragma: no cover - never let the detector thread die
                log.exception("Failure check crashed")

    def check_failures(self) -> None:
        """Suspect silent workers, re-queue their tiles and any tile whose lease expired."""
        now = self.clock()
        with self._lock:
            for w in self.workers.values():
                if w.alive and now - w.last_heartbeat > self.heartbeat_timeout:
                    w.alive = False
                    held = sorted(
                        (a for a in self._assigned.values() if a.worker_id == w.worker_id),
                        key=lambda a: (a.job_id, a.tile_index),
                    )
                    log.warning(
                        "Worker %s suspected dead (no heartbeat for %.1f s); "
                        "re-queuing its %d tile(s)",
                        w.worker_id, now - w.last_heartbeat, len(held),
                    )
                    # Each re-queue goes to the front, so add them last-first to keep order.
                    for a in reversed(held):
                        self._requeue(a, reason=f"worker {w.worker_id} suspected dead")

            for a in list(self._assigned.values()):
                if now - a.assigned_at > self.lease_timeout:
                    self._requeue(a, reason=f"lease expired ({now - a.assigned_at:.1f} s "
                                            f"with {a.worker_id})")

    def _requeue(self, a: Assignment, reason: str) -> None:
        """Put a tile back at the FRONT of the queue (lock must be held)."""
        job = self.jobs.get(a.job_id)
        self._assigned.pop(job.task_id(a.tile_index) if job else "", None)
        if job is None or job.state != JobState.RUNNING or a.tile_index in job.cores:
            return
        if job.attempts[a.tile_index] >= self.max_attempts:
            self._fail_job(job, f"Tile {a.tile_index} failed {job.attempts[a.tile_index]} "
                                f"times (last: {reason})")
            return
        job.reassigned_tiles += 1
        self._pending.appendleft((a.job_id, a.tile_index))
        log.info("Re-queued %s: %s", job.task_id(a.tile_index), reason)

    def _fail_job(self, job: Job, error: str) -> None:
        """Mark a job failed (lock must be held). Its leftover queue entries are skipped."""
        job.state = JobState.FAILED
        job.error = error
        job.finished_at = self.clock()
        for task_id in [t for t, a in self._assigned.items() if a.job_id == job.job_id]:
            del self._assigned[task_id]
        log.error("Job %s FAILED: %s", job.job_id, error)

    # ------------------------------------------------------------------ workers

    def register_worker(self, worker_id: str, hostname: str) -> None:
        with self._lock:
            previous = self.workers.get(worker_id)
            info = WorkerInfo(worker_id, hostname, self.clock())
            if previous is not None:
                info.tiles_done = previous.tiles_done
            self.workers[worker_id] = info
        log.info("Worker %s %s (host %s)", worker_id,
                 "re-registered" if previous else "registered", hostname)

    def _touch(self, worker_id: str) -> bool:
        """Record a sign of life (lock must be held). Returns False if the worker is unknown."""
        info = self.workers.get(worker_id)
        if info is None:
            return False
        info.last_heartbeat = self.clock()
        if not info.alive:
            info.alive = True
            log.warning("Worker %s is back (it was only slow or cut off, not dead)", worker_id)
        return True

    def heartbeat(self, worker_id: str) -> bool:
        """Record that a worker is alive. Returns False if the worker is unknown."""
        with self._lock:
            return self._touch(worker_id)

    def list_workers(self) -> List[Tuple[WorkerInfo, float, bool]]:
        """(info, seconds since last heartbeat, considered alive)."""
        now = self.clock()
        with self._lock:
            return [(w, now - w.last_heartbeat, w.alive) for w in self.workers.values()]

    # --------------------------------------------------------------------- jobs

    def submit_job(self, image: np.ndarray, filter_name: str, tile_size: int) -> Job:
        if filter_name not in FILTERS:
            raise ValueError(f"Unknown filter {filter_name!r}. Available: {', '.join(FILTERS)}")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")

        start = self.clock()
        tiles = split(image, tile_size, get_filter(filter_name).radius)
        job = Job(
            job_id=uuid.uuid4().hex[:8],
            filter_name=filter_name,
            tile_size=tile_size,
            image_shape=image.shape,
            tiles=tiles,
            created_at=start,
        )
        job.split_seconds = self.clock() - start
        with self._lock:
            self.jobs[job.job_id] = job
            self._pending.extend((job.job_id, t.index) for t in tiles)
        h, w = image.shape[:2]
        log.info(
            "Job %s: %dx%d image, filter %s, tile %d -> %d tiles (split %.1f ms)",
            job.job_id, w, h, filter_name, tile_size, len(tiles), job.split_seconds * 1000,
        )
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    # -------------------------------------------------------------------- tasks

    def request_task(self, worker_id: str) -> Optional[Tuple[Job, Tile]]:
        """Hand the next pending tile to ``worker_id``, or None if there is nothing to do."""
        with self._lock:
            self._touch(worker_id)  # asking for work is also a sign of life
            skipped: List[Tuple[str, int]] = []
            try:
                while self._pending:
                    job_id, index = self._pending.popleft()
                    job = self.jobs.get(job_id)
                    if job is None or job.state != JobState.RUNNING or index in job.cores:
                        continue  # job gone/failed, or tile already finished by someone
                    if self._should_avoid(job, index, worker_id):
                        skipped.append((job_id, index))  # leave it for another worker
                        continue
                    job.attempts[index] += 1
                    self._assigned[job.task_id(index)] = Assignment(
                        job_id, index, worker_id, self.clock()
                    )
                    return job, job.tiles[index]
                return None
            finally:
                self._pending.extendleft(reversed(skipped))  # back at the front, same order

    def _should_avoid(self, job: Job, index: int, worker_id: str) -> bool:
        """Don't give a tile back to a worker that already failed it, as long as some other
        live worker hasn't failed it too. Otherwise one broken worker, which fails instantly and
        asks again first, would burn through all the tile's retries by itself."""
        failed = job.failed_on.get(index)
        if not failed or worker_id not in failed:
            return False
        return any(w.alive and w.worker_id not in failed for w in self.workers.values())

    def submit_result(
        self, worker_id: str, job_id: str, tile_index: int, core: np.ndarray
    ) -> bool:
        """Store a filtered tile. Returns False (and ignores it) if it's a duplicate or invalid.

        When the last tile of a job arrives, the job is merged on this thread, outside the lock,
        so other workers can keep pulling tasks meanwhile.
        """
        with self._lock:
            self._touch(worker_id)
            job = self.jobs.get(job_id)
            if job is None or job.state != JobState.RUNNING:
                return False
            if not 0 <= tile_index < job.total_tiles:
                return False
            task_id = job.task_id(tile_index)
            if tile_index in job.cores:
                log.info("Duplicate result for %s from %s ignored", task_id, worker_id)
                return False
            tile = job.tiles[tile_index]
            if core.shape[:2] != (tile.height, tile.width):
                log.warning("Result for %s has wrong shape %s", task_id, core.shape)
                return False

            holder = self._assigned.pop(task_id, None)
            if holder is not None and holder.worker_id != worker_id:
                # The tile was re-sent to someone else, but the original worker finished first.
                log.info("Result for %s from %s arrived before the re-sent copy (%s); using it",
                         task_id, worker_id, holder.worker_id)
            job.cores[tile_index] = core
            job.tiles_per_worker[worker_id] += 1
            job.last_result_at = self.clock()
            if worker_id in self.workers:
                self.workers[worker_id].tiles_done += 1

            should_merge = job.done_tiles == job.total_tiles and not job.merging
            if should_merge:
                job.merging = True

        if should_merge:
            self._merge(job)
        return True

    def report_failure(self, worker_id: str, job_id: str, tile_index: int, error: str) -> None:
        """A worker could not process a tile. Re-queue it (or fail the job after max retries)."""
        with self._lock:
            self._touch(worker_id)
            job = self.jobs.get(job_id)
            if job is None:
                return
            holder = self._assigned.get(job.task_id(tile_index))
            if holder is None or holder.worker_id != worker_id:
                return  # already re-queued or finished elsewhere
            log.warning("Worker %s failed %s: %s", worker_id, job.task_id(tile_index), error)
            job.failed_on.setdefault(tile_index, set()).add(worker_id)
            self._requeue(holder, reason=f"error on {worker_id}: {error}")

    def _merge(self, job: Job) -> None:
        start = self.clock()
        try:
            result = merge(job.image_shape, ((job.tiles[i], c) for i, c in job.cores.items()))
        except Exception as exc:  # pragma: no cover - would be a bug in tiling
            with self._lock:
                self._fail_job(job, f"Merge failed: {exc}")
            return
        end = self.clock()
        with self._lock:
            job.result = result
            job.merge_seconds = end - start
            job.finished_at = end
            job.state = JobState.DONE
            job.cores.clear()  # free memory; result holds the pixels now
            for t in job.tiles:
                t.data = None  # type: ignore[assignment]
        log.info(
            "Job %s done in %.3f s (processing %.3f s, merge %.1f ms, %d tile(s) re-queued); "
            "tiles per worker: %s",
            job.job_id, job.total_seconds, job.processing_seconds, job.merge_seconds * 1000,
            job.reassigned_tiles, dict(job.tiles_per_worker),
        )
