"""The coordinator's brain: jobs, the task queue, workers and results.

All state lives here, guarded by one lock, so the gRPC server (which handles calls on many
threads at once) stays thin and the logic can be unit-tested without any networking.

Scheduling is PULL-based: tiles wait in a FIFO queue and each idle worker takes the next one when
it calls ``request_task``. A fast worker simply comes back more often and ends up doing more
tiles, which balances the load without the coordinator knowing anything about worker speed.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

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


@dataclass
class Assignment:
    worker_id: str
    assigned_at: float  # monotonic clock


class Scheduler:
    def __init__(self, heartbeat_interval: float = 2.0, heartbeat_timeout: float = 6.0):
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._lock = threading.Lock()
        self.jobs: Dict[str, Job] = {}
        self.workers: Dict[str, WorkerInfo] = {}
        self._pending: Deque[Tuple[str, int]] = deque()  # (job_id, tile_index)
        self._assigned: Dict[str, Assignment] = {}  # task_id -> who has it

    # ------------------------------------------------------------------ workers

    def register_worker(self, worker_id: str, hostname: str) -> None:
        with self._lock:
            self.workers[worker_id] = WorkerInfo(worker_id, hostname, time.perf_counter())
        log.info("Worker %s registered (host %s)", worker_id, hostname)

    def heartbeat(self, worker_id: str) -> bool:
        """Record that a worker is alive. Returns False if the worker is unknown."""
        with self._lock:
            info = self.workers.get(worker_id)
            if info is None:
                return False
            info.last_heartbeat = time.perf_counter()
            return True

    def list_workers(self) -> List[Tuple[WorkerInfo, float, bool]]:
        """(info, seconds since last heartbeat, considered alive)."""
        now = time.perf_counter()
        with self._lock:
            return [
                (w, now - w.last_heartbeat, now - w.last_heartbeat < self.heartbeat_timeout)
                for w in self.workers.values()
            ]

    # --------------------------------------------------------------------- jobs

    def submit_job(self, image: np.ndarray, filter_name: str, tile_size: int) -> Job:
        if filter_name not in FILTERS:
            raise ValueError(f"Unknown filter {filter_name!r}. Available: {', '.join(FILTERS)}")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")

        start = time.perf_counter()
        tiles = split(image, tile_size, get_filter(filter_name).radius)
        job = Job(
            job_id=uuid.uuid4().hex[:8],
            filter_name=filter_name,
            tile_size=tile_size,
            image_shape=image.shape,
            tiles=tiles,
            created_at=start,
        )
        job.split_seconds = time.perf_counter() - start
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
            if worker_id in self.workers:
                # Asking for work is also a sign of life.
                self.workers[worker_id].last_heartbeat = time.perf_counter()
            while self._pending:
                job_id, index = self._pending.popleft()
                job = self.jobs.get(job_id)
                if job is None or job.state != JobState.RUNNING or index in job.cores:
                    continue  # job gone, or this tile was already finished by someone else
                self._assigned[job.task_id(index)] = Assignment(worker_id, time.perf_counter())
                return job, job.tiles[index]
            return None

    def submit_result(
        self, worker_id: str, job_id: str, tile_index: int, core: np.ndarray
    ) -> bool:
        """Store a filtered tile. Returns False (and ignores it) if it's a duplicate.

        When the last tile of a job arrives, the job is merged on this thread, outside the lock,
        so other workers can keep pulling tasks meanwhile.
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or job.state != JobState.RUNNING:
                return False
            if not 0 <= tile_index < job.total_tiles:
                return False
            if tile_index in job.cores:
                log.info("Duplicate result for %s from %s ignored",
                         job.task_id(tile_index), worker_id)
                return False
            tile = job.tiles[tile_index]
            if core.shape[:2] != (tile.height, tile.width):
                log.warning("Result for %s has wrong shape %s", job.task_id(tile_index),
                            core.shape)
                return False

            job.cores[tile_index] = core
            job.tiles_per_worker[worker_id] += 1
            job.last_result_at = time.perf_counter()
            self._assigned.pop(job.task_id(tile_index), None)
            if worker_id in self.workers:
                self.workers[worker_id].tiles_done += 1

            should_merge = job.done_tiles == job.total_tiles and not job.merging
            if should_merge:
                job.merging = True

        if should_merge:
            self._merge(job)
        return True

    def _merge(self, job: Job) -> None:
        start = time.perf_counter()
        try:
            result = merge(job.image_shape, ((job.tiles[i], c) for i, c in job.cores.items()))
        except Exception as exc:  # pragma: no cover - would be a bug in tiling
            with self._lock:
                job.state = JobState.FAILED
                job.error = f"Merge failed: {exc}"
            log.exception("Job %s merge failed", job.job_id)
            return
        end = time.perf_counter()
        with self._lock:
            job.result = result
            job.merge_seconds = end - start
            job.finished_at = end
            job.state = JobState.DONE
            job.cores.clear()  # free memory; result holds the pixels now
            for t in job.tiles:
                t.data = None  # type: ignore[assignment]
        log.info(
            "Job %s done in %.3f s (processing %.3f s, merge %.1f ms); tiles per worker: %s",
            job.job_id, job.total_seconds, job.processing_seconds, job.merge_seconds * 1000,
            dict(job.tiles_per_worker),
        )
