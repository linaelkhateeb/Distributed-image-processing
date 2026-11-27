"""End to end over real gRPC: workers crash or fail mid-job and the job still finishes correctly.

Workers run as threads; a "crash" makes one worker stop instantly, without sending its result or
any more heartbeats, which is what the coordinator sees when a real worker process dies.
"""

import threading
import time

import numpy as np
import pytest

from client.cli import JobFailed, connect, run_job
from common.filters import get_filter
from common.image_io import make_sample_image
from common.rpc import pb
from coordinator.scheduler import Scheduler
from coordinator.server import serve
from worker.worker import Worker, WorkerCrashed

IMAGE = make_sample_image(300, 410, seed=10)
FILTER = "median"
EXPECTED = get_filter(FILTER)(IMAGE)


class CrashingWorker(Worker):
    """Dies abruptly (in-process) after ``crash_after`` tiles."""

    def crash(self):
        raise WorkerCrashed()


class FailingWorker(Worker):
    """Reports an error for every tile it gets (e.g. a machine missing a library)."""

    def process(self, reply):
        return pb.TaskResult(worker_id=self.worker_id, task_id=reply.task_id,
                             job_id=reply.job_id, tile_index=reply.tile.index,
                             error="RuntimeError: simulated failure")


@pytest.fixture
def make_cluster():
    """Start a coordinator with fast failure detection and the given workers."""
    started = []

    def start(workers_spec):
        scheduler = Scheduler(heartbeat_interval=0.2, heartbeat_timeout=1.0,
                              check_interval=0.1, max_attempts=3)
        server, scheduler, port = serve(port=0, scheduler=scheduler)
        stop = threading.Event()
        workers = [cls(f"localhost:{port}", name, **kw) for cls, name, kw in workers_spec]
        threads = [threading.Thread(target=w.run, args=(stop,), daemon=True) for w in workers]
        for t in threads:
            t.start()
        started.append((server, scheduler, stop, threads))
        return connect(f"localhost:{port}"), scheduler, workers, port

    yield start
    for server, scheduler, stop, threads in started:
        stop.set()
        for t in threads:
            t.join(timeout=5)
        server.stop(grace=None)
        scheduler.stop()


def test_job_survives_a_worker_crash(make_cluster):
    stub, scheduler, workers, _ = make_cluster([
        (CrashingWorker, "crasher", {"crash_after": 3, "slow": 0.02}),
        (Worker, "w1", {"slow": 0.02}),
        (Worker, "w2", {"slow": 0.02}),
    ])
    result, status = run_job(stub, IMAGE, FILTER, tile_size=40)  # 8 x 11 = 88 tiles

    assert np.array_equal(result, EXPECTED)
    assert status.reassigned_tiles >= 1  # the tile(s) the crasher was holding
    # It finished 3 tiles, but with pipelining the last result may still have been in flight
    # when it died; that result is lost and the tile is re-sent, so 2 or 3 arrive.
    assert 2 <= status.tiles_per_worker["crasher"] <= 3
    assert workers[0].crashed.is_set()

    time.sleep(0.3)
    alive = {w.worker_id: w.alive for w in stub.ListWorkers(pb.ListWorkersRequest()).workers}
    assert alive == {"crasher": False, "w1": True, "w2": True}


def test_job_survives_all_but_one_worker_crashing(make_cluster):
    stub, _, _, _ = make_cluster([
        (CrashingWorker, "c1", {"crash_after": 1, "slow": 0.02}),
        (CrashingWorker, "c2", {"crash_after": 2, "slow": 0.02}),
        (CrashingWorker, "c3", {"crash_after": 4, "slow": 0.02}),
        (Worker, "survivor", {"slow": 0.01}),
    ])
    result, status = run_job(stub, IMAGE, FILTER, tile_size=60)
    assert np.array_equal(result, EXPECTED)
    assert status.reassigned_tiles >= 3


def test_tiles_failing_on_one_worker_are_done_by_another(make_cluster):
    stub, _, _, _ = make_cluster([
        (FailingWorker, "broken", {}),
        (Worker, "ok", {"slow": 0.01}),
    ])
    result, status = run_job(stub, IMAGE, FILTER, tile_size=60)
    assert np.array_equal(result, EXPECTED)
    assert set(status.tiles_per_worker) == {"ok"}


def test_job_fails_cleanly_when_every_worker_fails(make_cluster):
    stub, _, _, _ = make_cluster([(FailingWorker, "b1", {}), (FailingWorker, "b2", {})])
    with pytest.raises(JobFailed, match="failed 3 times"):
        run_job(stub, IMAGE, FILTER, tile_size=60, timeout=30)


def test_job_waits_until_a_worker_joins(make_cluster):
    """Submitted with no workers: tiles wait in the queue until a worker connects."""
    stub, _, _, port = make_cluster([])
    stop = threading.Event()
    late = Worker(f"localhost:{port}", "late")

    def join_later():
        time.sleep(0.5)
        late.run(stop)

    t = threading.Thread(target=join_later, daemon=True)
    t.start()
    try:
        result, status = run_job(stub, IMAGE, FILTER, tile_size=100, timeout=30)
    finally:
        stop.set()
        t.join(timeout=5)
    assert np.array_equal(result, EXPECTED)
    assert dict(status.tiles_per_worker) == {"late": status.total_tiles}
