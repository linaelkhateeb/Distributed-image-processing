"""End to end over real gRPC: a coordinator server and several workers in this process.

Workers run as threads here for speed; in real use each is a separate process or machine, but
they talk to the coordinator through exactly the same gRPC calls.
"""

import threading

import numpy as np
import pytest

from client.cli import JobFailed, connect, run_job
from common.filters import get_filter
from common.image_io import make_sample_image
from common.rpc import pb
from coordinator.server import serve
from worker.worker import Worker

IMAGE = make_sample_image(300, 410, seed=8)


@pytest.fixture(params=[1, 0], ids=["prefetch1", "prefetch0"])
def cluster(request):
    """Coordinator + 3 workers; runs every test with and without pipelining."""
    server, scheduler, port = serve(port=0)
    stop = threading.Event()
    workers = [Worker(f"localhost:{port}", f"w{i}", prefetch=request.param) for i in range(3)]
    threads = [threading.Thread(target=w.run, args=(stop,), daemon=True) for w in workers]
    for t in threads:
        t.start()
    stub = connect(f"localhost:{port}")
    yield stub, scheduler, workers
    stop.set()
    for t in threads:
        t.join(timeout=5)
    server.stop(grace=None)
    scheduler.stop()


@pytest.mark.parametrize("filter_name", ["grayscale", "blur", "sobel", "median_heavy"])
def test_distributed_result_matches_sequential(cluster, filter_name):
    stub, _, _ = cluster
    result, status = run_job(stub, IMAGE, filter_name, tile_size=64)
    assert np.array_equal(result, get_filter(filter_name)(IMAGE))
    assert status.state == "DONE"
    assert status.done_tiles == status.total_tiles == 5 * 7
    assert sum(status.tiles_per_worker.values()) == status.total_tiles


def test_several_jobs_back_to_back(cluster):
    stub, _, _ = cluster
    for tile_size in (50, 128, 1000):
        result, _ = run_job(stub, IMAGE, "sharpen", tile_size)
        assert np.array_equal(result, get_filter("sharpen")(IMAGE))


def test_invalid_job_is_reported(cluster):
    stub, _, _ = cluster
    with pytest.raises(JobFailed, match="Unknown filter"):
        run_job(stub, IMAGE, "nope", 64)


def test_workers_are_listed(cluster):
    stub, _, _ = cluster
    run_job(stub, IMAGE, "invert", 64)
    reply = stub.ListWorkers(pb.ListWorkersRequest())
    assert {w.worker_id for w in reply.workers} == {"w0", "w1", "w2"}
    assert all(w.alive for w in reply.workers)
    assert sum(w.tiles_done for w in reply.workers) == 5 * 7
