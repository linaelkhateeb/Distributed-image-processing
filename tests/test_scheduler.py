"""Coordinator logic without networking: queue, results, duplicates, merging."""

import numpy as np
import pytest

from common.filters import get_filter
from common.image_io import make_sample_image
from common.tiling import crop_halo
from coordinator.job import JobState
from coordinator.scheduler import Scheduler

IMAGE = make_sample_image(100, 130, seed=7)


def work(scheduler, worker_id):
    """Take one task and process it like a worker would. Returns (job, tile, core) or None."""
    assigned = scheduler.request_task(worker_id)
    if assigned is None:
        return None
    job, tile = assigned
    core = crop_halo(get_filter(job.filter_name)(tile.data), tile)
    return job, tile, core


def test_job_completes_and_matches_sequential():
    s = Scheduler()
    s.register_worker("w1", "host")
    job = s.submit_job(IMAGE, "blur", tile_size=32)
    assert job.total_tiles == 4 * 5

    while (item := work(s, "w1")) is not None:
        j, tile, core = item
        assert s.submit_result("w1", j.job_id, tile.index, core)

    assert job.state == JobState.DONE
    assert np.array_equal(job.result, get_filter("blur")(IMAGE))
    assert job.tiles_per_worker == {"w1": job.total_tiles}
    assert s.workers["w1"].tiles_done == job.total_tiles


def test_tiles_are_handed_out_once_in_order():
    s = Scheduler()
    job = s.submit_job(IMAGE, "invert", tile_size=50)
    indices = []
    while (assigned := s.request_task("w")) is not None:
        indices.append(assigned[1].index)
    assert indices == list(range(job.total_tiles))


def test_two_workers_share_the_queue():
    s = Scheduler()
    job = s.submit_job(IMAGE, "sharpen", tile_size=25)
    turn = 0
    while True:
        worker = ("a", "b")[turn % 2]
        item = work(s, worker)
        if item is None:
            break
        s.submit_result(worker, job.job_id, item[1].index, item[2])
        turn += 1
    assert job.state == JobState.DONE
    assert set(job.tiles_per_worker) == {"a", "b"}
    assert np.array_equal(job.result, get_filter("sharpen")(IMAGE))


def test_duplicate_result_is_ignored():
    s = Scheduler()
    job = s.submit_job(IMAGE, "invert", tile_size=200)  # one tile
    _, tile, core = work(s, "w1")
    assert s.submit_result("w1", job.job_id, tile.index, core) is True
    assert s.submit_result("w2", job.job_id, tile.index, core) is False
    assert job.tiles_per_worker == {"w1": 1}


def test_result_for_unknown_job_or_bad_tile_is_rejected():
    s = Scheduler()
    job = s.submit_job(IMAGE, "invert", tile_size=50)
    core = np.zeros((50, 50, 3), np.uint8)
    assert s.submit_result("w", "nope", 0, core) is False
    assert s.submit_result("w", job.job_id, 999, core) is False
    assert s.submit_result("w", job.job_id, 0, np.zeros((3, 3, 3), np.uint8)) is False


def test_jobs_are_processed_in_submission_order():
    s = Scheduler()
    first = s.submit_job(IMAGE, "invert", tile_size=200)
    second = s.submit_job(IMAGE, "invert", tile_size=200)
    assert s.request_task("w")[0] is first
    assert s.request_task("w")[0] is second


@pytest.mark.parametrize("filter_name, tile_size", [("nope", 10), ("blur", 0)])
def test_invalid_job_is_rejected(filter_name, tile_size):
    with pytest.raises(ValueError):
        Scheduler().submit_job(IMAGE, filter_name, tile_size)


def test_heartbeat_from_unknown_worker():
    s = Scheduler()
    assert s.heartbeat("ghost") is False
    s.register_worker("w", "host")
    assert s.heartbeat("w") is True
