"""Failure handling in the coordinator, with a fake clock so tests are instant and exact."""

import numpy as np
import pytest

from common.filters import get_filter
from common.image_io import make_sample_image
from common.tiling import crop_halo
from coordinator.job import JobState
from coordinator.scheduler import Scheduler

IMAGE = make_sample_image(90, 120, seed=9)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def sched(clock):
    s = Scheduler(heartbeat_timeout=6.0, lease_timeout=30.0, max_attempts=3, clock=clock)
    s.register_worker("w1", "h")
    s.register_worker("w2", "h")
    return s


def filtered_core(job, tile):
    return crop_halo(get_filter(job.filter_name)(tile.data), tile)


def finish(sched, worker_id):
    """Let one worker process everything left in the queue."""
    while (assigned := sched.request_task(worker_id)) is not None:
        job, tile = assigned
        sched.submit_result(worker_id, job.job_id, tile.index, filtered_core(job, tile))


# ---------------------------------------------------------------- failure detector

def test_silent_worker_is_suspected_and_its_tiles_requeued(sched, clock):
    job = sched.submit_job(IMAGE, "blur", tile_size=30)  # 3 x 4 = 12 tiles
    held = [sched.request_task("w1")[1].index for _ in range(2)]  # w1 takes tiles 0, 1

    clock.advance(5)
    sched.heartbeat("w2")
    sched.check_failures()
    assert sched.workers["w1"].alive  # 5 s < 6 s timeout: not yet suspected

    clock.advance(2)  # w1 silent for 7 s
    sched.heartbeat("w2")
    sched.check_failures()
    assert not sched.workers["w1"].alive
    assert sched.workers["w2"].alive
    assert job.reassigned_tiles == 2

    # The re-queued tiles go to the FRONT of the queue, in order, so w2 gets them next.
    for expected_index in held:
        j, tile = sched.request_task("w2")
        assert tile.index == expected_index
        sched.submit_result("w2", j.job_id, tile.index, filtered_core(j, tile))

    finish(sched, "w2")
    assert job.state == JobState.DONE
    assert np.array_equal(job.result, get_filter("blur")(IMAGE))


def test_heartbeating_worker_is_not_suspected(sched, clock):
    sched.submit_job(IMAGE, "blur", tile_size=30)
    sched.request_task("w1")
    for _ in range(10):
        clock.advance(2)
        sched.heartbeat("w1")
        sched.heartbeat("w2")
        sched.check_failures()
    assert sched.workers["w1"].alive


def test_suspected_worker_that_comes_back_is_alive_again(sched, clock):
    clock.advance(10)
    sched.check_failures()
    assert not sched.workers["w1"].alive
    assert sched.heartbeat("w1")
    assert sched.workers["w1"].alive


# ------------------------------------------------------------------- idempotence

def test_late_result_from_suspected_worker_is_used_if_it_arrives_first(sched, clock):
    job = sched.submit_job(IMAGE, "invert", tile_size=200)  # one tile
    _, tile = sched.request_task("w1")
    core = filtered_core(job, tile)  # both workers compute the same pixels
    clock.advance(10)
    sched.heartbeat("w2")
    sched.check_failures()  # w1 suspected; tile re-queued
    _, same_tile = sched.request_task("w2")
    assert same_tile.index == tile.index

    # w1 was only slow: its result arrives first and is used...
    assert sched.submit_result("w1", job.job_id, tile.index, core)
    # ...and w2's copy of the same task is ignored.
    assert not sched.submit_result("w2", job.job_id, tile.index, core)
    assert job.state == JobState.DONE
    assert job.tiles_per_worker == {"w1": 1}


def test_late_duplicate_after_reassignment_is_ignored(sched, clock):
    job = sched.submit_job(IMAGE, "invert", tile_size=200)
    _, tile = sched.request_task("w1")
    late_core = filtered_core(job, tile)
    clock.advance(10)
    sched.heartbeat("w2")
    sched.check_failures()
    _, tile2 = sched.request_task("w2")
    assert sched.submit_result("w2", job.job_id, tile2.index, filtered_core(job, tile2))
    assert not sched.submit_result("w1", job.job_id, tile.index, late_core)
    assert np.array_equal(job.result, get_filter("invert")(IMAGE))


# ------------------------------------------------------------------------ lease

def test_lease_expiry_requeues_tile_of_live_worker(sched, clock):
    job = sched.submit_job(IMAGE, "blur", tile_size=200)
    sched.request_task("w1")
    for _ in range(16):  # 32 s, heartbeating the whole time
        clock.advance(2)
        sched.heartbeat("w1")
        sched.check_failures()
    assert sched.workers["w1"].alive
    assert job.reassigned_tiles == 1
    assert sched.request_task("w2") is not None


# ------------------------------------------------------------- errors and retries

def test_reported_error_requeues_tile_for_another_worker(sched):
    job = sched.submit_job(IMAGE, "sharpen", tile_size=200)
    _, tile = sched.request_task("w1")
    sched.report_failure("w1", job.job_id, tile.index, "out of memory")
    assert job.reassigned_tiles == 1
    finish(sched, "w2")
    assert job.state == JobState.DONE
    assert job.tiles_per_worker == {"w2": 1}


def test_error_report_from_non_holder_is_ignored(sched):
    job = sched.submit_job(IMAGE, "sharpen", tile_size=200)
    _, tile = sched.request_task("w1")
    sched.report_failure("w2", job.job_id, tile.index, "not mine")
    assert job.reassigned_tiles == 0


def test_failed_tile_is_not_given_back_to_the_worker_that_failed_it(sched):
    job = sched.submit_job(IMAGE, "sharpen", tile_size=60)  # 2 x 2 = 4 tiles
    _, tile = sched.request_task("w1")
    sched.report_failure("w1", job.job_id, tile.index, "broken machine")
    # Tile 0 is back at the front, but w1 skips it while w2 (healthy) is available...
    assert sched.request_task("w1")[1].index == 1
    # ...and w2 gets it next.
    assert sched.request_task("w2")[1].index == 0


def test_tile_that_fails_everywhere_fails_the_job(sched):
    job = sched.submit_job(IMAGE, "sharpen", tile_size=60)  # 2 x 2 = 4 tiles
    for worker in ("w1", "w2", "w1"):  # once every live worker has failed it, retry anyone
        _, tile = sched.request_task(worker)
        assert tile.index == 0  # the failed tile is always next in line
        sched.report_failure(worker, job.job_id, 0, "boom")
    assert job.state == JobState.FAILED
    assert "Tile 0 failed 3 times" in job.error
    assert sched.request_task("w2") is None  # leftover tiles of the failed job are skipped


def test_other_jobs_continue_after_one_fails(clock):
    s = Scheduler(max_attempts=3, clock=clock)
    s.register_worker("w1", "h")
    bad = s.submit_job(IMAGE, "invert", tile_size=200)
    good = s.submit_job(IMAGE, "invert", tile_size=200)
    for _ in range(3):  # only one worker, so it gets the failing tile every time
        job, tile = s.request_task("w1")
        assert job is bad
        s.report_failure("w1", bad.job_id, tile.index, "boom")
    finish(s, "w1")
    assert bad.state == JobState.FAILED
    assert good.state == JobState.DONE
