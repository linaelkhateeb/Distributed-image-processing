"""Worker process: pull a tile, filter it, send it back, repeat.

Usage:
    python -m worker.worker --coordinator 192.168.1.10:50051
    python -m worker.worker                      # coordinator on this machine

A background thread sends a heartbeat every couple of seconds so the coordinator knows this
worker is alive (the coordinator uses that to detect failed workers).
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from collections import deque
from typing import Deque, Optional

import grpc

from common.filters import get_filter, set_opencv_threads
from common.rpc import (
    GRPC_OPTIONS,
    from_image_data,
    normalize_address,
    pb,
    pb_grpc,
    tile_from_info,
    to_image_data,
)
from common.tiling import crop_halo

log = logging.getLogger("worker")


class Worker:
    def __init__(
        self,
        coordinator: str = "localhost",
        worker_id: Optional[str] = None,
        poll_interval: float = 0.05,
        prefetch: int = 1,
    ):
        self.address = normalize_address(coordinator)
        self.hostname = socket.gethostname()
        self.worker_id = worker_id or f"{self.hostname}-{os.getpid()}"
        self.poll_interval = poll_interval
        # How many task requests to keep in flight while filtering the current tile.
        # 0 = strictly one-at-a-time (request, filter, submit, repeat).
        self.prefetch = prefetch
        self.heartbeat_interval = 2.0
        self.tiles_done = 0
        self._channel = grpc.insecure_channel(self.address, options=GRPC_OPTIONS)
        self._stub = pb_grpc.CoordinatorStub(self._channel)

    # ----------------------------------------------------------------- setup

    def _register(self, stop: threading.Event) -> bool:
        """Register with the coordinator, retrying until it is reachable."""
        warned = False
        while not stop.is_set():
            try:
                reply = self._stub.RegisterWorker(
                    pb.RegisterRequest(worker_id=self.worker_id, hostname=self.hostname),
                    timeout=5,
                )
                self.heartbeat_interval = reply.heartbeat_interval or self.heartbeat_interval
                log.info("Registered with coordinator at %s as %s",
                         self.address, self.worker_id)
                return True
            except grpc.RpcError as exc:
                if not warned:
                    log.warning("Coordinator at %s not reachable (%s); retrying...",
                                self.address, exc.code().name)
                    warned = True
                stop.wait(1.0)
        return False

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_interval):
            try:
                reply = self._stub.Heartbeat(
                    pb.HeartbeatRequest(worker_id=self.worker_id), timeout=5
                )
                if not reply.known:
                    # Coordinator restarted and forgot us.
                    log.warning("Coordinator does not know this worker; re-registering")
                    self._register(stop)
            except grpc.RpcError as exc:
                log.warning("Heartbeat failed: %s", exc.code().name)

    # ------------------------------------------------------------------ work

    def process(self, reply: "pb.TaskReply") -> "pb.TaskResult":
        """Filter one tile and crop its halo. Pure computation, no networking."""
        data = from_image_data(reply.data)
        tile = tile_from_info(reply.tile, data)
        start = time.perf_counter()
        filtered = get_filter(reply.filter_name)(data)
        core = crop_halo(filtered, tile)
        seconds = time.perf_counter() - start
        return pb.TaskResult(
            worker_id=self.worker_id,
            task_id=reply.task_id,
            job_id=reply.job_id,
            tile_index=reply.tile.index,
            core=to_image_data(core),
            process_seconds=seconds,
        )

    def run(self, stop: Optional[threading.Event] = None) -> None:
        """Main loop. Returns when ``stop`` is set (or on Ctrl+C when run from main)."""
        stop = stop or threading.Event()
        if not self._register(stop):
            return
        heartbeat = threading.Thread(target=self._heartbeat_loop, args=(stop,),
                                     name=f"heartbeat-{self.worker_id}", daemon=True)
        heartbeat.start()

        # Every gRPC call costs a few milliseconds of latency, which can be as long as filtering
        # a tile. To hide it, the worker pipelines: requests for the next tiles are already in
        # flight while it filters the current one, and results are submitted in the background.
        request = pb.TaskRequest(worker_id=self.worker_id)
        requests: Deque["grpc.Future"] = deque()  # outstanding RequestTask calls
        submits: Deque["grpc.Future"] = deque()  # outstanding SubmitResult calls

        def keep_requests_in_flight(n: int) -> None:
            while len(requests) < n:
                requests.append(self._stub.RequestTask.future(request, timeout=10))

        while not stop.is_set():
            try:
                t0 = time.perf_counter()
                keep_requests_in_flight(1)
                reply = requests.popleft().result()
                if not reply.has_task:
                    stop.wait(self.poll_interval)
                    continue
                t1 = time.perf_counter()
                keep_requests_in_flight(self.prefetch)  # ask for the next tile(s) now
                result = self.process(reply)
                t2 = time.perf_counter()
                if self.prefetch == 0:
                    self._stub.SubmitResult(result, timeout=30)
                else:
                    submits.append(self._stub.SubmitResult.future(result, timeout=30))
                    # Collect finished submits (raises if one failed); cap how many are pending.
                    while submits and (submits[0].done() or len(submits) > 2):
                        submits.popleft().result()
                t3 = time.perf_counter()
                self.tiles_done += 1
                log.debug("%s: wait %.1f ms, filter %.1f ms, submit %.1f ms",
                          reply.task_id, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)
            except grpc.RpcError as exc:
                log.warning("Coordinator call failed (%s); retrying in 1 s", exc.code().name)
                requests.clear()
                submits.clear()
                stop.wait(1.0)

        for fut in submits:  # let the last results reach the coordinator
            try:
                fut.result()
            except grpc.RpcError:
                pass
        self._channel.close()
        log.info("Worker %s stopped after %d tiles", self.worker_id, self.tiles_done)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a worker.")
    parser.add_argument("--coordinator", default="localhost",
                        help="coordinator address, host[:port] (default localhost:50051)")
    parser.add_argument("--id", default=None, help="worker id (default hostname-pid)")
    parser.add_argument("--threads", type=int, default=1,
                        help="OpenCV threads per worker (default 1)")
    parser.add_argument("--prefetch", type=int, default=1,
                        help="task requests kept in flight while filtering (default 1; "
                             "0 = no pipelining)")
    parser.add_argument("--verbose", action="store_true",
                        help="log timing for every tile (wait / filter / submit)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s.%(msecs)03d %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    set_opencv_threads(args.threads)
    worker = Worker(args.coordinator, args.id, prefetch=args.prefetch)
    stop = threading.Event()
    try:
        worker.run(stop)
    except KeyboardInterrupt:
        stop.set()
        log.info("Worker %s stopped after %d tiles", worker.worker_id, worker.tiles_done)


if __name__ == "__main__":
    main()
