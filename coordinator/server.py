"""Coordinator gRPC server.

Usage:
    python -m coordinator.server --port 50051

Workers and clients connect to this machine's IP address on that port.
"""

from __future__ import annotations

import argparse
import logging
import socket
from concurrent import futures
from typing import Tuple

import grpc

from common.rpc import (
    DEFAULT_PORT,
    GRPC_OPTIONS,
    from_image_data,
    pb,
    pb_grpc,
    tile_to_info,
    to_image_data,
)
from coordinator.job import JobState
from coordinator.scheduler import Scheduler

log = logging.getLogger("coordinator")


class CoordinatorServicer(pb_grpc.CoordinatorServicer):
    """Translates gRPC calls into Scheduler calls. No logic of its own."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    # ---------------------------------------------------------------- workers

    def RegisterWorker(self, request, context):
        self.scheduler.register_worker(request.worker_id, request.hostname)
        return pb.RegisterReply(
            accepted=True, heartbeat_interval=self.scheduler.heartbeat_interval
        )

    def Heartbeat(self, request, context):
        return pb.HeartbeatReply(known=self.scheduler.heartbeat(request.worker_id))

    def RequestTask(self, request, context):
        assigned = self.scheduler.request_task(request.worker_id)
        if assigned is None:
            return pb.TaskReply(has_task=False)
        job, tile = assigned
        return pb.TaskReply(
            has_task=True,
            task_id=job.task_id(tile.index),
            job_id=job.job_id,
            filter_name=job.filter_name,
            tile=tile_to_info(tile),
            data=to_image_data(tile.data),
        )

    def SubmitResult(self, request, context):
        accepted = self.scheduler.submit_result(
            request.worker_id,
            request.job_id,
            request.tile_index,
            from_image_data(request.core),
        )
        return pb.SubmitReply(accepted=accepted)

    # ----------------------------------------------------------------- client

    def SubmitJob(self, request, context):
        try:
            job = self.scheduler.submit_job(
                from_image_data(request.image), request.filter_name, request.tile_size
            )
        except ValueError as exc:
            return pb.JobReply(error=str(exc))
        return pb.JobReply(job_id=job.job_id, total_tiles=job.total_tiles)

    def GetStatus(self, request, context):
        job = self.scheduler.get_job(request.job_id)
        if job is None:
            return pb.StatusReply(job_id=request.job_id, state="UNKNOWN",
                                  error="No such job")
        return pb.StatusReply(
            job_id=job.job_id,
            state=job.state,
            total_tiles=job.total_tiles,
            done_tiles=job.done_tiles if job.state == JobState.RUNNING else job.total_tiles,
            tiles_per_worker=dict(job.tiles_per_worker),
            split_seconds=job.split_seconds,
            processing_seconds=job.processing_seconds,
            merge_seconds=job.merge_seconds,
            total_seconds=job.total_seconds,
            error=job.error,
        )

    def GetResult(self, request, context):
        job = self.scheduler.get_job(request.job_id)
        if job is None:
            return pb.ResultReply(ready=False, error="No such job")
        if job.state != JobState.DONE:
            return pb.ResultReply(ready=False, error=job.error)
        return pb.ResultReply(ready=True, image=to_image_data(job.result))

    def ListWorkers(self, request, context):
        return pb.ListWorkersReply(
            workers=[
                pb.WorkerStatus(
                    worker_id=w.worker_id,
                    hostname=w.hostname,
                    seconds_since_heartbeat=since,
                    tiles_done=w.tiles_done,
                    alive=alive,
                )
                for w, since, alive in self.scheduler.list_workers()
            ]
        )


def serve(
    port: int = DEFAULT_PORT, scheduler: Scheduler | None = None, max_threads: int = 32
) -> Tuple[grpc.Server, Scheduler, int]:
    """Start the server (non-blocking). ``port=0`` picks a free port. Returns the bound port."""
    scheduler = scheduler or Scheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_threads),
                         options=GRPC_OPTIONS)
    pb_grpc.add_CoordinatorServicer_to_server(CoordinatorServicer(scheduler), server)
    bound = server.add_insecure_port(f"0.0.0.0:{port}")
    if bound == 0:
        raise RuntimeError(f"Could not bind port {port}")
    server.start()
    return server, scheduler, bound


def _local_ips() -> list[str]:
    try:
        return sorted({ip for *_, (ip, *_) in socket.getaddrinfo(socket.gethostname(), None,
                                                                  socket.AF_INET)})
    except OSError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the coordinator.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--heartbeat-interval", type=float, default=2.0)
    parser.add_argument("--heartbeat-timeout", type=float, default=6.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    scheduler = Scheduler(args.heartbeat_interval, args.heartbeat_timeout)
    server, _, port = serve(args.port, scheduler)
    ips = ", ".join(f"{ip}:{port}" for ip in _local_ips()) or f"<this machine>:{port}"
    log.info("Coordinator listening on port %d. Workers can connect to: %s", port, ips)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.stop(grace=1)


if __name__ == "__main__":
    main()
