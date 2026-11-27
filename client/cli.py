"""Command-line client.

Usage:
    python -m client.cli submit data/input/sample.png --filter blur --tile-size 256
    python -m client.cli submit photo.jpg --filter sobel --coordinator 192.168.1.10 --verify
    python -m client.cli workers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import grpc
import numpy as np

from common.filters import FILTERS
from common.image_io import load_image, save_image
from common.rpc import GRPC_OPTIONS, from_image_data, normalize_address, pb, pb_grpc, to_image_data


class JobFailed(RuntimeError):
    pass


def connect(coordinator: str) -> pb_grpc.CoordinatorStub:
    channel = grpc.insecure_channel(normalize_address(coordinator), options=GRPC_OPTIONS)
    return pb_grpc.CoordinatorStub(channel)


def run_job(
    stub: pb_grpc.CoordinatorStub,
    image: np.ndarray,
    filter_name: str,
    tile_size: int,
    poll_interval: float = 0.05,
    on_progress: Optional[Callable[["pb.StatusReply"], None]] = None,
    timeout: float = 3600,
) -> Tuple[np.ndarray, "pb.StatusReply"]:
    """Submit a job, wait for it to finish, return (result image, final status)."""
    reply = stub.SubmitJob(
        pb.JobRequest(image=to_image_data(image), filter_name=filter_name, tile_size=tile_size)
    )
    if reply.error:
        raise JobFailed(reply.error)

    deadline = time.perf_counter() + timeout
    while True:
        status = stub.GetStatus(pb.StatusRequest(job_id=reply.job_id))
        if on_progress:
            on_progress(status)
        if status.state == "DONE":
            break
        if status.state in ("FAILED", "UNKNOWN"):
            raise JobFailed(status.error or status.state)
        if time.perf_counter() > deadline:
            raise JobFailed(f"Timed out waiting for job {reply.job_id}")
        time.sleep(poll_interval)

    result = stub.GetResult(pb.ResultRequest(job_id=reply.job_id))
    if not result.ready:
        raise JobFailed(result.error or "Result not ready")
    return from_image_data(result.image), status


def before_after(before: np.ndarray, after: np.ndarray, max_width: int = 2400) -> np.ndarray:
    """Side-by-side comparison image, scaled down if very wide."""
    if after.ndim == 2:
        after = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
    if before.ndim == 2:
        before = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    pair = np.hstack([before, after])
    if pair.shape[1] > max_width:
        scale = max_width / pair.shape[1]
        pair = cv2.resize(pair, (max_width, int(pair.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
    return pair


def _print_progress(status: "pb.StatusReply") -> None:
    total = max(status.total_tiles, 1)
    width = 30
    filled = int(width * status.done_tiles / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {status.done_tiles}/{status.total_tiles} tiles  "
          f"{status.total_seconds:6.2f} s", end="", flush=True)


def cmd_submit(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    h, w = image.shape[:2]
    stem = Path(args.image).stem
    output = Path(args.output or f"data/output/{stem}_{args.filter}.png")
    print(f"Submitting {w}x{h} image, filter {args.filter}, tile {args.tile_size} "
          f"to {normalize_address(args.coordinator)}")

    stub = connect(args.coordinator)
    try:
        result, status = run_job(stub, image, args.filter, args.tile_size,
                                 on_progress=_print_progress)
    except grpc.RpcError as exc:
        print(f"\nCould not reach coordinator: {exc.code().name}", file=sys.stderr)
        return 1
    except JobFailed as exc:
        print(f"\nJob failed: {exc}", file=sys.stderr)
        return 1
    print()

    print(f"Job {status.job_id} done: total {status.total_seconds:.3f} s "
          f"(split {status.split_seconds * 1000:.1f} ms, "
          f"processing {status.processing_seconds:.3f} s, "
          f"merge {status.merge_seconds * 1000:.1f} ms)")
    for worker_id, count in sorted(status.tiles_per_worker.items()):
        print(f"  {worker_id}: {count} tiles")

    save_image(output, result)
    print(f"Saved result to {output}")
    if not args.no_before_after:
        pair_path = output.with_name(output.stem + "_before_after" + output.suffix)
        save_image(pair_path, before_after(image, result))
        print(f"Saved before/after to {pair_path}")

    if args.verify:
        from baseline.sequential import run_sequential

        expected, seconds = run_sequential(image, args.filter)
        same = np.array_equal(result, expected)
        print(f"Verify against sequential ({seconds:.3f} s on this machine): "
              f"{'MATCH' if same else 'MISMATCH'}")
        if not same:
            return 2
    return 0


def cmd_workers(args: argparse.Namespace) -> int:
    stub = connect(args.coordinator)
    try:
        reply = stub.ListWorkers(pb.ListWorkersRequest(), timeout=5)
    except grpc.RpcError as exc:
        print(f"Could not reach coordinator: {exc.code().name}", file=sys.stderr)
        return 1
    if not reply.workers:
        print("No workers registered.")
        return 0
    print(f"{'worker':<28} {'host':<20} {'alive':<6} {'last heartbeat':>15} {'tiles':>7}")
    for w in reply.workers:
        print(f"{w.worker_id:<28} {w.hostname:<20} {'yes' if w.alive else 'NO':<6} "
              f"{w.seconds_since_heartbeat:>13.1f} s {w.tiles_done:>7}")
    return 0


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--coordinator", default="localhost",
                        help="coordinator address, host[:port] (default localhost:50051)")
    parser = argparse.ArgumentParser(description="Distributed image processing client.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="process an image", parents=[common])
    p.add_argument("image")
    p.add_argument("--filter", default="blur", choices=sorted(FILTERS))
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--output", help="output path (default data/output/<name>_<filter>.png)")
    p.add_argument("--verify", action="store_true",
                   help="also run the filter locally and check the result matches")
    p.add_argument("--no-before-after", action="store_true",
                   help="don't save the side-by-side comparison image")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("workers", help="list workers known to the coordinator",
                       parents=[common])
    p.set_defaults(func=cmd_workers)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
