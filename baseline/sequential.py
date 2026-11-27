"""Sequential (single-machine, single-process) baseline.

This is the reference the distributed system is compared against:
- correctness: the distributed output must equal ``run_sequential``'s output pixel for pixel;
- performance: speedup = T_sequential / T_distributed.

``run_tiled_local`` runs the same split -> filter -> merge pipeline the distributed system uses,
but in one process. It isolates the cost of tiling itself from the cost of the network.

Usage:
    python -m baseline.sequential data/input/sample.png data/output/blur.png --filter blur
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Tuple

import numpy as np

from common.filters import FILTERS, get_filter, set_opencv_threads
from common.image_io import load_image, save_image
from common.tiling import crop_halo, merge, split


def run_sequential(image: np.ndarray, filter_name: str) -> Tuple[np.ndarray, float]:
    """Filter the whole image at once. Returns (result, seconds)."""
    flt = get_filter(filter_name)
    start = time.perf_counter()
    result = flt(image)
    return result, time.perf_counter() - start


def run_tiled_local(
    image: np.ndarray, filter_name: str, tile_size: int
) -> Tuple[np.ndarray, float]:
    """Split into tiles with a halo, filter each tile, merge. Returns (result, seconds)."""
    flt = get_filter(filter_name)
    start = time.perf_counter()
    tiles = split(image, tile_size, flt.radius)
    cores = [(t, crop_halo(flt(t.data), t)) for t in tiles]
    result = merge(image.shape, cores)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a filter sequentially and time it.")
    parser.add_argument("input", help="input image path")
    parser.add_argument("output", help="output image path")
    parser.add_argument("--filter", default="blur", choices=sorted(FILTERS))
    parser.add_argument("--repeat", type=int, default=3, help="timed runs (default 3)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="if set, also time the tiled single-process pipeline and check it matches",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="OpenCV internal threads (default 1 = truly sequential)",
    )
    args = parser.parse_args()

    set_opencv_threads(args.threads)
    image = load_image(args.input)
    h, w = image.shape[:2]
    print(f"Image: {w}x{h}, filter: {args.filter}, OpenCV threads: {args.threads}")

    run_sequential(image, args.filter)  # warm-up run, not timed (OpenCV initialisation)
    times = []
    for _ in range(args.repeat):
        result, seconds = run_sequential(image, args.filter)
        times.append(seconds)
    save_image(args.output, result)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"Sequential: {mean * 1000:.1f} ms +/- {std * 1000:.1f} ms "
          f"({h * w / mean / 1e6:.1f} Mpixels/s) over {len(times)} runs")
    print(f"Saved {args.output}")

    if args.tile_size:
        tiled_times = []
        for _ in range(args.repeat):
            tiled, seconds = run_tiled_local(image, args.filter, args.tile_size)
            tiled_times.append(seconds)
        same = np.array_equal(tiled, result)
        print(f"Tiled local (tile {args.tile_size}): "
              f"{statistics.mean(tiled_times) * 1000:.1f} ms, matches sequential: {same}")


if __name__ == "__main__":
    main()
