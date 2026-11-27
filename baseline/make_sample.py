"""Generate a synthetic test image.

Usage:
    python -m baseline.make_sample data/input/sample.png --width 4000 --height 3000
"""

from __future__ import annotations

import argparse

from common.image_io import make_sample_image, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic test image.")
    parser.add_argument("output", help="output image path, e.g. data/input/sample.png")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    save_image(args.output, make_sample_image(args.height, args.width, args.seed))
    print(f"Saved {args.width}x{args.height} image to {args.output}")


if __name__ == "__main__":
    main()
