"""Split an image into tiles with a halo, and merge processed tiles back.

Why a halo? A neighbourhood filter (blur, sharpen, edges, median) computes each output pixel from
the pixels within ``radius`` of it. If we filtered bare tiles, pixels near a tile edge would be
missing their neighbours from the next tile and the result would show seams. So each tile is cut
with an extra border of ``halo`` pixels taken from the neighbouring tiles; after filtering, that
border is cropped off and only the tile's own "core" pixels are kept.

At the image's outer edges there is nothing to borrow, so the halo is clipped there. The filter
then handles that edge exactly as it does when filtering the whole image, which keeps the
distributed result identical to the sequential one.

    image
    ┌───────────────┬───────────────┐
    │   core 0      │   core 1      │      padded tile 1 = core 1 plus up to `halo`
    │            ┌──┼───────────────┤      pixels on each side, clipped at the image edge
    │            │##│   core 1      │
    ├────────────┴──┼───────────────┤
    │   core 2      │   core 3      │
    └───────────────┴───────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class Tile:
    index: int  # position in row-major order; unique within an image
    y: int  # top-left of the core in image coordinates
    x: int
    height: int  # size of the core (the pixels this tile is responsible for)
    width: int
    pad_top: int  # where the core starts inside `data` (amount of halo actually present)
    pad_left: int
    data: np.ndarray  # core plus halo, cut from the source image


def split(image: np.ndarray, tile_size: int, halo: int) -> List[Tile]:
    """Cut ``image`` into ``tile_size`` x ``tile_size`` cores, each with up to ``halo`` extra pixels.

    Tiles on the right and bottom edges are smaller when the image size is not a multiple of
    ``tile_size``.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")

    img_h, img_w = image.shape[:2]
    tiles: List[Tile] = []
    index = 0
    for y in range(0, img_h, tile_size):
        for x in range(0, img_w, tile_size):
            h = min(tile_size, img_h - y)
            w = min(tile_size, img_w - x)
            y0 = max(0, y - halo)
            x0 = max(0, x - halo)
            y1 = min(img_h, y + h + halo)
            x1 = min(img_w, x + w + halo)
            tiles.append(
                Tile(
                    index=index,
                    y=y,
                    x=x,
                    height=h,
                    width=w,
                    pad_top=y - y0,
                    pad_left=x - x0,
                    data=np.ascontiguousarray(image[y0:y1, x0:x1]),
                )
            )
            index += 1
    return tiles


def crop_halo(processed: np.ndarray, tile: Tile) -> np.ndarray:
    """Keep only the core pixels of a filtered padded tile."""
    return processed[
        tile.pad_top : tile.pad_top + tile.height,
        tile.pad_left : tile.pad_left + tile.width,
    ]


def merge(
    image_shape: Tuple[int, ...], cores: Iterable[Tuple[Tile, np.ndarray]]
) -> np.ndarray:
    """Place filtered cores back into a full image.

    ``image_shape`` is the source image's shape; only its height and width are used, because a
    filter may change the number of channels (e.g. grayscale turns BGR into a single channel).
    Every tile from ``split`` must be supplied exactly once.
    """
    img_h, img_w = image_shape[:2]
    output = None
    covered = 0
    seen = set()
    for tile, core in cores:
        if tile.index in seen:
            raise ValueError(f"Tile {tile.index} supplied more than once")
        seen.add(tile.index)
        if core.shape[:2] != (tile.height, tile.width):
            raise ValueError(
                f"Tile {tile.index}: core shape {core.shape[:2]} != "
                f"expected {(tile.height, tile.width)}"
            )
        if output is None:
            output = np.empty((img_h, img_w) + core.shape[2:], dtype=core.dtype)
        output[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = core
        covered += tile.height * tile.width

    if output is None:
        raise ValueError("No tiles to merge")
    if covered != img_h * img_w:
        raise ValueError(f"Tiles cover {covered} pixels, image has {img_h * img_w}")
    return output
