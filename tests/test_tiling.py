import numpy as np
import pytest

from common.image_io import make_sample_image
from common.tiling import crop_halo, merge, split

IMAGE = make_sample_image(103, 157, seed=2)


@pytest.mark.parametrize("tile_size", [1, 16, 50, 103, 157, 500])
@pytest.mark.parametrize("halo", [0, 1, 7])
def test_split_then_merge_without_filter_is_identity(tile_size, halo):
    tiles = split(IMAGE, tile_size, halo)
    out = merge(IMAGE.shape, [(t, crop_halo(t.data, t)) for t in tiles])
    assert np.array_equal(out, IMAGE)


def test_cores_cover_image_exactly_once():
    tiles = split(IMAGE, 32, 3)
    count = np.zeros(IMAGE.shape[:2], dtype=int)
    for t in tiles:
        count[t.y : t.y + t.height, t.x : t.x + t.width] += 1
    assert (count == 1).all()


def test_tile_indices_are_unique_and_row_major():
    tiles = split(IMAGE, 40, 2)
    assert [t.index for t in tiles] == list(range(len(tiles)))
    # 103x157 with 40px tiles -> 3 rows x 4 columns
    assert len(tiles) == 3 * 4
    assert (tiles[1].y, tiles[1].x) == (0, 40)
    assert (tiles[4].y, tiles[4].x) == (40, 0)


def test_edge_tiles_are_smaller():
    tiles = split(IMAGE, 40, 0)
    last = tiles[-1]
    assert (last.height, last.width) == (103 - 80, 157 - 120)


def test_halo_is_clipped_at_image_edges():
    halo = 5
    tiles = split(IMAGE, 40, halo)
    first = tiles[0]  # top-left: no room for halo above or left
    assert (first.pad_top, first.pad_left) == (0, 0)
    assert first.data.shape[:2] == (40 + halo, 40 + halo)
    middle = tiles[5]  # row 1, column 1: full halo on every side
    assert (middle.pad_top, middle.pad_left) == (halo, halo)
    assert middle.data.shape[:2] == (40 + 2 * halo, 40 + 2 * halo)


def test_halo_contains_neighbouring_pixels():
    tiles = split(IMAGE, 40, 5)
    t = tiles[5]
    assert np.array_equal(t.data, IMAGE[t.y - 5 : t.y + 45, t.x - 5 : t.x + 45])


def test_merge_rejects_missing_tile():
    tiles = split(IMAGE, 40, 0)
    cores = [(t, t.data) for t in tiles[:-1]]
    with pytest.raises(ValueError, match="cover"):
        merge(IMAGE.shape, cores)


def test_merge_rejects_duplicate_tile():
    tiles = split(IMAGE, 40, 0)
    cores = [(t, t.data) for t in tiles] + [(tiles[0], tiles[0].data)]
    with pytest.raises(ValueError, match="more than once"):
        merge(IMAGE.shape, cores)


@pytest.mark.parametrize("tile_size, halo", [(0, 1), (-5, 1), (10, -1)])
def test_invalid_arguments(tile_size, halo):
    with pytest.raises(ValueError):
        split(IMAGE, tile_size, halo)
