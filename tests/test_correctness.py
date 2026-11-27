"""The key correctness property of the project.

Splitting an image into tiles with a halo, filtering each tile independently, and merging must
give exactly the same pixels as filtering the whole image at once. If this holds, the distributed
system (which does the same thing, just on different machines) is correct by construction.
"""

import numpy as np
import pytest

from baseline.sequential import run_sequential, run_tiled_local
from common.filters import FILTERS, get_filter
from common.image_io import make_sample_image
from common.tiling import crop_halo, merge, split

IMAGES = {
    "small_square": make_sample_image(64, 64, seed=3),
    "odd_size": make_sample_image(257, 511, seed=4),
    "tall": make_sample_image(300, 45, seed=5),
}

TILE_SIZES = [16, 37, 64, 128, 1000]


@pytest.mark.parametrize("image_name", sorted(IMAGES))
@pytest.mark.parametrize("tile_size", TILE_SIZES)
@pytest.mark.parametrize("filter_name", sorted(FILTERS))
def test_tiled_equals_sequential(filter_name, tile_size, image_name):
    image = IMAGES[image_name]
    expected, _ = run_sequential(image, filter_name)
    actual, _ = run_tiled_local(image, filter_name, tile_size)
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), (
        f"{filter_name}, tile {tile_size}, {image_name}: "
        f"{np.count_nonzero(actual != expected)} pixels differ"
    )


@pytest.mark.parametrize("filter_name", sorted(FILTERS))
def test_tiled_equals_sequential_on_grayscale_input(filter_name):
    gray = get_filter("grayscale")(IMAGES["odd_size"])
    expected, _ = run_sequential(gray, filter_name)
    actual, _ = run_tiled_local(gray, filter_name, 50)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "filter_name", [n for n, f in sorted(FILTERS.items()) if f.radius > 0]
)
def test_without_halo_neighbourhood_filters_show_seams(filter_name):
    """Shows why the halo is needed: with halo=0 the tile borders come out wrong."""
    image = IMAGES["odd_size"]
    flt = get_filter(filter_name)
    expected = flt(image)
    tiles = split(image, 64, halo=0)
    actual = merge(image.shape, [(t, crop_halo(flt(t.data), t)) for t in tiles])
    assert not np.array_equal(actual, expected)
