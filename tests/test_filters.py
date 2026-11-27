import numpy as np
import pytest

from common.filters import FILTERS, apply_filter, get_filter
from common.image_io import make_sample_image

IMAGE = make_sample_image(60, 80, seed=1)

GRAY_OUTPUT = {"grayscale", "sobel"}


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_output_shape_and_dtype(name):
    out = apply_filter(name, IMAGE)
    assert out.dtype == np.uint8
    if name in GRAY_OUTPUT:
        assert out.shape == IMAGE.shape[:2]
    else:
        assert out.shape == IMAGE.shape


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_filter_changes_image(name):
    out = apply_filter(name, IMAGE)
    if out.shape == IMAGE.shape:
        assert not np.array_equal(out, IMAGE)


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_filters_accept_grayscale_input(name):
    gray = apply_filter("grayscale", IMAGE)
    out = apply_filter(name, gray)
    assert out.shape == gray.shape


def test_radius_is_non_negative_int():
    for flt in FILTERS.values():
        assert isinstance(flt.radius, int) and flt.radius >= 0


def test_pixelwise_filters_have_zero_radius():
    assert get_filter("grayscale").radius == 0
    assert get_filter("invert").radius == 0


def test_invert_twice_is_identity():
    assert np.array_equal(apply_filter("invert", apply_filter("invert", IMAGE)), IMAGE)


def test_unknown_filter_raises():
    with pytest.raises(ValueError, match="Unknown filter"):
        get_filter("does_not_exist")
