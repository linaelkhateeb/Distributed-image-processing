import numpy as np
import pytest
from PIL import Image, features

from common.image_io import load_image, make_sample_image, save_image

IMAGE = make_sample_image(40, 60, seed=6)


def test_png_round_trip_is_lossless(tmp_path):
    path = tmp_path / "img.png"
    save_image(path, IMAGE)
    assert np.array_equal(load_image(path), IMAGE)


def test_non_ascii_path(tmp_path):
    path = tmp_path / "صورة.png"
    save_image(path, IMAGE)
    assert np.array_equal(load_image(path), IMAGE)


@pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF")
def test_avif_falls_back_to_pillow(tmp_path):
    path = tmp_path / "img.avif"
    # Pillow takes RGB; our arrays are BGR.
    Image.fromarray(IMAGE[..., ::-1]).save(path, quality=100)
    loaded = load_image(path)
    assert loaded.shape == IMAGE.shape and loaded.dtype == np.uint8
    # AVIF is lossy, so compare roughly: channels must not be swapped.
    assert np.abs(loaded.astype(int) - IMAGE.astype(int)).mean() < 20


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_garbage_file_raises(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Could not decode"):
        load_image(path)
