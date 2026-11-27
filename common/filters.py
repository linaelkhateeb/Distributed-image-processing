"""Image filters used by the baseline and by workers.

Every filter declares a ``radius``: how many pixels of neighbourhood it reads around each output
pixel. That radius is the halo a tile needs so that filtering the tile gives exactly the same
pixels as filtering the whole image (see ``common/tiling.py``).

Filters take and return ``uint8`` NumPy arrays in OpenCV's layout: ``(H, W)`` for grayscale or
``(H, W, 3)`` for BGR colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import cv2
import numpy as np

FilterFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Filter:
    name: str
    radius: int  # halo size in pixels needed around each tile
    fn: FilterFn
    description: str

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.fn(image)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _grayscale(image: np.ndarray) -> np.ndarray:
    return _to_gray(image).copy()


def _invert(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def _gaussian(ksize: int) -> FilterFn:
    def fn(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    return fn


_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def _sharpen(image: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, _SHARPEN_KERNEL)


def _sobel(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    return cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)


def _median(ksize: int) -> FilterFn:
    def fn(image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, ksize)

    return fn


FILTERS: Dict[str, Filter] = {
    f.name: f
    for f in [
        # Pixel-wise (radius 0): cheap, no halo needed.
        Filter("grayscale", 0, _grayscale, "Convert to grayscale"),
        Filter("invert", 0, _invert, "Invert colours"),
        # Neighbourhood filters: need a halo of `radius` pixels.
        Filter("blur", 2, _gaussian(5), "Gaussian blur, 5x5"),
        Filter("blur_heavy", 15, _gaussian(31), "Gaussian blur, 31x31 (large halo)"),
        Filter("sharpen", 1, _sharpen, "Sharpen, 3x3 kernel"),
        Filter("sobel", 1, _sobel, "Sobel edge detection (grayscale output)"),
        Filter("median", 2, _median(5), "Median filter, 5x5"),
        Filter("median_heavy", 7, _median(15), "Median filter, 15x15 (expensive)"),
    ]
}


def set_opencv_threads(n: int) -> None:
    """Limit OpenCV's internal multithreading.

    OpenCV parallelises many filters across all CPU cores by default. For honest measurements the
    sequential baseline uses 1 thread, and each worker uses a fixed number of threads, so that any
    speedup comes from our distribution of tiles and not from OpenCV.
    """
    cv2.setNumThreads(n)


def get_filter(name: str) -> Filter:
    try:
        return FILTERS[name]
    except KeyError:
        raise ValueError(f"Unknown filter {name!r}. Available: {', '.join(FILTERS)}") from None


def apply_filter(name: str, image: np.ndarray) -> np.ndarray:
    return get_filter(name)(image)
