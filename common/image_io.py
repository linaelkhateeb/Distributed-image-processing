"""Loading, saving and generating images.

``cv2.imread``/``cv2.imwrite`` fail silently on Windows paths with non-ASCII characters, so files
are read and written as bytes and decoded/encoded with OpenCV instead.

OpenCV's pip wheels can't decode some formats (e.g. AVIF, HEIC-style photos from phones), so
``load_image`` falls back to Pillow for those.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _decode_with_pillow(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # respect phone-camera rotation
            rgb = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a BGR ``uint8`` array (OpenCV layout)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such image: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        image = _decode_with_pillow(path)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image as {path.suffix!r}")
    buf.tofile(str(path))


def make_sample_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A synthetic BGR test image: colour gradients, shapes, edges and noise.

    It has smooth areas, sharp edges and fine texture, so every filter visibly changes it and
    seams between tiles would show up.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.empty((height, width, 3), dtype=np.float32)
    image[..., 0] = 255 * xx / max(width - 1, 1)
    image[..., 1] = 255 * yy / max(height - 1, 1)
    image[..., 2] = 127 + 127 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    image = image.astype(np.uint8)

    for _ in range(max(4, (height * width) // 400_000)):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(5, max(6, min(height, width) // 12)))
        if rng.random() < 0.5:
            cv2.circle(image, (cx, cy), r, color, thickness=-1)
        else:
            cv2.rectangle(image, (cx - r, cy - r), (cx + r, cy + r), color, thickness=3)

    noise = rng.integers(-20, 21, size=image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
