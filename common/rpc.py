"""gRPC helpers shared by the coordinator, workers and client.

The Python code for the .proto (``image_service_pb2*.py``) is generated, not committed. It is
regenerated automatically on import whenever it is missing or older than the .proto file, so
nobody has to remember to run protoc.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from common.tiling import Tile

_ROOT = Path(__file__).resolve().parent.parent
_PROTO = _ROOT / "common" / "proto" / "image_service.proto"
_GENERATED = [
    _ROOT / "common" / "proto" / "image_service_pb2.py",
    _ROOT / "common" / "proto" / "image_service_pb2_grpc.py",
]

# Images can be large (a 4000x3000 colour image is 36 MB of raw pixels), so raise gRPC's 4 MB
# default message limit.
MAX_MESSAGE_BYTES = 512 * 1024 * 1024
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]

DEFAULT_PORT = 50051


def _ensure_generated() -> None:
    proto_mtime = _PROTO.stat().st_mtime
    if all(p.exists() and p.stat().st_mtime >= proto_mtime for p in _GENERATED):
        return
    from grpc_tools import protoc

    # Compiling with the repo root as include path makes the generated code import
    # `from common.proto import image_service_pb2`, which works from anywhere in the project.
    code = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{_ROOT}",
            f"--python_out={_ROOT}",
            f"--grpc_python_out={_ROOT}",
            str(_PROTO),
        ]
    )
    if code != 0:
        raise RuntimeError(f"protoc failed with exit code {code}")
    importlib.invalidate_caches()


_ensure_generated()

from common.proto import image_service_pb2 as pb  # noqa: E402
from common.proto import image_service_pb2_grpc as pb_grpc  # noqa: E402


def to_image_data(array: np.ndarray) -> "pb.ImageData":
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels, got {array.dtype}")
    array = np.ascontiguousarray(array)
    channels = 1 if array.ndim == 2 else array.shape[2]
    return pb.ImageData(
        pixels=array.tobytes(),
        height=array.shape[0],
        width=array.shape[1],
        channels=channels,
    )


def from_image_data(msg: "pb.ImageData") -> np.ndarray:
    array = np.frombuffer(msg.pixels, dtype=np.uint8)
    if msg.channels == 1:
        return array.reshape(msg.height, msg.width)
    return array.reshape(msg.height, msg.width, msg.channels)


def tile_to_info(tile: Tile) -> "pb.TileInfo":
    return pb.TileInfo(
        index=tile.index,
        y=tile.y,
        x=tile.x,
        height=tile.height,
        width=tile.width,
        pad_top=tile.pad_top,
        pad_left=tile.pad_left,
    )


def tile_from_info(info: "pb.TileInfo", data: np.ndarray) -> Tile:
    return Tile(
        index=info.index,
        y=info.y,
        x=info.x,
        height=info.height,
        width=info.width,
        pad_top=info.pad_top,
        pad_left=info.pad_left,
        data=data,
    )


def normalize_address(address: str) -> str:
    """Accept 'host', 'host:port' or ':port'."""
    if ":" not in address:
        return f"{address}:{DEFAULT_PORT}"
    if address.startswith(":"):
        return f"localhost{address}"
    return address
