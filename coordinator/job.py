"""State of one image-processing job on the coordinator."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.tiling import Tile


class JobState:
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class Job:
    job_id: str
    filter_name: str
    tile_size: int
    image_shape: Tuple[int, ...]
    tiles: List[Tile]
    created_at: float = field(default_factory=time.perf_counter)  # monotonic clock
    state: str = JobState.RUNNING
    cores: Dict[int, np.ndarray] = field(default_factory=dict)  # tile index -> filtered core
    tiles_per_worker: Counter = field(default_factory=Counter)
    merging: bool = False
    split_seconds: float = 0.0
    last_result_at: Optional[float] = None
    merge_seconds: float = 0.0
    finished_at: Optional[float] = None
    result: Optional[np.ndarray] = None
    error: str = ""

    def task_id(self, tile_index: int) -> str:
        """Stable id for a tile's task. A re-sent tile keeps the same id, so results are
        idempotent: whichever copy arrives first is stored, later copies are ignored."""
        return f"{self.job_id}:{tile_index}"

    @property
    def total_tiles(self) -> int:
        return len(self.tiles)

    @property
    def done_tiles(self) -> int:
        return len(self.cores)

    @property
    def processing_seconds(self) -> float:
        end = self.last_result_at if self.last_result_at is not None else time.perf_counter()
        return end - self.created_at

    @property
    def total_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.perf_counter()
        return end - self.created_at
