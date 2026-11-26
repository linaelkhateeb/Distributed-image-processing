# Distributed Image Processing — Project Plan

CSC410 Distributed Systems project. Apply image filters to large images by splitting them into
tiles and processing the tiles in parallel on several nodes, focusing on performance,
scalability and fault tolerance.

## Goals

1. **Faster:** speedup as workers are added.
2. **Scalable:** more work handled by adding workers.
3. **Fault tolerant:** killing a worker mid-job still gives a correct result.

## Architecture

```
 CLI client ──► Coordinator ──gRPC──► Worker 1 … Worker N
   (upload,      (split → schedule →     (filter tiles,
    filter,       monitor → merge)        send heartbeats)
    before/after)
```

**Tech:** Python, NumPy + OpenCV (filters), gRPC (communication), matplotlib (graphs), pytest.

## Core methodologies

| # | Methodology | What we do | Course link |
|---|---|---|---|
| 1 | Tiling with a halo | Split the image into tiles, each with a border of `kernel_size/2` pixels so neighborhood filters see their neighbors; crop the border after processing so there are no seams. | L1 sharding, L3 MapReduce |
| 2 | Coordinator and workers over RPC | Coordinator splits, assigns, tracks and merges; workers filter tiles; all communication over gRPC with a `.proto` interface. | L5 RPC, Lab 4, L3 master/worker |
| 3 | Pull-based scheduling | An idle worker asks for the next tile, so faster machines take more work (dynamic load balancing). | L1 scalability |
| 4 | Heartbeats and re-assignment | Workers heartbeat every ~2 s; a worker that misses the timeout is suspected dead and its unfinished tiles are re-queued. | L10 failure detectors, L12 |
| 5 | Task IDs and idempotent tiles | Every tile task has a unique ID; re-sending is safe (at-least-once), duplicate results are ignored. | L5 RPC semantics, L12 idempotence |
| 6 | Sequential baseline and correctness check | Single-machine reference; distributed output must match it pixel for pixel. | — |

All timings use a monotonic clock (`time.perf_counter()`).

## Build steps (in order)

Estimated times assume building with Claude: Claude writes most of the code, we run, test and
understand it.

### Phase 0 — Setup (≈15 min)
- [x] 1. Repo structure:
  ```
  common/        filters.py, tiling.py, proto/
  coordinator/   server.py, scheduler.py, job.py
  worker/        worker.py
  client/        cli.py
  baseline/      sequential.py
  benchmarks/    run_experiments.py, plot.py
  tests/
  ```
- [x] 2. Virtual environment and `requirements.txt` (opencv-python, numpy, grpcio, grpcio-tools, matplotlib, pytest)
- [x] 3. `.gitignore`, minimal README

### Phase 1 — Image processing core, single machine (≈1–1.5 h)
- [ ] 4. Filters (`filters.py`): grayscale, invert, Gaussian blur, sharpen, Sobel edges, median. Each filter declares its kernel radius (the halo it needs).
- [ ] 5. Tiling (`tiling.py`): `split(image, tile_size, halo)` returns tiles with coordinates and halo; `merge(tiles)` crops each halo and places the tile. Handles edge tiles and sizes that aren't multiples of the tile size.
- [ ] 6. Sequential baseline (`sequential.py`): load, filter, save, time.
- [ ] 7. Correctness tests: split → filter → merge must equal the sequential output pixel for pixel, for every filter, several tile sizes and odd image sizes. **Most important test in the project.**

### Phase 2 — Distributed system, happy path (≈2–3 h)
- [ ] 8. gRPC interface (`image_service.proto`):
  - Worker → coordinator: `RegisterWorker`, `RequestTask` (pull), `SubmitResult(task_id, tile)`, `Heartbeat`
  - Client → coordinator: `SubmitJob`, `GetStatus`, `GetResult`
  - Raise the 4 MB default message limit.
- [ ] 9. Coordinator: job registry, task queue with unique `task_id`s, per-task state (pending → assigned(worker, time) → done); merge when all tiles of a job are done.
- [ ] 10. Worker: pull → filter → submit loop, plus a heartbeat thread; takes a worker ID and the coordinator address.
- [ ] 11. CLI client: `submit image.png --filter blur`, poll progress, save the result and a before/after image.
- [ ] 12. End-to-end run on one machine: coordinator + 3–4 workers; output matches the sequential version.

### Phase 3 — Fault tolerance (≈1.5–2 h)
- [ ] 13. Heartbeat monitor: worker marked *suspected dead* after a timeout (e.g. 6 s).
- [ ] 14. Re-assignment: a dead worker's tiles go back to pending; tasks also have a lease timeout (catches stuck-but-alive workers).
- [ ] 15. Idempotence and dedup: results for already-done tasks are ignored by `task_id`.
- [ ] 16. Retry limit: a tile that fails too often is marked failed.
- [ ] 17. Failure injection: worker flag `--crash-after N`; also test a real kill.
- [ ] 18. Fault tests: kill a worker mid-job; job completes and output still matches the sequential version.

### Phase 4 — Measurements and experiments (≈1 h to build + 2–4 h to run)
- [ ] 19. Metrics: per-tile timing, which worker, total job time → CSV.
- [ ] 20. Experiment runner (`run_experiments.py`), each with repeats (mean ± std):
  - Speedup / efficiency with 1, 2, 4, 8 workers vs the sequential baseline
  - Tile size sweep (64 → 1024 px)
  - Cheap vs expensive filters
  - Kill a worker mid-job
- [ ] 21. Plots (`plot.py`): speedup, efficiency, tile size, failure overhead. Explain with Amdahl's law.
- [ ] 22. Multi-machine run: coordinator on one laptop, workers on others, same Wi-Fi; open the port in Windows Firewall; re-run experiments (the real results).

### Phase 5 — Demo and documentation (≈4–6 h, mostly the report)
- [ ] 23. Demo visuals: before/after, and the result with tiles colored by the worker that processed them.
- [ ] 24. README: architecture diagram, setup, running on one or several machines.
- [ ] 25. Report: introduction, architecture, the six methodologies tied to lectures, results and graphs with Amdahl's law, limitations (coordinator is a single point of failure), future work (Numba/GPU, standby coordinator).
- [ ] 26. Demo script: submit a job, show workers sharing work, kill a worker live, show the correct result.

### Phase 6 — Optional, only if there's time
- [ ] 27. Hot-standby coordinator (≈2–3 h)
- [ ] 28. MPI baseline (≈1 h)
- [ ] 29. Web UI (≈1–2 h)

## Time estimate

| Part | Time |
|---|---|
| Code (Phases 0–3) | ≈5–7 h |
| Experiments incl. multi-laptop setup and runs | ≈3–5 h |
| Report, README, demo prep | ≈4–6 h |
| **Core total** | **≈12–18 h** (2–3 focused days) |

Most of the time goes into running, debugging across machines, waiting on experiment repeats,
and understanding the code well enough to explain it (halo, idempotence, failure detection).

## Out of scope

Numba/CuPy (future work only), clock synchronization, Lamport timestamps, gossip membership,
distributed mutual exclusion algorithms.
