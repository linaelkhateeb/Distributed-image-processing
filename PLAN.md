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
- [x] 4. Filters (`filters.py`): grayscale, invert, Gaussian blur, sharpen, Sobel edges, median, plus `blur_heavy` (31x31, large halo) and `median_heavy` (15x15, expensive) for experiments. Each filter declares its kernel radius (the halo it needs).
- [x] 5. Tiling (`tiling.py`): `split(image, tile_size, halo)` returns tiles with coordinates and halo; `crop_halo` + `merge` put the cores back. Handles edge tiles and sizes that aren't multiples of the tile size.
- [x] 6. Sequential baseline (`sequential.py`): load, filter, save, time (warm-up run, OpenCV limited to 1 thread). Also `make_sample.py` to generate a test image.
- [x] 7. Correctness tests: split → filter → merge equals the sequential output pixel for pixel, for every filter, 5 tile sizes and 3 image shapes; plus a test showing that without the halo, neighbourhood filters produce seams. 190 tests pass.

### Phase 2 — Distributed system, happy path (≈2–3 h)
- [x] 8. gRPC interface (`common/proto/image_service.proto`, Python code generated automatically by `common/rpc.py`):
  - Worker → coordinator: `RegisterWorker`, `RequestTask` (pull), `SubmitResult(task_id, tile)`, `Heartbeat`
  - Client → coordinator: `SubmitJob`, `GetStatus`, `GetResult`, `ListWorkers`
  - Pixels sent raw (no PNG encoding); message limit raised to 512 MB.
- [x] 9. Coordinator (`coordinator/scheduler.py`, `job.py`, `server.py`): job registry, FIFO task queue, task id `<job_id>:<tile_index>`, assignments tracked; merge when all tiles are in. Duplicate results already ignored (part of step 15).
- [x] 10. Worker: pull → filter → submit loop with a heartbeat thread. **Pipelining (`--prefetch`, default 1)**: the next tile is requested while the current one is filtered and results are submitted in the background, hiding gRPC latency.
- [x] 11. CLI client: `submit` (progress bar, timings, tiles per worker, result + before/after image, `--verify`) and `workers`.
- [x] 12. End-to-end run on one machine: coordinator + 4 worker processes; results match the sequential version pixel for pixel. 218 tests pass.

**Phase 2 findings (one laptop, i5-8300H, 4 cores / 8 threads, 4000x3000 image):**
- A single empty gRPC call costs ~4.5 ms on this machine (Python gRPC on Windows), ~7 ms after a short idle. Each tile needs 2 calls, so with small/cheap tiles workers spent more time waiting than filtering.
- Pipelining (prefetch 1) cut median_heavy with 1 worker from 5.15 s to 2.64 s; with 4 workers from 1.07 s to 0.87 s.
- The single Python coordinator handles ~220 tiles/s, so it becomes the bottleneck with small tiles. Larger tiles help: median_heavy with 4 workers took 1.01 s (256 px) → 0.75 s (512 px) → 0.74 s (1024 px), vs ≈1.1 s sequential (≈1.5x).
- Cheap filters (blur) are always slower distributed (0.33 s vs 0.02 s): moving 36 MB of pixels costs more than the filtering. Expected; this is the cheap-vs-expensive experiment.
- On one machine the coordinator, client and workers compete for 4 cores; some workers run visibly slower. Real speedup numbers should come from the multi-laptop run (step 22).

### Phase 3 — Fault tolerance (≈1.5–2 h)
- [x] 13. Heartbeat monitor: a failure-detector thread in the coordinator marks a worker *suspected dead* after 6 s without a heartbeat (`--heartbeat-timeout`). A suspected worker that speaks again is marked alive.
- [x] 14. Re-assignment: a suspected worker's tiles go back to the **front** of the queue; tiles also have a 30 s lease (`--lease-timeout`) for stuck-but-alive workers. Workers report processing errors (`TaskResult.error`) so the tile is re-sent immediately.
- [x] 15. Idempotence and dedup: first result for a `task_id` wins; later copies are ignored (also covers a "dead" worker that was only slow).
- [x] 16. Retry limit: a tile handed out 3 times (`--max-attempts`) without a result fails its job; other jobs continue. A tile is not given back to a worker that already failed it while another live worker can take it (otherwise one broken worker burns all retries by itself).
- [x] 17. Failure injection: worker flags `--crash-after N` (dies abruptly holding a tile) and `--slow SECONDS` (straggler).
- [x] 18. Fault tests: fake-clock unit tests (`test_fault_tolerance.py`) and real-gRPC tests where workers crash or fail mid-job (`test_worker_failures.py`). Live run: 4 worker processes, one crashes after 20 tiles, one hard-killed mid-job → 3 tiles re-sent, result matches sequential. 234 tests pass.

**Phase 3 findings:**
- Recovery cost ≈ the heartbeat timeout: the live run took 7.4 s instead of ~1 s because the last tiles waited 6 s for the failed workers to be suspected. Shorter timeouts recover faster but risk suspecting workers that are only slow (L10 trade-off; this is the "kill a worker" experiment).
- With pipelining, a result still in flight when a worker dies is lost; the coordinator re-sends that tile, so correctness is unaffected.
- Possible extension (not in scope): speculative "backup" copies of the last in-flight tiles when the queue is empty, as MapReduce does for stragglers.

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
