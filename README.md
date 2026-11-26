# Distributed Image Processing

CSC410 Distributed Systems project: apply image filters to large images by splitting them into
tiles and processing the tiles in parallel across several worker nodes, with a coordinator that
schedules work, detects failed workers and re-assigns their tiles.

See [PLAN.md](PLAN.md) for the full design and build steps.

## Layout

| Folder | Contents |
|---|---|
| `common/` | Filters, tiling (split/merge with halo), gRPC `.proto` |
| `coordinator/` | Coordinator server: job registry, scheduler, failure detection, merging |
| `worker/` | Worker process: pulls tiles, filters them, sends heartbeats |
| `client/` | Command-line client |
| `baseline/` | Sequential (single-machine) reference implementation |
| `benchmarks/` | Experiment runner and plotting |
| `tests/` | pytest tests |
| `data/` | Input images and outputs (not committed) |

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv   # Python 3.11+
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run tests:

```powershell
pytest
```

Run instructions for the coordinator, workers and client will be added as they are built.
