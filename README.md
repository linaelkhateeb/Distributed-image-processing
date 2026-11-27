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

Create the virtual environment from your own terminal (not from inside another tool), so it points
at a Python installation your terminal can see. Python 3.11 or newer works; this project is set up
with 3.14:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation or `python` fails with "No Python at ...", the `.venv` was built with a Python that
no longer exists. Delete it and create it again:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.14 -m venv .venv
```

Every command below assumes the venv is active (your prompt starts with `(.venv)`).

Images: OpenCV reads PNG/JPEG/BMP/TIFF/WebP; other formats such as AVIF are read through Pillow.

Run tests:

```powershell
pytest
```

## Sequential baseline

Generate a 4000x3000 test image, then filter it on one machine:

```powershell
python -m baseline.make_sample data/input/sample.png
python -m baseline.sequential data/input/sample.png data/output/blur.png --filter blur --tile-size 256
```

`--tile-size` also runs the tiled pipeline in one process and checks it matches the sequential
result. OpenCV is limited to 1 thread by default (`--threads` to change), so the baseline is
truly sequential.

Filters: `grayscale`, `invert`, `blur`, `blur_heavy`, `sharpen`, `sobel`, `median`,
`median_heavy`.

## Running the distributed system

Each command runs in its own terminal, with the venv active. The gRPC code is generated
automatically the first time.

**1. Coordinator** (prints the addresses workers can connect to):

```powershell
python -m coordinator.server
```

**2. Workers** (start as many as you like; on the same machine or on others):

```powershell
python -m worker.worker
python -m worker.worker --coordinator 192.168.1.10
```

Useful options: `--id NAME`, `--prefetch N` (task requests kept in flight while filtering;
default 1, `0` disables pipelining), `--verbose` (per-tile timings).

**3. Client:**

```powershell
python -m client.cli workers
python -m client.cli submit data/input/sample.png --filter median_heavy --tile-size 512 --verify
```

`submit` shows a progress bar, the timings and how many tiles each worker did, saves the result
and a before/after image in `data/output/`, and with `--verify` checks the result against the
sequential version. Add `--coordinator HOST[:PORT]` when the coordinator is on another machine.

### Several machines

All machines on the same network, each with the project and venv set up. On the coordinator
machine, allow the port through Windows Firewall once (PowerShell as administrator):

```powershell
New-NetFirewallRule -DisplayName "Image coordinator" -Direction Inbound -Protocol TCP -LocalPort 50051 -Action Allow
```

Then start workers on the other machines with `--coordinator <coordinator IP>`.
