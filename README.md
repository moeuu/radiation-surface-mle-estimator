# 3D Estimation

3D Estimation is a standalone rotating-shield radiation simulation and surface-reconstruction repository. It contains two connected workflows:

1. `main.py` runs the local acquisition stack with either the analytic Python transport backend or the native Geant4 backend. It can persist every finalized observation as a versioned measurement log.
2. `estimate-radiation-mle` replays that log and fits a non-negative, all-history Poisson maximum-likelihood map on the room and obstacle surfaces.

The MLE estimates isotope-specific surface density in detector `cps@1m/m²`. This is a detector count-rate-equivalent strength density, not activity in Bq. Multiplying a patch density by its exact area gives that patch's integrated detector `cps@1m` strength.

For implementation details, see [MLE architecture](docs/mle/architecture.md) and [measurement-log format](docs/mle/measurement_log.md).

## Standalone repository guarantee

This repository is an independent checkout. Installation, testing, acquisition, replay, and reporting do not require a particle-filter sibling checkout. There is no sibling runtime, path, import, package, submodule, or synchronization dependency. Simulation backends, detector and shield physics, spectrum processing, obstacle/source assets, the live estimator, the MLE, and the Geant4 sidecar source all live here.

[COMMON_RUNTIME_SNAPSHOT.json](COMMON_RUNTIME_SNAPSHOT.json) and [UPSTREAM_PF_COMMIT](UPSTREAM_PF_COMMIT) record where the vendored runtime snapshot came from. They are provenance only: they are not an update mechanism, a compatibility check against another checkout, or a runtime dependency. Changes in another repository do not flow into this one automatically.

Audit this property at any time:

```bash
uv run python scripts/check_standalone.py
uv run python scripts/check_standalone.py --json
```

The audit rejects external dependency specifications, external or broken symlinks, Git submodules, direct sibling paths and runtime sync hooks; it also checks that the required vendored artifacts exist and local Python sources parse.

## Requirements and setup

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Analytic runs: no external simulator
- Geant4 runs: a local Geant4 installation exposing `geant4-config`, plus `g++` with C++17 support
- CUDA is optional and is used only when MLE response construction is explicitly run with `--gpu`

Create the environment and install the locked project dependencies:

```bash
uv sync
```

Run the repository checks:

```bash
uv run python scripts/check_standalone.py
uv run pytest
```

The supported test command is pytest; the project configuration discovers tests under `tests/`.

## Create a measurement log

`--measurement-log-dir` can be added to a normal local acquisition run. The destination must be a new path. A completed run publishes a self-contained directory suitable for count-domain or spectral replay.

### Analytic Python backend

```bash
uv run python main.py \
  --python-cui \
  --headless \
  --environment-mode fixed \
  --obstacle-config obstacle_layouts/Ex1_obstacles.json \
  --source-config source_layouts/Ex1.json \
  --max-steps 8 \
  --measurement-log-dir results/measurement-logs/analytic-ex1
```

`--python-cui` selects the local analytic backend and its checked-in `configs/python/high_fidelity_no_isaac.json` runtime configuration. The explicit fixed layouts make the example reproducible; random environments and surface-constrained random sources remain available through `main.py --help`.

### Geant4 backend

Build the local sidecar once:

```bash
uv run python scripts/build_geant4_sidecar.py
```

Then run the standard no-GUI Geant4 entry point:

```bash
uv run python main.py \
  --geant4-cui \
  --headless \
  --environment-mode fixed \
  --obstacle-config obstacle_layouts/Ex1_obstacles.json \
  --source-config source_layouts/Ex1.json \
  --max-steps 8 \
  --measurement-log-dir results/measurement-logs/geant4-ex1
```

`--geant4-cui` uses the checked-in `configs/geant4/variance_reduction_external_no_isaac_32threads.json` configuration and auto-starts `build/geant4_sidecar`. Running `main.py` without a mode also defaults to `geant4-cui`, but the explicit flag is recommended in scripts.

Each observation is finalized after adaptive-dwell merging and spectrum processing, then durably staged before the local live-estimator update. On successful completion the staging data is consolidated and atomically published. The log contains raw spectra, processed isotope counts and covariance when available, detector and shield state, timing, resolved runtime/environment data, optional truth for evaluation, and snapshot provenance. See [measurement-log format](docs/mle/measurement_log.md) for the exact schema and failure behavior.

## Replay with surface MLE

The installed CLI has three subcommands:

```text
estimate-radiation-mle replay
estimate-radiation-mle fit-spectrum
estimate-radiation-mle report
```

### Count-domain replay

`replay` fits the processed `response_poisson` isotope-count channels:

```bash
uv run estimate-radiation-mle replay \
  --run-dir results/measurement-logs/analytic-ex1 \
  --mle-config configs/mle/default_count.json \
  --output-dir results/mle/analytic-ex1-count \
  --cpu
```

### Line-resolved spectral replay

`fit-spectrum` fits the raw energy-bin counts with line-specific shield and obstacle attenuation plus the detector pulse response:

```bash
uv run estimate-radiation-mle fit-spectrum \
  --run-dir results/measurement-logs/analytic-ex1 \
  --mle-config configs/mle/default_spectral.json \
  --output-dir results/mle/analytic-ex1-spectral \
  --cpu
```

The command determines its mode: `replay` forces count mode and `fit-spectrum` forces spectral mode even if the supplied JSON contains another `mode`. A supplied configuration's `isotope_names` and order must exactly match the log. If `--mle-config` is omitted, the CLI derives the isotope order from the log and uses `MLEConfig` defaults for that mode.

If `--output-dir` is omitted, results go to `RUN_DIR/mle_count` or `RUN_DIR/mle_spectral`. Existing MLE report files are protected; pass `--overwrite` to replace them. Use `--no-debias` to skip the support-selected unregularized refit, and `--json` for a machine-readable fit summary.

### Read a saved report

```bash
uv run estimate-radiation-mle report \
  --estimate results/mle/analytic-ex1-count

uv run estimate-radiation-mle report \
  --estimate results/mle/analytic-ex1-count/mle_estimate.npz \
  --json
```

`report` validates and summarizes an existing result; it does not refit the data.

## Model in brief

The environment is tiled into exact rectangular floor, ceiling, wall, obstacle-top, and exposed obstacle-side patches. Covered floor areas and internal obstacle faces are excluded. Each patch carries exact vertices and area, one-point centroid or four-point Gauss quadrature, a stable ID, and physical shared-edge adjacency.

For patch `g`, isotope `i`, and density `rho[g,i]`:

```text
integrated patch strength s[g,i] = area[g] * density[g,i]
density unit                         detector cps@1m/m²
integrated-strength unit             detector cps@1m
```

Quadrature weights are normalized within each patch, while area is applied separately. Count response construction starts as `M × G × I`; spectral response construction produces the line-resolved `M × B × G × I` tensor for measurements, energy bins, patches, and isotopes.

The convex objective combines Poisson negative log likelihood with optional:

- area-weighted L1 sparsity;
- shared-edge-length-weighted total variation (TV);
- an area-weighted isotope group penalty per patch;
- non-negative count-background or spectral background/scatter nuisance terms with L1/L2 penalties.

Optional coarse-to-fine levels refine strong patches into four children and transfer density as a warm start. The default debias stage selects support from the regularized map, fixes all other response columns to zero, and refits without L1, TV, or group shrinkage.

Diagnostics include objective terms and history, convergence changes, KKT residual, Poisson deviance, full residuals, optional deterministic held-out deviance, response rank/conditioning/correlation checks, and connected surface hotspot clusters.

## CPU and GPU

The checked-in MLE configurations use CPU response construction. `--cpu` forces that path. `--gpu` enables the local PyTorch CUDA implementation of the physical response kernel; the NumPy/SciPy optimization itself remains on CPU.

GPU use is explicit and strict: if the configured device (default `cuda`) is unavailable, the run fails rather than silently changing devices. `gpu_device`, `gpu_dtype` (`float32` or `float64`), and `response_chunk_size` are JSON configuration fields. Chunking bounds kernel-evaluation working memory, although the complete response tensor is still materialized for the solver.

## Result files

Every successful replay writes:

- `mle_estimate.npz`: numerical estimate, patch geometry/lineage/adjacency, predictions, fitted nuisance parameters, and summary scalars;
- `mle_diagnostics.json`: diagnostics plus the fully resolved MLE configuration;
- `hotspot_clusters.json`: connected hotspot summaries when cluster diagnostics are present.

Reports are deterministic, pickle-free, and staged before publication. The NPZ stores a SHA-256 binding to the diagnostics JSON, and loading validates that binding plus patch adjacency and mirrored metadata.

## Repository map

- [main.py](main.py): analytic and Geant4 acquisition entry point
- [configs/mle](configs/mle): count, spectral, and L1/TV examples
- [src/runtime/measurement_log.py](src/runtime/measurement_log.py): versioned log persistence and validation
- [src/three_d_estimation](src/three_d_estimation): replay, surface patches, response builders, solver, diagnostics, and reports
- [docs/mle/architecture.md](docs/mle/architecture.md): forward model and optimization architecture
- [docs/mle/measurement_log.md](docs/mle/measurement_log.md): durable log lifecycle and schema
- [scripts/check_standalone.py](scripts/check_standalone.py): standalone dependency audit
