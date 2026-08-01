# 3D Estimation

3D Estimation is the standalone surface-MLE estimator for rotating-shield radiation
measurements. For new shared experiments, production acquisition is owned by
`Rotating-shield-particle-filter`; this repository consumes its truth-free raw
full-spectrum MeasurementLog v2 and does not require simulation-code synchronization.
It retains the historical local acquisition snapshot only for legacy reproducibility.
It contains two workflows:

1. `estimate-radiation-mle fit-spectrum` consumes the current shared raw-spectrum v2
   log (or a legacy v1 log) and fits a non-negative, all-history Poisson maximum-
   likelihood map on room and obstacle surfaces.
2. The historical `main.py` acquisition snapshot remains available for reproducing
   old standalone runs, but it is not the production source for new common experiments.

The MLE estimates isotope-specific surface density in detector `cps@1m/m²`. This is a detector count-rate-equivalent strength density, not activity in Bq. Multiplying a patch density by its exact area gives that patch's integrated detector `cps@1m` strength.

For implementation details, see [MLE architecture](docs/mle/architecture.md) and [measurement-log format](docs/mle/measurement_log.md).

## Estimator isolation guarantee

Installation, replay, and reporting remain independently testable. Shared runs connect
through serialized MeasurementLog artifacts and subprocess CLIs, never sibling Python
imports, path dependencies, symlinks, or copied source. This keeps the estimator
isolated while making the PF repository the only production simulation implementation.

[COMMON_RUNTIME_SNAPSHOT.json](COMMON_RUNTIME_SNAPSHOT.json) and [UPSTREAM_PF_COMMIT](UPSTREAM_PF_COMMIT) record where the legacy vendored runtime snapshot came from. They are provenance only. Shared production logs carry their own runtime commit and complete forward-model identity, so no manual source synchronization is required.

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

Each observation is finalized after adaptive-dwell merging and spectrum processing, then durably staged before the local live-estimator update. On successful completion the staging data is consolidated and atomically published. The log contains raw spectra, processed isotope counts and covariance when available, detector and shield state, timing, resolved runtime/environment data, forward-model identity, and repository provenance. Truth is stored only in a separate evaluation directory. See [measurement-log format](docs/mle/measurement_log.md) for the exact schema and failure behavior.

## Replay with surface MLE

The installed CLI has six subcommands:

```text
estimate-radiation-mle replay
estimate-radiation-mle fit-spectrum
estimate-radiation-mle report
estimate-radiation-mle forward-conformance
estimate-radiation-mle materialize-prefix
estimate-radiation-mle score-future
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

### Causal hybrid-provider boundaries

Standalone cold replay remains the default and does not depend on PF state. The
orchestrator may additionally request an exact station-complete prefix and initialize
its next count fit from a prior report:

```bash
uv run estimate-radiation-mle materialize-prefix \
  --run-dir results/measurement-logs/analytic-ex1 \
  --cutoff-step 23 --cutoff-station 5 --assert-station-complete \
  --output-dir results/prefixes/through-23

uv run estimate-radiation-mle replay \
  --run-dir results/prefixes/through-23 \
  --mle-config configs/mle/default_count.json \
  --initial-estimate results/mle/prefix-through-15 \
  --output-dir results/mle/prefix-through-23 --cpu
```

`--initial-estimate` supplies optimizer initialization only. The response, objective,
and diagnostics are rebuilt over the complete current prefix; previous objective terms
are not inherited. The output records exact prefix and prior-artifact hashes.

Future-only verification freezes that earlier count model and compares its full
prediction with the same model after zeroing each candidate cluster:

```bash
uv run estimate-radiation-mle score-future \
  --run-dir results/prefixes/through-31 \
  --mle-config configs/mle/default_count.json \
  --snapshot-estimate results/mle/prefix-through-23 \
  --snapshot results/snapshots/through-23.json \
  --output results/scores/through-31.json
```

No parameter is refit on future rows. The reported quantity is a frozen-model count
log predictive likelihood ratio, not a Bayes factor. These commands never accept PF
particles, PF candidate support, or truth; they are provider boundaries for the
separate orchestrator repository.

### Read a saved report

```bash
uv run estimate-radiation-mle report \
  --estimate results/mle/analytic-ex1-count

uv run estimate-radiation-mle report \
  --estimate results/mle/analytic-ex1-count/mle_estimate.npz \
  --json
```

`report` validates and summarizes an existing result; it does not refit the data.

### Generate forward-response conformance output

The provider-neutral axes exercise three isotopes, three detector poses, all 64
Fe/Pb pairs, four source surfaces, and two obstacle conditions:

```bash
uv run estimate-radiation-mle forward-conformance \
  --axes fixtures/forward_response_conformance.json \
  --output results/mle_forward_response.npz
```

The deterministic NPZ contains exactly `case_ids` and `unit_response` for 4,608
unit-strength cases. The versioned manifest registry is accepted only because
this local implementation is bound by that full numerical conformance test.

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

Diagnostics include objective terms and history, convergence changes, KKT residual, Poisson deviance, full residuals, grouped held-out deviance, response rank/conditioning/correlation checks, connected surface hotspot clusters, and complete estimator/log/config provenance. Held-out grouping defaults to whole stations and can instead preserve same-XY height stacks or shield-program blocks; individual-row splitting is explicit diagnostic mode only.

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
