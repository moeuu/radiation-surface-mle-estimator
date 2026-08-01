# 3D Estimation

This repository contains the surface maximum-likelihood estimator only.

## Repository boundary

The shared `Rotating-shield-simulation-runtime` repository owns:

- Geant4 and analytic transport runtimes;
- environment, obstacle, detector, and rotating-shield physics;
- isotope and full-spectrum response models;
- observation generation; and
- raw full-spectrum MeasurementLog schema version 2.

This repository owns:

- surface-patch construction used by MLE;
- count/spectral response-matrix assembly for the MLE objective;
- regularized optimization and debias refits;
- MLE warm starts, diagnostics, and reports; and
- the public `estimate-radiation-mle` CLI.

There is intentionally no `native/`, `measurement/`, `sim/`, `spectrum/`, PF,
planner, source-layout, obstacle-layout, or MeasurementLog-writer copy here.
MLE imports the installed shared runtime package.

## Replay a shared raw log

The production MeasurementLog v2 contract stores raw integer spectra. Use the
spectral MLE path for that contract:

```bash
uv run estimate-radiation-mle fit-spectrum \
  --run-dir /path/to/measurement-log-v2 \
  --mle-config configs/mle/default_spectral.json \
  --output-dir results/mle-spectral \
  --cpu --json
```

The legacy `replay` command is the count-domain MLE and requires an explicitly
derived count observation contract; it does not invent isotope counts from raw
MeasurementLog v2 spectra.

Acquisition is run from the shared runtime or through the orchestrator, for
example:

```bash
uv run --project ../Rotating-shield-simulation-runtime rotating-shield-sim \
  run-plan --help
```

## Development

```bash
uv sync
uv run python scripts/check_repository_boundary.py
uv run pytest
```
