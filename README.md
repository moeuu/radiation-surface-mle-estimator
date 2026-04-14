# 3D Estimation

This repository estimates radiation intensity on the six surfaces of a room from shielded measurements sampled near the floor.

## Environment

This project is managed with `uv`.

Create the environment and install dependencies with:

```bash
uv sync
```

## Run

Run the default configuration without plots:

```bash
uv run estimate-radiation
uv run python -m three_d_estimation
```

Useful options:

```bash
uv run estimate-radiation --max-iter 500 --seed 7 --measurement-ratio 0.5
uv run estimate-radiation --plot
uv run estimate-radiation --json
```

The command prints:

- the random seed
- the number of sampled measurement points
- the number of shield orientations
- the `A` matrix shape
- the flattened `q` shape
- optimizer settings
- the final Poisson log-likelihood score

## Project layout

- `src/three_d_estimation/config.py`: runtime configuration dataclasses and defaults
- `src/three_d_estimation/cli.py`: command-line entrypoint
- `src/three_d_estimation/pipeline.py`: end-to-end estimation pipeline
- `src/three_d_estimation/geometry.py`: grid and face-shape utilities
- `src/three_d_estimation/measurement.py`: attenuation, measurements, and system matrix generation
- `src/three_d_estimation/optimization.py`: likelihood, gradient, and Adam optimizer
- `src/three_d_estimation/priors.py`: prior-update and MAP helper logic
- `src/three_d_estimation/plotting.py`: plotting helpers
- `src/three_d_estimation/animation.py`: animation helpers
- `scripts/`: Python conversions of the former notebook workflows

## Tests

Run the unit tests with:

```bash
uv run python -m unittest discover -s tests
```

The tests cover reproducible measurement sampling, grid generation, matrix shapes, round-trip face restoration, and a small end-to-end pipeline run.

## Notes

- Notebook-owned helper functions have been moved into `src` modules, and obsolete notebooks were removed.
- Former notebook workflows now live as Python scripts under `scripts/`.
- Grid generation now uses the configured room dimensions instead of hard-coded `10 x 10 x 10`.
- Optimizer routines no longer trigger plotting as a side effect.
