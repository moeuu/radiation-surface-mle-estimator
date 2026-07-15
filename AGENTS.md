# AGENTS.md

## Repository role

This repository is a fully standalone 3-D radiation-estimation application.
It contains its own snapshot of the simulation, environment, detector,
shielding, obstacle-transport, spectrum, planning, and visualization code.
The MLE implementation lives under `src/three_d_estimation/`.

## Repository isolation

- Never import, install, execute, copy from, or write to a sibling repository.
- Do not add path dependencies, Git dependencies, submodules, external
  symlinks, runtime sync scripts, or CI checkouts of another repository.
- `UPSTREAM_PF_COMMIT` is provenance only. It is not a runtime or build input.
- All normal builds, tests, simulations, and replays must work when this is the
  only repository present.

## Git publishing

- Commit and push this repository only on the `main` branch.
- Never create or push `agent/*`, feature, release, or any other branch.
- Before every push, verify that the current branch is exactly `main` and use
  the explicit command `git push origin main`.
- Do not push tags or open pull requests unless the user explicitly requests
  that separate action.

## Physical semantics

- `intensity_cps_1m` is expected net detector cps at 1 m, not total gamma/s.
- MLE code must call the local `RuntimeObservationModel` and
  `ContinuousKernel`; do not duplicate shield, obstacle, detector, or spectrum
  physics in the estimator.
- Preserve spectra, bin edges, variances, detector poses, live times, and Fe/Pb
  orientation indices in measurement logs.
- Production observations must come from a simulator or an imported log, not
  from expected-count substitutions.

## Performance

- Batch response construction across measurements, patches, quadrature points,
  shield pairs, and spectrum bins.
- A scalar implementation is allowed only as a deterministic test oracle.
- Every GPU path needs a CPU/GPU equivalence test.

## Style and tests

- Use Python 3.12 and `uv`.
- Follow PEP 8; every function must have a docstring.
- Comments and docstrings are written in English.
- Run `uv run pytest` before completion.
- Run `uv run python scripts/check_standalone.py` before completion.
