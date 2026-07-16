# Shared MeasurementLog v1 conformance fixture

`measurement_log/` is a truth-free, estimator-independent log accepted by the
standalone pure-PF and pure-MLE replay contracts. It contains twelve causal
rows at four stations/heights, three Fe/Pb views per station, raw spectra,
spectrum variances, and complete three-isotope counts/covariance. The exact
MeasurementLog inventory digest is
`b52a0066f47d83bcd44aebfc97ff213f3200477c3a8d3f722fee40b2a29c7a69`.

Regenerate the deterministic binary observation member after moving the
existing file aside with:

```bash
uv run python scripts/create_shared_measurement_log_fixture.py \
  --output /tmp/shared-observations.npz
```

The fixture intentionally contains no particles, PF modes, candidate source
positions, planner state, or truth artifact. Count and spectral MLE tests both
consume the same bytes.

Run both standalone providers with the checked-in fast conformance config:

```bash
uv run estimate-radiation-mle replay \
  --run-dir fixtures/shared_measurement_log_v1/measurement_log \
  --mle-config configs/mle/shared_fixture_fast.json \
  --output-dir /tmp/shared-count-mle

uv run estimate-radiation-mle fit-spectrum \
  --run-dir fixtures/shared_measurement_log_v1/measurement_log \
  --mle-config configs/mle/shared_fixture_fast.json \
  --output-dir /tmp/shared-spectral-mle
```
