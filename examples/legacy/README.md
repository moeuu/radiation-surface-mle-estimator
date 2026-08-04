# Legacy IAS research snapshot

This directory is an archive, not part of the installable online MLE package.

- `shield_aware_surface_mle.ipynb` preserves the pre-runtime IAS workflow.
- `data/` preserves the original GT tables, count CSV files, and rendered frames.
- `figures/` preserves the former repository-root diagnostic figures.

The complete Python implementation immediately before the runtime/PF refactor remains
recoverable from the Git tag `legacy-ias-mle-before-pf-runtime-import`. Production code
must not import this archive: its detector, shield, measurement, and synthetic-observation
logic predates the shared `rotating-shield-simulation-runtime` physical contract.
