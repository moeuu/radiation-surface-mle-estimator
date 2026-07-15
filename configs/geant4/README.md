# Geant4 Runtime Configs

Use these top-level configs for current simulations:

- `variance_reduction_external_no_isaac_32threads.json`: standard no-GUI
  Geant4/PF full simulation config. This is the default for `main.py`,
  `--cui`, and `--full-simulation`. It loads the repo-stable PF
  transport-response calibration in `calibration/` for the
  incident-gamma-energy variance-reduction runtime and uses a conservative
  response-Poisson line-basis BIC margin for count extraction. The
  transport-response calibration uses optical-depth feature caps to prevent
  polynomial extrapolation in very rare high-shield-depth aperture rays.
- `variance_reduction_external_gui_32threads.json`: standard Geant4/PF full
  simulation config plus an Isaac Sim sidecar for visualization. It inherits
  the no-GUI config; the intended runtime difference is Isaac Sim startup only.
- `high_fidelity_external_no_isaac.json`: full-transport verification config.
  It is intentionally slower, does not inherit the variance-reduction
  transport-response calibration, and should be selected explicitly.
- `external_gui_scene.json`: explicit USD-backed Manchester Drum Store scene.
- `shield_validation_scene.json`: material/shield validation config.

Older duplicated or scene-specific configs are isolated under `legacy/`.
Do not select those from standard runtime entry points. If a legacy config is
needed for a benchmark or reproduction, keep it explicit in the command and do
not promote it back to the top-level runtime set.
