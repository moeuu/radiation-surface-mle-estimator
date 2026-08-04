# RA-L MLE closed-loop full simulation

`ral-full-simulation` runs a live surface-MLE-controlled acquisition against the
shared `Rotating-shield-simulation-runtime`. It does not use a precomputed action
plan and it never imports PF code or simulation physics.

## Ownership and private scenario

The runtime owns external Geant4, detector and shield geometry, obstacle transport,
spectra, source realization, raw observations, and MeasurementLog v2 publication.
The private scenario contains only physical-scene and run identity inputs:

- the Cs-137 x4, Co-60 x3, and Eu-154 x2 realized positions and strengths;
- environment and obstacle geometry;
- the authoritative runtime physics configuration;
- the final MeasurementLog output location;
- bookkeeping identity such as run ID, backend, isotope names, and metadata.

An adaptive scenario has no `actions` field. It does not contain a station count, a
views-per-station value, a measurement-position list, or a shield-angle list. Source
truth remains inside the runtime process. The estimator receives only a truth-free
run context, reachable candidate positions, motion costs, and persisted raw spectra.

The standard physical profile remains runtime-owned:

- external multithreaded Geant4 with 32 threads;
- `ral_eu154` with Co-60, Cs-137, and Eu-154;
- 0--1700 keV with 851 bins;
- full secondary transport and native detector-response sampling;
- exact unit-weight histories with no thinning or weighted transport.

## Closed-loop execution

The live loop is:

1. The runtime starts the private scene and publishes reachable truth-free candidates.
2. The first observation is taken at the environment's initial detector position,
   because no data-conditioned MLE exists yet.
3. The runtime durably stages that raw spectrum before returning it.
4. Every spectrum in the current shield program is buffered without an intermediate
   fit.
5. When the program closes its station, one coarse matrix-free MLE refits the complete
   causal history, asks the runtime to refine promising candidate neighborhoods, and
   reranks the returned safe poses.
6. Local Fisher information plus explicit vertical/support ambiguity scores jointly
   select the next position and complete Fe/Pb station program. The program objective
   marginalizes one rate nuisance shared by all views at that point.
7. The runtime executes that program. Every view has the same station ID, and only its
   final record has `station_complete=true`.
8. Steps 3--7 repeat until the compound stop rule passes or an emergency safety bound
   is reached.

`ral_full_planning.json` currently uses `shield_program_length = 2`. This is a short,
newly selected program—not the former fixed eight-view design. Its views are jointly
optimized by station-block beam search rather than selected greedily one at a time.
The estimate remains unchanged until both views are durable and the station closes,
which preserves the statistical block used by fitting, grouped validation, and
bootstrap.

The stop rule is estimator-owned and loaded from `ral_full_stop.json`. It requires all
of the following, rather than information gain alone:

- converged recent fits with bounded KKT residual;
- minimum counts of records, independent 3-D poses, and height levels;
- sufficient source-relative elevation span;
- stable deviance, surface map, and hotspot-cluster centroids over a window;
- bounded response-column correlation and systematic station residuals;
- minimum floor/ceiling response separation;
- low expected information gain for the configured patience window.

`--max-measurements 256` is only a failure-safety bound, not a requested record count
and not an RA-L structural contract.

## Scalable fitting and calibration

The production profile uses a streamed `ResponseOperator`; it never materializes the
complete `M x B x G x I` response tensor. Forward and adjoint products stream over
measurement, energy, and patch chunks. Per-measurement response blocks are cached on
disk under a physics/configuration hash, so extending a causal prefix computes only
new rows. CPU and CUDA implementations share equivalence tests and diagnostics report
peak response-block memory.

Online fits use the coarser `online_patch_spacing_m` grid without refinement. Final
reporting restarts on the configured full grid, performs coarse-to-fine refinement,
debiases the selected support, and computes uncertainty. This prevents the online
latency budget from silently reducing final spatial resolution.

The checked-in profile remains Poisson until a real independent calibration artifact
is supplied; no synthetic correction is embedded in the repository. Produce the
shared artifact in the runtime from at least two independent Geant4 calibration
environments with all 64 shield pairs, then set:

```json
{
  "discrepancy_calibration_path": "/absolute/path/to/calibration.json",
  "spectral_likelihood": "calibrated_overdispersed"
}
```

The MLE then adds calibrated background/scatter, low-dimensional shield leakage,
station-rate, low-rank spectral residual, optional gain/resolution drift nuisance
families with shrinkage, and uses the calibrated negative-binomial dispersion. The
runtime only owns this common calibration contract; all MLE nuisance selection and
likelihood behavior remain in this repository.

## Regularization, uncertainty, and final holdout

`ral_regularization_tuning.json` performs grouped station/same-XY validation over the
L1/TV grid and applies the one-standard-error rule. Tuning and final environment IDs
must differ. The selected weights are then frozen in the final profile.

Final uncertainty includes an active-support Laplace covariance, station-block
bootstrap, patch selection frequencies, cluster centroid and integrated-strength
intervals, isotope/surface mass probabilities, z intervals, and ceiling probability.
The final `ral-holdout` command additionally verifies that tuning, discrepancy
calibration, and the unseen Geant4 evaluation environment are disjoint before fitting.

## Readiness check

```bash
uv run estimate-radiation-mle ral-full-simulation \
  --preflight-only --json
```

This checks the runtime config, Geant4 sidecar, full-spectrum registry and SHA-256,
physical fidelity fields, isotope order, streamed spectral MLE profile, compound stop
profile, and short adaptive shield-program planning profile.

## Start a new MLE-controlled physical acquisition

```bash
uv run estimate-radiation-mle ral-full-simulation \
  --scenario /secure/runtime/ral-mix9-scenario.json \
  --output-dir /home/moeu/research/ral-runs/ral-mix9-mle \
  --dashboard-public-host HOSTNAME \
  --json
```

The MLE process passes only the scenario path to
`rotating-shield-sim run-adaptive-session`; it never opens the private file. The
runtime remains alive across observations. Its JSON-lines protocol accepts one selected
action at a time and publishes the immutable MeasurementLog only when the MLE stops.
The launcher also requests runtime's private `ral-mix9` profile check, so source
cardinalities are enforced without exposing the realized source list to MLE.

For a persistent multi-hour run:

```bash
tmux new-session -d -s ral_mle \
  "cd /home/moeu/research/radiation-surface-mle-estimator && \
   uv run estimate-radiation-mle ral-full-simulation \
   --scenario /secure/runtime/ral-mix9-scenario.json \
   --output-dir /home/moeu/research/ral-runs/ral-mix9-mle \
   > /home/moeu/research/ral-runs/ral-mix9-mle.log 2>&1"
```

## Replay an existing physical log

An already completed compatible RA-L log can be analyzed without acquisition:

```bash
uv run estimate-radiation-mle ral-full-simulation \
  --run-dir /path/to/measurement-log-v2 \
  --output-dir /home/moeu/research/ral-runs/ral-mix9-mle \
  --json
```

Log validation enforces the Geant4/full-transport/isotope contract and causal station
markers. It deliberately does not enforce a station count, views per station, record
count, or fixed live time.
