# Measurement-log format

The measurement log is the versioned, estimator-independent boundary between normal acquisition and offline surface MLE. It preserves the exact finalized inputs available before the live-estimator update, so count and spectral reconstructions can be rerun without simulator state or another repository.

See the [README](../../README.md) for acquisition/replay commands and [MLE architecture](architecture.md) for the forward model.

## Creating a log

Add `--measurement-log-dir` to either supported non-GUI acquisition entry point:

```bash
uv run python main.py \
  --python-cui \
  --headless \
  --max-steps 8 \
  --measurement-log-dir results/measurement-logs/analytic-run

uv run python main.py \
  --geant4-cui \
  --headless \
  --max-steps 8 \
  --measurement-log-dir results/measurement-logs/geant4-run
```

The target must not already exist. This is intentional: a run never merges with or silently replaces an older generation. Choose a new directory name, or move/delete an old log explicitly before acquisition.

The normal runtime configurations use `spectrum_count_method: response_poisson`, so their logs contain both raw spectra and processed isotope counts suitable for:

```bash
uv run estimate-radiation-mle replay --run-dir RUN_DIR
uv run estimate-radiation-mle fit-spectrum --run-dir RUN_DIR
```

## Schema version

The implemented public schema version is `1`, defined by `MEASUREMENT_LOG_SCHEMA_VERSION`. Loaders require an exact version match; there is no best-effort interpretation of unknown schemas.

A finalized directory contains:

```text
RUN_DIR/
├── run_manifest.json
├── runtime_config.resolved.json
├── environment.json
├── observations.npz
├── observation_metadata.jsonl
├── upstream_pf_commit.txt
└── truth_sources.json                 # optional
```

The historical `upstream_pf_commit` field/file name is retained as part of schema v1. Its content identifies the source commit of the locally vendored runtime snapshot. It is provenance only; loading a log does not locate, import, compare, or synchronize another checkout.

## Run manifest

`run_manifest.json` is canonical, strict JSON with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | currently exactly `1` |
| `upstream_pf_commit` | string | vendored snapshot provenance |
| `runtime_config_sha256` | lowercase hex string | SHA-256 of the exact canonical resolved-config file |
| `source_rate_model` | string | required by replay to be `detector_cps_1m` |
| `isotopes` | string array | unique channel names in persisted tensor order |
| `environment` | object | environment payload, duplicated exactly in `environment.json` |
| `obstacle_layout_path` | string or null | portable repository-relative provenance path |
| `source_layout_path` | string or null | portable repository-relative provenance path |
| `sim_backend` | string | for example `analytic` or `geant4` |
| `spectrum_count_method` | string | finalized spectrum-to-isotope extraction method |
| `metadata` | object | run-level metadata |
| `record_count` | positive integer | number of rows `M` |
| `energy_bin_count` | positive integer | number of spectrum bins `B` |

Normal acquisition embeds room dimensions, detector position, environment mode, and the serialized obstacle grid in the environment payload. It also records `runtime_repository: 3D_estimation` and `vendored_runtime_snapshot: true` in run metadata.

## Context files

### `runtime_config.resolved.json`

This is the strict, canonical JSON configuration actually resolved by the runtime, augmented with the source-rate model, candidate isotope order, measurement/adaptive-dwell settings, environment mode, simulator backend, and spectrum count method.

The manifest SHA-256 is calculated over the exact file representation. The loader hashes the file bytes and rejects any mismatch. A `RunContext` also refuses to save if its configuration has changed since its hash was computed.

### `environment.json`

This contains the environment object used by replay. The loader requires it to equal the manifest's embedded environment object exactly. Normal logs embed `obstacle_grid`, avoiding dependence on the original layout file.

If an older/manual log references an obstacle layout instead, standalone replay accepts only a relative path without parent traversal. It searches inside the run directory and then this repository's local `obstacle_layouts/`; it rejects paths that escape those roots. A configured transport-response model path receives equivalent run-local/repository-local containment checks before its JSON is inlined.

### `upstream_pf_commit.txt`

This newline-terminated string must equal `run_manifest.json.upstream_pf_commit`. It mirrors local snapshot provenance and has no operational upstream behavior.

### `truth_sources.json`

When known simulation truth is available, this optional strict JSON array records source isotope, position, and `intensity_cps_1m` for evaluation. It is not used to build the MLE likelihood or initialize the reconstructed map. Absence is valid for real acquisitions or blind evaluation.

## Observation array schema

Let:

- `M = record_count`;
- `B = energy_bin_count`;
- `I = len(isotopes)`.

`observations.npz` is a deterministic, uncompressed ZIP/NPY archive with no pickle/object arrays. Its members are:

| Member | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `step_id` | `int64` | `M` | unique finalized simulator step ID |
| `station_id` | `int64` | `M` | acquisition station/pose grouping |
| `detector_pose_xyz` | `float64` | `M × 3` | detector position in metres |
| `detector_quat_wxyz` | `float64` | `M × 4` | nonzero detector quaternion |
| `fe_orientation_index` | `int64` | `M` | Fe shield orientation |
| `pb_orientation_index` | `int64` | `M` | Pb shield orientation |
| `live_time_s` | `float64` | `M` | finalized measurement live time |
| `travel_time_s` | `float64` | `M` | robot travel charged to the step |
| `shield_actuation_time_s` | `float64` | `M` | shield movement charged to the step |
| `energy_bin_edges_keV` | `float64` | `B + 1` | common, strictly increasing energy edges |
| `spectrum_counts` | `float64` | `M × B` | non-negative finalized raw spectra |
| `spectrum_variance` | `float64` | `M × B` | variance rows, with NaN storage where absent |
| `spectrum_variance_present` | `bool` | `M` | whether each complete variance row exists |
| `isotope_counts` | `float64` | `M × I` | processed isotope counts, NaN where absent |
| `isotope_counts_present` | `bool` | `M × I` | per-channel count presence |
| `isotope_counts_record_present` | `bool` | `M` | distinguishes a missing mapping from an empty/partial mapping |
| `isotope_count_covariance` | `float64` | `M × I × I` | extraction covariance, NaN where absent |
| `isotope_count_covariance_present` | `bool` | `M × I × I` | per-entry covariance presence |
| `isotope_count_covariance_record_present` | `bool` | `M` | whether each record supplied a covariance mapping |

Consumers must use the presence masks rather than treating stored NaNs as numerical observations. The loader reconstructs absent optional values as `None`.

The persistence layer can represent partial isotope mappings, but MLE conversion intentionally requires a stronger all-history contract: if isotope counts are present, every record must contain exactly every manifest isotope; if covariance is present, every record must contain a complete `I × I` mapping. Normal runtime logging writes these complete mappings.

## Per-observation metadata

`observation_metadata.jsonl` contains exactly one compact JSON object per NPZ row, in the same order:

```json
{"metadata":{"measurement_record_finalized_before_pf_update":true},"station_id":0,"step_id":0}
```

Each object has:

- `station_id`, which must match `observations.npz`;
- `step_id`, which must match `observations.npz`;
- `metadata`, a strict JSON object containing simulator metadata plus finalized processing metadata.

Normal runtime records add:

- `measurement_record_finalized_before_pf_update: true`;
- `count_variance_by_isotope`;
- `spectrum_count_method`.

Simulator metadata can additionally retain transport or adaptive-dwell diagnostics. Metadata is descriptive; core geometry, spectra, timing, counts, and covariance stay in typed NPZ arrays.

## Finalized record semantics

A `MeasurementRecord` is constructed only after the observation is ready for estimator use. It includes:

- the detector pose and both selected shield orientations;
- actual accumulated live time after adaptive dwell;
- travel and shield-actuation time;
- the merged raw spectrum and its exact energy edges;
- optional spectrum variance;
- finalized processed isotope counts;
- optional complete isotope-count covariance;
- merged simulator/processing metadata.

Values must be finite, times/counts/variances non-negative, spectra non-empty, energy edges strictly increasing, and quaternions nonzero. Within a log, step IDs are unique, spectrum shapes agree, and energy-bin edges are exactly identical.

This placement is important: the log stores the same finalized observation that is about to enter the live estimator, not an earlier simulator fragment and not a reconstruction synthesized later from estimator state.

## Durable append ordering

`MeasurementLogRecorder` separates per-record durability from final public publication.

### Recorder initialization

For target `parent/run-name`, the recorder:

1. refuses creation if `parent/run-name` already exists;
2. creates a private sibling directory named like `parent/.run-name.inprogress-*`;
3. writes and fsyncs canonical `run_context.json`;
4. creates the private `records/` directory and fsyncs its entry in the staging parent.

### Each append

For each finalized record, the recorder:

1. validates the new record against context and all already staged records;
2. writes a one-record `observation.npz` and `metadata.json` into a temporary shard directory;
3. flushes/fsyncs each file;
4. fsyncs the temporary shard directory;
5. atomically renames it to a stable name such as `00000000-step-00000042`;
6. fsyncs the shard parent;
7. only then returns to the acquisition loop.

The acquisition code calls this append before constructing/mutating the live-estimator measurement state. A process failure cannot make an un-fsynced estimator input appear to have been durably logged.

### Final publication

`finalize()` calls the public saver with the validated in-memory record order. That saver:

1. creates another sibling directory named like `.run-name.tmp-*`;
2. writes every public JSON/JSONL/NPZ member with exclusive creation;
3. flushes and fsyncs each file;
4. fsyncs the completed temporary directory;
5. atomically renames the complete directory to `run-name`;
6. fsyncs the target's parent;
7. removes the private in-progress record shards and fsyncs the parent again.

The public target therefore appears as one complete generation, never as a partially populated directory.

## Failure behavior

- On normal completion with records, the log is finalized and its path is printed.
- On normal completion with no finalized observation, the empty recorder staging is removed and no public log is created.
- If the acquisition raises after one or more records completed, the runtime attempts to publish a log containing those completed records before re-raising the original failure.
- If acquisition raises before the first record completes, no public log is published; the context-only private staging directory can remain for inspection.
- If that failure-path finalization also fails, the private `.inprogress-*` staging directory is deliberately retained and its path is printed. No automatic recovery CLI is currently exposed; preserve the directory for inspection.
- The public saver removes its own temporary `.tmp-*` directory if publication fails.
- Existing public targets are never replaced by measurement logging.

The append-before-update invariant protects completed estimator inputs; it does not claim transactional rollback of simulator or live-estimator side effects outside the log.

## Determinism and integrity validation

For identical context and records on the same Python/NumPy representation, public log bytes are deterministic:

- JSON keys are sorted, values are strict (no NaN/infinity), indentation/newlines are fixed;
- JSONL keys and separators are fixed;
- NPZ members use fixed insertion order, uncompressed storage, fixed 1980 ZIP timestamps, fixed permissions, and `allow_pickle=False`.

`load_measurement_log` validates before returning immutable records:

- all required files exist;
- schema version is supported;
- resolved-config SHA-256 matches;
- environment file matches the manifest;
- provenance text matches the manifest;
- manifest record/bin counts are positive;
- every NPZ member has its exact expected shape and loadable dtype;
- metadata line count and identifiers align with NPZ rows;
- optional truth is a JSON array of objects;
- record-level physical validations and cross-record energy/step constraints hold.

Replay adds further checks for source-rate semantics, exact isotope ordering, local path containment, environment dimensions, physical observation-model construction, surface geometry, and response shapes.

## Portability

A normal log is portable together with this standalone repository because it embeds the resolved runtime configuration, environment, and obstacle grid. Relative source/obstacle layout paths remain provenance aids; truth sources are separately persisted when available.

Portability does not mean the log contains a second copy of every large executable or Python module. Replay uses the versioned local implementation in this repository, while `upstream_pf_commit.txt` identifies the historical vendored snapshot used during acquisition. There is no sibling checkout lookup or synchronization step.
