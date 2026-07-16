# MeasurementLog v1

`MeasurementLog` is the estimator-independent boundary between acquisition and
standalone surface MLE. It contains finalized measurements and resolved physical
model metadata, never particle state, PF candidates, estimator output, or truth.

See the [README](../../README.md) for commands and
[MLE architecture](architecture.md) for reconstruction details.

## Directory contract

Schema version 1 contains exactly the estimator inputs below:

```text
RUN_DIR/
├── run_manifest.json
├── runtime_config.resolved.json
├── environment.json
├── forward_model_manifest.json
├── observations.npz
├── observation_metadata.jsonl
└── repository_commit.txt
```

`repository_commit.txt` is the canonical provenance filename. The loader accepts
the historical `upstream_pf_commit.txt` only as a reader alias for existing local
logs; new logs never write it.

Truth is forbidden below `RUN_DIR`. Simulation truth belongs in a separate
evaluation directory, for example:

```text
EVALUATION_DIR/
└── truth.json
```

`save_evaluation_truth` and `load_evaluation_truth` manage that separate artifact.
Neither `load_measurement_log` nor replay reads it.

The boundary is fail-closed beyond the canonical filename: any artifact path
whose normalized name contains `truth` or `source-layout` is rejected. Run,
environment, context, and per-record metadata are scanned recursively for
truth, realized source layouts/positions/coordinates/locations, and source
lists. Strings pointing to those artifacts are rejected as well. Source-rate,
source-strength, and source-extent model parameters remain valid physics.

## Run manifest

`run_manifest.json` includes:

- `schema_version`, exactly `1`;
- non-empty `run_id`;
- `repository_commit`, matching `repository_commit.txt`;
- `resolved_config_sha256`, matching the exact resolved-config file bytes;
- `forward_model_manifest_sha256`, matching the exact forward manifest bytes;
- `source_rate_model`, exactly `detector_cps_1m`;
- `source_rate_semantics`, exactly
  `{quantity: expected_net_detector_count_rate, unit: cps,
  normalization_distance_m: 1.0}`;
- all six `model_identifiers` entries: detector, shield, environment, obstacle,
  transport, and spectrum, each with a non-empty `id` and lowercase SHA-256;
- exact `index_conventions` for causal step order, unique actions, and
  nondecreasing station groups;
- `artifact_hashes` for every estimator-input file other than the manifest
  itself;
- isotope order, environment, simulator/spectrum method, an optional relative
  obstacle-layout path, metadata, and record/bin counts;
- `source_layout_path`, retained only as a schema-v1 compatibility sentinel
  whose value must be `null`.

Unknown schema versions, missing artifacts, hash mismatches, incompatible source
semantics, and model-identity mismatches fail before records are returned.

## Forward-model manifest

`forward_model_manifest.json` binds replay to physical semantics. It records:

- repository and resolved-config identity;
- the six model IDs and hashes;
- distance `m`, time `s`, energy `keV`, source-strength
  `detector_cps_1m`, and linear-attenuation `cm^-1` units;
- inverse-square/near-field distance behavior;
- detector geometry binding;
- the 8 × 8 Fe/Pb orientation-pair response;
- line-segment obstacle attenuation;
- linear live-time scaling;
- energy-bin-integrated isotope-line response.

The registered fixture additionally carries the exact production
`line_mu_by_isotope` table. Every ordered line records energy, normalized
weight, Fe attenuation, and Pb attenuation. Its shield-model hash binds the
full table; its spectrum-model hash binds the energy/weight subset.

Native logs are checked against a manifest derived from their complete local
resolved configuration. The versioned
`rotating-shield-analytic-conformance-v1` fixture is the only registry binding:
its exact six IDs/hashes and every semantic field are fail-closed. That binding
is accepted because the local implementation is verified numerically over all
4,608 canonical forward-response cases. An unknown registry ID or any changed
field is rejected; registry matching is not a general fallback.

Every file-backed obstacle, transport, detector, or spectrum asset selected by
replay must use a safe relative path resolved below the run directory or this
standalone repository. The corresponding component payload binds both that
portable path and the SHA-256 of the raw asset bytes. A file changed in place
therefore produces a model-identity mismatch even when its configured path is
unchanged. Registered conformance manifests cannot reference arbitrary
file-backed assets because their hashes are fixed by the registry.

Generate the numerical provider output with:

```bash
uv run estimate-radiation-mle forward-conformance \
  --axes fixtures/forward_response_conformance.json \
  --output results/mle_forward_response.npz
```

The output contains exactly `case_ids` and `unit_response`. Ordering is isotope,
listed detector pose, Fe 0–7, Pb 0–7, listed source, then listed obstacle.

## Observation arrays

Let `M` be records, `B` energy bins, and `I` isotopes. `observations.npz` is a
deterministic, uncompressed, pickle-free archive with these exact members:

| Member | Dtype | Shape |
| --- | --- | --- |
| `step_id`, `action_id`, `station_id` | `int64` | `M` |
| `detector_pose_xyz` | `float64` | `M × 3` |
| `detector_quat_wxyz` | `float64` | `M × 4` |
| `fe_orientation_index`, `pb_orientation_index` | `int64` | `M` |
| `live_time_s`, `travel_time_s`, `shield_actuation_time_s` | `float64` | `M` |
| `energy_bin_edges_keV` | `float64` | `B + 1` |
| `spectrum_counts`, `spectrum_variance` | `float64` | `M × B` |
| `spectrum_variance_present` | `bool` | `M` |
| `isotope_counts` | `float64` | `M × I` |
| `isotope_counts_present` | `bool` | `M × I` |
| `isotope_counts_record_present` | `bool` | `M` |
| `isotope_count_covariance` | `float64` | `M × I × I` |
| `isotope_count_covariance_present` | `bool` | `M × I × I` |
| `isotope_count_covariance_record_present` | `bool` | `M` |

Presence masks are authoritative. Absent numeric values must be stored as NaN;
present values must be finite. Counts and variances are non-negative. Every
stored covariance is complete, symmetric, finite, and positive semidefinite.

Live time is strictly positive. Travel and shield-actuation times are
non-negative. Fe/Pb indices are integers in `[0, 7]`. Quaternions must be finite,
nonzero, and normalized within `rtol=1e-9`, `atol=1e-12`. Energy edges are finite
and strictly increasing. Steps are strictly increasing, actions are unique, and
stations are nondecreasing in causal file order.

## Per-row metadata

`observation_metadata.jsonl` has one compact JSON object per array row with
exactly these keys:

```json
{"action_id":0,"array_index":0,"metadata":{},"run_id":"run-1","station_id":0,"step_id":0}
```

Every identifier must match the corresponding NPZ row. `metadata` may hold
descriptive acquisition fields such as `shield_program_block_id`; numerical
geometry, shields, spectra, covariance, and timing remain in typed arrays.

## Content digest and provenance

`measurement_log_sha256(RUN_DIR)` inventories every regular file recursively as
`{relative POSIX path: raw file SHA-256}`, serializes that mapping with the
runtime canonical JSON encoder plus its terminating newline, and hashes those
bytes. The digest includes `run_manifest.json`. Symlinks, non-regular files, and
truth/source-layout artifact names inside the log fail validation.

MLE provenance records:

- estimator repository and commit;
- estimator family `surface_mle` and variant `count` or `spectral`;
- candidate domain `complete_surface_dictionary`;
- `uses_pf_state: false` and `uses_pf_candidates: false`;
- measurement run/schema/repository identity;
- measurement-log and forward-manifest hashes;
- raw estimator-config file hash (`config_sha256`);
- resolved semantic estimator-config hash
  (`resolved_estimator_config_sha256`).

## Durable publication

`MeasurementLogRecorder` fsyncs a finalized one-record shard before acquisition
may update any live estimator. Finalization validates the complete causal
sequence, writes all public artifacts into a sibling temporary directory,
fsyncs them, and atomically renames the directory into place. Existing targets
are never replaced. Failure after completed records retains or publishes those
inputs for diagnosis; it never fabricates missing rows.

The public representation is deterministic: strict sorted JSON, compact sorted
JSONL, fixed NPZ member order, fixed ZIP timestamps/permissions, no pickle, and
no object arrays.
