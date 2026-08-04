# Historical GT measurement tables

These CSV files are preserved as legacy research artifacts from the former
`moeuu/Radiation_distribution_machine_learning` repository at commit
`d07b4ead`. Line endings were normalized to LF during migration; numeric
values and marker columns were retained.

| File | Original Git blob |
| --- | --- |
| `result.csv` | `6b75ef3892ee204431b3c0e61c9f4180a4eaffee` |
| `result_3points.csv` | `aae4736d96402fe4ec2e3ce17bc47714f64a7a24` |
| `result_5points.csv` | `5c9e33919aba6ec95c8af2b5a4010ee826cb0b95` |
| `result_noshield.csv` | `d2bf0258bb33377f0d9f2c3fc887504fc3a198a4` |

The shield-aware files use `x`, `y`, `z`, `value`, and `shield`.
The three- and five-source tables also retain the original unlabeled marker
column containing `*` on selected rows.

These tables are not MeasurementLog v2 inputs and are not consumed by the
maintained estimator example. Use
`../shield_aware_surface_mle.ipynb` for the current executable workflow.
