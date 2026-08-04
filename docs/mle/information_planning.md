# MLE information planning

The MLE planner selects a detector station and a sequence of Fe/Pb shield
orientations after a station-complete spectral fit. It deliberately does not use
the particle-filter planner's particle expected-information-gain objective. The
implemented criterion is local Poisson Fisher `D_s`-optimal design under a Laplace
approximation around the current surface MLE.

## Ownership boundary

The shared runtime owns candidate generation, obstacle/traversability checks, route
construction, and travel-cost calculation. It supplies only truth-free candidate
positions and optional allowed shield-pair IDs. This repository evaluates those
candidates with the shared `ContinuousKernel`; it does not copy a simulator, map,
shield model, spectrum model, or PF state.

For `R` shared shield orientations, one pair ID follows the runtime convention:

```text
pair_id = fe_orientation_index * R + pb_orientation_index
```

The selected action contains a `measurement_program` whose rows give the pair ID,
Fe/Pb indices, live time, and `station_complete` flag. A runtime controller can map
those fields directly into its native acquisition actions and remains responsible
for durably staging every resulting `MeasurementRecord` before returning it to the
MLE.

The runtime publishes a nested Sobol sample of the true continuous free volume, with
head/mast/base collision checks and separately calculated motion, settling, and
shield-actuation costs. The MLE may return high-scoring candidate indices in a
generic refinement request. The runtime—not this package—generates and
collision-checks the local 3-D neighbors, after which the MLE reranks the refined
set. PF, MLE, and future estimators therefore share one physical workspace.

## Local source parameterization

The complete fitted surface can contain many patch/isotope coordinates. Directly
forming a dense Fisher matrix for all of them is unnecessarily expensive, while
keeping only nonzero MLE patches would make a sparse fit permanently blind to
unseen regions. The planner therefore builds a bounded local basis:

- the strongest fitted patch/isotope strengths get individual scaled modes;
- every remaining coordinate is retained in residual exploration modes grouped,
  in order of available capacity, by object, surface kind, or isotope;
- residual perturbations are distributed by physical patch area.

Consequently every patch/isotope coordinate contributes to at least one planning
mode, including coordinates whose current MLE is zero. The basis labels and scales
are written into every planning artifact.

## Expected Fisher criterion

Let `z` denote the reduced source modes and `eta` the fitted background/scatter
nuisance modes. For one candidate pose and shield pair, the shared line-resolved
response gives the expected spectrum `mu` and local Jacobian `J`. Its expected
Poisson Fisher information is

```text
F = J^T diag(1 / max(mu, mu_floor)) J.
```

Historical measurements are rebuilt with the same shared kernel. With stabilizing
Laplace precision `lambda I`, the current precision is

```text
P0 = lambda I + sum_historical F_m.
```

For any precision `P`, the source-only `D_s` value after marginalizing nuisance
parameters is

```text
psi(P) = logdet(P) - logdet(P_eta_eta).
```

This equals the log determinant of the source Schur-complement precision. For a
candidate program `A`, the reported information gain is

```text
IG(A) = 0.5 * [psi(P0 + sum_(a in A) F_a) - psi(P0)].
```

All views at one point can share an unknown multiplicative station-rate offset. For
each candidate row, the derivative with respect to its future log-rate is `mu`; the
planner augments the Fisher block with

```text
F_zs = sum_bins J
F_ss = sum_bins mu.
```

One rate parameter is shared by the entire candidate shield program and is included
in the nuisance Schur complement. This stops common absolute-count changes from
being mistaken for directional information.

Shield pairs at each pose are optimized as a complete ordered station program with a
bounded beam search, without repeating a pair. With the checked-in two-view RA-L
program and beam width 64, all possible first views survive to the second expansion.
Pair-specific vertical/support ambiguity utility participates in this search instead
of being added only after the shield angles have already been chosen. The final
station score is

```text
score = IG + mean_selected_pair_ambiguity_utility
           - motion_cost_weight * runtime_travel_cost
           - rotation_cost_weight * total_Fe_Pb_rotation_radians.
```

Both cost terms are disabled by default. Their weights have units of nats per unit
of the corresponding cost and must be calibrated for the acquisition platform.

## Vertical and support-ambiguity objectives

Local determinant gain can be large while two scientifically important hypotheses
remain almost indistinguishable. Every action therefore reports, and the planner can
weight, direct floor-versus-ceiling spectral separation, response-correlation
reduction, z-direction Fisher information, separation of recent alternative MLE
supports, source-relative elevation diversity, and early geometry exploration.

These terms use the complete surface dictionary or recent MLE fits, never PF
particles or PF candidates. They remain separate in the planning artifact so a large
aggregate score cannot hide vertical ambiguity. Candidate-density convergence is
tested across nested Sobol prefixes; one finite candidate set is not presented as an
exact continuous optimum.

This is a local design criterion, not a global Bayesian guarantee. It is most useful
after a stable station-complete fit. The residual modes, explicit ambiguity terms,
positive precision floor, and nuisance Schur complement make it safer than simply
ranking expected counts, but the result still depends on the current MLE and response
calibration.

## Runtime candidate JSON

The standalone CLI accepts this strict shell:

```json
{
  "candidate_poses_xyz": [[0.5, 0.5, 1.0], [1.5, 0.5, 1.0]],
  "travel_costs": [0.0, 1.25],
  "allowed_pair_ids": [0, 1, 8, 9],
  "current_pair_id": 0
}
```

Only `candidate_poses_xyz` is required. If `current_pair_id` is omitted, the last
causal measurement supplies it. The planner never accepts source truth, PF
particles, or PF candidates.

`plan-next` verifies that the saved MLE belongs to the runtime run and resolved MLE
configuration when those identities are present. Its covered step IDs must be an
exact prefix of the available log, and only that prefix contributes Fisher
information. Online planning is stricter still: it is available only when the
latest durable record closes a station and the latest fit covers all received
records.

Configuration defaults are in
[`configs/mle/default_planning.json`](../../configs/mle/default_planning.json).
