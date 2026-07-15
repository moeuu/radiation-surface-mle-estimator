"""Generate RA-L ablation configurations without mixing baseline logic into DSS-PP."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from measurement.model import EnvironmentConfig
from measurement.obstacles import build_obstacle_grid
from measurement.source_surfaces import generate_surface_sources
from runtime_defaults import (
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_NO_ROTATION_OVERHEAD_S,
    DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = (
    ROOT / "configs" / "geant4" / "variance_reduction_external_no_isaac_32threads.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ral_ablation"
DEFAULT_ISOTOPES = ("Cs-137", "Co-60", "Eu-154")


@dataclass(frozen=True)
class AblationCase:
    """Describe a fixed-source-cardinality RA-L ablation case."""

    name: str
    description: str
    isotopes: tuple[str, ...]
    source_count: int
    isotope_counts: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class AblationVariant:
    """Describe one module-ablation variant."""

    name: str
    description: str
    overrides: Mapping[str, Any]
    cli_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class AblationPlanEntry:
    """Store one executable ablation trial."""

    case: str
    variant: str
    seed: int
    config_path: Path
    source_path: Path
    command: tuple[str, ...]


DEFAULT_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase(
        name="mix9_multi_isotope_cardinality",
        description=(
            "Main RA-L task: 4 Cs-137, 3 Co-60, and 2 Eu-154 sources with "
            "same-isotope ambiguity inside a multi-isotope STE problem."
        ),
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        source_count=9,
        isotope_counts=(("Cs-137", 4), ("Co-60", 3), ("Eu-154", 2)),
    ),
)

DEFAULT_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        name="proposed",
        description="Full proposed temporal shield program and DSS-PP.",
        overrides={},
    ),
    AblationVariant(
        name="no_shield",
        description=(
            "Remove shield attenuation while taking one unshielded spectrum per "
            "measurement station."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="fixed_shield",
        description="Repeat a fixed Fe/Pb posture pair for every shield view.",
        overrides={
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
        },
    ),
    AblationVariant(
        name="round_robin_shield",
        description="Cycle Fe/Pb posture pairs without posterior-dependent selection.",
        overrides={
            "baseline_shield_policy": {
                "name": "round_robin",
                "start_pair_id": 0,
                "advance_by_pose": True,
            },
            "strict_planned_shield_program": True,
            "dss_pp": {
                "adaptive_program_length_enable": False,
            },
        },
    ),
    AblationVariant(
        name="one_step_path",
        description="Use the existing greedy one-step pose planner instead of DSS-PP.",
        overrides={
            "path_planner": "one_step",
            "strict_planned_shield_program": True,
        },
    ),
    AblationVariant(
        name="eig_only_path",
        description=(
            "Keep active planning and shield programs but remove explicit "
            "signature, obstacle-shadow, correlation-reduction, and elevation "
            "terms from DSS-PP."
        ),
        overrides={
            "dss_pp": {
                "signature_weight": 0.0,
                "temporal_separation_weight": 0.0,
                "temporal_decorrelation_weight": 0.0,
                "temporal_logdet_weight": 0.0,
                "temporal_cover_weight": 0.0,
                "correlation_reduction_weight": 0.0,
                "same_isotope_direct_separation_guard": False,
                "environment_signature_weight": 0.0,
                "occlusion_boundary_weight": 0.0,
                "vertical_environment_signature_weight": 0.0,
                "elevation_signature_weight": 0.0,
                "elevation_condition_weight": 0.0,
                "station_condition_weight": 0.0,
                "high_surface_pair_boost": 1.0,
                "high_surface_cross_stratum_boost": 1.0,
            },
        },
    ),
    AblationVariant(
        name="passive_serpentine_path",
        description="Use a deterministic coverage path from the baseline package.",
        overrides={
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
        },
    ),
    AblationVariant(
        name="baseline_passive_no_shield",
        description=(
            "Ordinary mobile-PF baseline: no shield attenuation and passive "
            "serpentine coverage path with one unshielded spectrum per "
            "measurement station."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="baseline_passive_equal_time_no_shield",
        description=(
            "Passive no-shield baseline with the same per-station physical "
            "live-time budget as the proposed shield program."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="baseline_passive_fixed_shield",
        description=(
            "Passive-path baseline with a fixed Fe/Pb shield posture and no "
            "posterior-dependent shield selection, using the same spectra "
            "budget as the proposed method."
        ),
        overrides={
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
        },
    ),
    AblationVariant(
        name="baseline_passive_no_shield_single_view",
        description=(
            "Ordinary mobile-PF baseline: no shield and one spectrum per "
            "measurement station."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="baseline_passive_fixed_shield_single_view",
        description=(
            "Passive-path fixed-shield baseline with one spectrum per "
            "measurement station."
        ),
        overrides={
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
    ),
    AblationVariant(
        name="baseline_onestep_no_shield",
        description=(
            "Greedy one-step planner baseline without shield attenuation and "
            "with one unshielded spectrum per measurement station."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "path_planner": "one_step",
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="baseline_onestep_fixed_shield",
        description=(
            "Greedy one-step planner baseline with a fixed Fe/Pb shield posture "
            "and the same spectra budget as the proposed method."
        ),
        overrides={
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "path_planner": "one_step",
        },
    ),
    AblationVariant(
        name="baseline_onestep_no_shield_single_view",
        description=(
            "Greedy one-step planner baseline without shield attenuation and "
            "with one spectrum per measurement station."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "path_planner": "one_step",
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="baseline_onestep_fixed_shield_single_view",
        description=(
            "Greedy one-step fixed-shield baseline with one spectrum per "
            "measurement station."
        ),
        overrides={
            "orientation_k": 1,
            "min_rotations_per_pose": 1,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "path_planner": "one_step",
            "dss_pp": {
                "program_length": 1,
                "residual_program_length": 1,
            },
        },
    ),
    AblationVariant(
        name="no_residual_birth",
        description="Disable residual-coded source birth while preserving PF capacity.",
        overrides={
            "birth_max_per_update": 0,
            "split_prob": 0.0,
            "split_residual_guided": False,
            "split_residual_always_try": False,
            "birth_use_shield_coded_residual": False,
            "birth_residual_always_try": False,
            "birth_residual_expand_structural_particles": False,
            "birth_residual_force_proposal_on_gate": False,
            "birth_residual_suppress_death": False,
            "birth_global_rescue_enable": False,
            "residual_decomposition_enable": False,
            "peak_suppression_enable": False,
            "report_mle_rescue_enable": False,
            "report_mle_rescue_max_residual_candidates": 0,
            "runtime_report_rescue_enable": False,
            "runtime_report_rescue_particle_fraction": 0.0,
            "runtime_report_rescue_weight": 0.0,
            "runtime_report_rescue_quarantine_enable": False,
            "runtime_report_rescue_quarantine_weight": 0.0,
            "runtime_report_rescue_candidate_weight": 0.0,
            "runtime_report_rescue_memory_enable": False,
            "runtime_report_rescue_memory_max_sources": 0,
        },
    ),
    AblationVariant(
        name="no_verification",
        description=(
            "Disable tentative-source verification and refit-after-remove "
            "report pruning for logged estimator-side replay ablations."
        ),
        overrides={
            "pseudo_source_verification_enable": False,
            "source_prune_refit_after_remove": False,
            "report_strength_refit_preserve_cardinality": True,
        },
    ),
    AblationVariant(
        name="no_obstacle_signature",
        description="Keep obstacle attenuation in PF/Geant4 but remove obstacle terms from DSS-PP utility.",
        overrides={
            "dss_pp": {
                "environment_signature_weight": 0.0,
                "occlusion_boundary_weight": 0.0,
                "vertical_environment_signature_weight": 0.0,
            },
        },
    ),
    AblationVariant(
        name="no_pf_obstacle_attenuation",
        description=(
            "Keep Geant4 obstacles and obstacle-aware planning but remove "
            "known-obstacle attenuation from the PF observation kernel."
        ),
        overrides={"pf_obstacle_attenuation": False},
    ),
    AblationVariant(
        name="volume_source_prior",
        description=(
            "Use legacy full-volume PF source-position support instead of "
            "known room, floor, ceiling, and obstacle surfaces."
        ),
        overrides={"source_surface_prior": False},
    ),
)


def _parallel_runtime_overrides(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-fidelity-changing compute settings for generated trials."""
    logical_workers = int(
        base_config.get(
            "python_worker_count",
            base_config.get("cpu_worker_count", 32),
        )
    )
    workers = max(1, logical_workers)
    dss_runtime = base_config.get("dss_pp", {})
    if not isinstance(dss_runtime, Mapping):
        dss_runtime = {}
    return {
        "thread_count": max(1, int(base_config.get("thread_count", workers))),
        "python_worker_count": workers,
        "pose_selection_workers": max(
            1,
            int(base_config.get("pose_selection_workers", workers)),
        ),
        "ig_workers": max(1, int(base_config.get("ig_workers", workers))),
        "parallel_isotope_updates": bool(
            base_config.get("parallel_isotope_updates", True)
        ),
        "parallel_isotope_workers": max(
            1,
            int(base_config.get("parallel_isotope_workers", workers)),
        ),
        "dss_pp": {
            "program_eval_workers": max(
                1,
                int(dss_runtime.get("program_eval_workers", workers)),
            ),
            "candidate_preselect_enable": bool(
                dss_runtime.get("candidate_preselect_enable", True)
            ),
        },
    }


def _deep_update(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive dictionary merge of base and overrides."""
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a path."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON object to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_base_config_path(value: object, *, base_config_path: Path) -> str | None:
    """Resolve a config-relative path so generated configs remain relocatable."""
    if not isinstance(value, str) or value.strip() == "":
        return None
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.as_posix()
    return (base_config_path.parent / raw_path).resolve().as_posix()


def _case_isotope_sequence(case: AblationCase) -> tuple[str, ...]:
    """Return the exact isotope sequence used for source generation."""
    if case.isotope_counts is None:
        return tuple(str(isotope) for isotope in case.isotopes)
    expanded: list[str] = []
    for isotope, count in case.isotope_counts:
        source_count = int(count)
        if source_count < 0:
            raise ValueError(f"Negative isotope count for {isotope}: {source_count}")
        expanded.extend([str(isotope)] * source_count)
    if len(expanded) != int(case.source_count):
        raise ValueError(
            f"Case {case.name} isotope_counts expand to {len(expanded)} sources, "
            f"but source_count is {case.source_count}."
        )
    return tuple(expanded)


def _case_isotope_count_metadata(case: AblationCase) -> dict[str, int]:
    """Return isotope-count metadata for generated source layouts."""
    counts: dict[str, int] = {}
    if case.isotope_counts is not None:
        for isotope, count in case.isotope_counts:
            counts[str(isotope)] = int(count)
        return counts
    for idx in range(max(1, int(case.source_count))):
        isotope = str(case.isotopes[idx % len(case.isotopes)])
        counts[isotope] = counts.get(isotope, 0) + 1
    return counts


def _free_cell_measurement_points(
    obstacle_grid: Any,
    env: EnvironmentConfig,
) -> np.ndarray:
    """Return detector-height reference points over reachable free grid cells."""
    points: list[tuple[float, float, float]] = []
    origin_x, origin_y = obstacle_grid.origin
    cell_size = float(obstacle_grid.cell_size)
    for ix in range(int(obstacle_grid.grid_shape[0])):
        for iy in range(int(obstacle_grid.grid_shape[1])):
            if not obstacle_grid.is_cell_free((ix, iy)):
                continue
            x = float(origin_x) + (float(ix) + 0.5) * cell_size
            y = float(origin_y) + (float(iy) + 0.5) * cell_size
            if 0.0 <= x <= float(env.size_x) and 0.0 <= y <= float(env.size_y):
                points.append((x, y, float(env.detector_position[2])))
    if not points:
        points.append(
            (
                float(env.detector_position[0]),
                float(env.detector_position[1]),
                float(env.detector_position[2]),
            )
        )
    return np.asarray(points, dtype=float)


def _source_generation_options(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return source-placement options shared with standard random generation."""
    preferred_raw = base_config.get("random_source_preferred_max_z_m", 5.0)
    preferred_max_z_m = None if preferred_raw is None else float(preferred_raw)
    return {
        "visibility_filter": bool(base_config.get("random_source_visibility_filter", True)),
        "visibility_min_fraction": float(
            base_config.get("random_source_min_visible_fraction", 0.1)
        ),
        "visibility_clear_path_max_m": float(
            base_config.get("random_source_clear_path_max_m", 0.01)
        ),
        "visibility_batch_size": int(
            base_config.get("random_source_visibility_batch_size", 256)
        ),
        "visibility_max_attempts_per_source": int(
            base_config.get("random_source_visibility_max_attempts_per_source", 4096)
        ),
        "max_ceiling_sources": int(
            base_config.get("random_source_max_ceiling_sources", 1)
        ),
        "preferred_max_z_m": preferred_max_z_m,
        "same_isotope_min_distance_m": float(
            base_config.get("random_source_same_isotope_min_distance_m", 2.0)
        ),
    }


def _case_source_layout(
    case: AblationCase,
    *,
    obstacle_seed: int,
    source_seed: int,
    intensity_cps_1m: float | Sequence[float],
    source_generation_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a surface-constrained source layout for one case and seed."""
    env = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=10.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=env.size_x,
        size_y=env.size_y,
        cell_size=1.0,
        blocked_fraction=0.4,
        rng_seed=obstacle_seed,
        keep_free_points=[(env.detector_position[0], env.detector_position[1])],
        passage_width_m=1.0,
    )
    rng = np.random.default_rng(source_seed)
    options = dict(source_generation_options or {})
    visibility_points = None
    if bool(options.get("visibility_filter", False)):
        visibility_points = _free_cell_measurement_points(grid, env)
    isotope_sequence = _case_isotope_sequence(case)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=isotope_sequence,
        intensity_cps_1m=intensity_cps_1m,
        rng=rng,
        count=case.source_count,
        obstacle_height_m=2.0,
        visibility_measurement_points=visibility_points,
        visibility_min_fraction=float(options.get("visibility_min_fraction", 0.0)),
        visibility_clear_path_max_m=float(
            options.get("visibility_clear_path_max_m", 0.01)
        ),
        visibility_batch_size=int(options.get("visibility_batch_size", 256)),
        visibility_max_attempts_per_source=int(
            options.get("visibility_max_attempts_per_source", 4096)
        ),
        max_ceiling_sources=options.get("max_ceiling_sources", 1),
        preferred_max_z_m=options.get("preferred_max_z_m", 5.0),
        same_isotope_min_distance_m=float(
            options.get("same_isotope_min_distance_m", 2.0)
        ),
    )
    return {
        "name": f"ral_ablation_{case.name}_seed_{source_seed}",
        "metadata": {
            "case": case.name,
            "description": case.description,
            "isotope_counts": _case_isotope_count_metadata(case),
            "source_seed": int(source_seed),
            "obstacle_seed": int(obstacle_seed),
            "sampling": "surface-constrained room/obstacle source placement",
            "visibility_filter": bool(options.get("visibility_filter", False)),
            "visibility_min_fraction": float(
                options.get("visibility_min_fraction", 0.0)
            ),
            "same_isotope_min_distance_m": float(
                options.get("same_isotope_min_distance_m", 2.0)
            ),
            "intensity_model": "intensity_cps_1m is expected net detector cps at 1 m",
            "intensity_sampling": _intensity_sampling_metadata(intensity_cps_1m),
        },
        "sources": [
            {
                "isotope": source.isotope,
                "position": [round(float(v), 6) for v in source.position],
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
            for source in sources
        ],
    }


def _intensity_sampling_metadata(
    intensity_cps_1m: float | Sequence[float],
) -> dict[str, float | str]:
    """Return metadata describing source-strength sampling for a case."""
    if isinstance(intensity_cps_1m, Sequence) and not isinstance(
        intensity_cps_1m,
        (str, bytes),
    ):
        if len(intensity_cps_1m) != 2:
            raise ValueError("intensity range must contain exactly two values.")
        lo = float(intensity_cps_1m[0])
        hi = float(intensity_cps_1m[1])
        return {"mode": "uniform", "min_cps_1m": lo, "max_cps_1m": hi}
    return {"mode": "fixed", "cps_1m": float(intensity_cps_1m)}


def _variant_config(
    base_config: Mapping[str, Any],
    *,
    base_config_path: Path,
    case: AblationCase,
    variant: AblationVariant,
    seed: int,
    output_tag: str,
) -> dict[str, Any]:
    """Return the runtime config for one ablation variant."""
    config = _deep_update(base_config, _parallel_runtime_overrides(base_config))
    config = _deep_update(config, variant.overrides)
    config["random_seed_base"] = int(seed)
    # Keep the browser progress page stable across ablation runs. The final
    # result files still use output_tag, so only the live progress view is shared.
    config["cui_split_view_dir"] = DEFAULT_CUI_SPLIT_VIEW_DIR
    for path_key in ("usd_path", "random_environment_base_usd_path"):
        resolved_path = _resolve_base_config_path(
            config.get(path_key),
            base_config_path=base_config_path,
        )
        if resolved_path is not None:
            config[path_key] = resolved_path
    config.setdefault("metadata", {})
    if isinstance(config["metadata"], dict):
        config["metadata"].update(
            {
                "ral_ablation_case": case.name,
                "ral_ablation_variant": variant.name,
                "ral_ablation_seed": int(seed),
            }
        )
    return config


def build_ablation_plan(
    *,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = (2026050901, 2026050902, 2026050903),
    cases: Sequence[AblationCase] = DEFAULT_ABLATION_CASES,
    variants: Sequence[AblationVariant] = DEFAULT_ABLATION_VARIANTS,
    intensity_cps_1m: float | Sequence[float] = DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
) -> list[AblationPlanEntry]:
    """Build and write config/source files for RA-L ablation trials."""
    base_config_path = Path(base_config_path).expanduser().resolve()
    base_config = _load_json(base_config_path)
    source_options = _source_generation_options(base_config)
    entries: list[AblationPlanEntry] = []
    config_dir = Path(output_dir) / "configs"
    source_dir = Path(output_dir) / "sources"
    for case in cases:
        for seed in seeds:
            source_seed = int(seed) + 17
            source_payload = _case_source_layout(
                case,
                obstacle_seed=int(seed),
                source_seed=source_seed,
                intensity_cps_1m=intensity_cps_1m,
                source_generation_options=source_options,
            )
            source_path = source_dir / f"{case.name}_seed_{seed}.json"
            _write_json(source_path, source_payload)
            for variant in variants:
                tag = f"{case.name}_{variant.name}_seed_{seed}"
                config = _variant_config(
                    base_config,
                    base_config_path=base_config_path,
                    case=case,
                    variant=variant,
                    seed=int(seed),
                    output_tag=tag,
                )
                config_path = config_dir / f"{tag}.json"
                _write_json(config_path, config)
                command = _trial_command(
                    config_path=config_path,
                    source_path=source_path,
                    obstacle_seed=int(seed),
                    output_tag=tag,
                    extra_args=variant.cli_args,
                )
                entries.append(
                    AblationPlanEntry(
                        case=case.name,
                        variant=variant.name,
                        seed=int(seed),
                        config_path=config_path,
                        source_path=source_path,
                        command=command,
                    )
                )
    return entries


def _trial_command(
    *,
    config_path: Path,
    source_path: Path,
    obstacle_seed: int,
    output_tag: str,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the standard full-simulation command for one ablation trial."""
    return (
        "uv",
        "run",
        "python",
        "main.py",
        "--full-simulation",
        "--sim-config",
        config_path.as_posix(),
        "--environment-mode",
        "random",
        "--obstacle-seed",
        str(int(obstacle_seed)),
        "--source-config",
        source_path.as_posix(),
        "--birth",
        "--measurement-time-s",
        f"{DEFAULT_MEASUREMENT_TIME_S:g}",
        "--output-tag",
        output_tag,
        *tuple(extra_args),
    )


def write_ablation_plan(
    entries: Sequence[AblationPlanEntry],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write a CSV manifest and shell command file for ablation entries."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    script_path = out / "run_all.sh"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "variant",
                "seed",
                "config_path",
                "source_path",
                "command",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "case": entry.case,
                    "variant": entry.variant,
                    "seed": entry.seed,
                    "config_path": entry.config_path.as_posix(),
                    "source_path": entry.source_path.as_posix(),
                    "command": " ".join(entry.command),
                }
            )
    with script_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for entry in entries:
            handle.write(" ".join(entry.command))
            handle.write("\n")
    script_path.chmod(0o755)
    return manifest_path, script_path
