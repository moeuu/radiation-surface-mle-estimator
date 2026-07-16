"""Configuration objects for standalone count and spectral surface MLE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MLEConfig:
    """Configure surface patches, response construction, fitting, and reporting."""

    mode: Literal["count", "spectral"] = "count"
    isotope_names: tuple[str, ...] = ("Cs-137", "Co-60", "Eu-154")
    patch_spacing_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quadrature_order: int = 1
    obstacle_height_m: float = 2.0
    l1_weight: float = 0.0
    tv_weight: float = 0.0
    isotope_group_weight: float = 0.0
    nuisance_l1_weight: float = 0.0
    nuisance_l2_weight: float = 0.0
    fit_background_nuisance: bool = True
    fit_scatter_nuisance: bool = True
    max_iterations: int = 4000
    tolerance: float = 1.0e-6
    objective_tolerance: float = 1.0e-7
    check_interval: int = 20
    step_safety: float = 0.95
    over_relaxation: float = 1.0
    min_mean: float = 1.0e-12
    response_chunk_size: int = 262144
    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_dtype: Literal["float32", "float64"] = "float64"
    continuum_to_peak: float = 2.0
    backscatter_fraction: float = 0.03
    support_threshold_fraction: float = 1.0e-3
    debias_refit: bool = True
    coarse_to_fine_levels: int = 0
    refinement_fraction: float = 0.1
    response_correlation_threshold: float = 0.995
    cluster_threshold_fraction: float = 0.1
    cluster_min_strength_cps_1m: float = 0.0
    held_out_fraction: float = 0.0
    held_out_grouping: Literal[
        "station_id",
        "same_xy_height",
        "shield_program_block",
        "row",
    ] = "station_id"
    held_out_xy_tolerance_m: float = 1.0e-6
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Validate values that affect physical or optimization semantics."""
        if self.mode not in {"count", "spectral"}:
            raise ValueError("mode must be 'count' or 'spectral'.")
        names = tuple(str(value).strip() for value in self.isotope_names)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("isotope_names must contain unique non-empty names.")
        spacing = tuple(float(value) for value in self.patch_spacing_m)
        if len(spacing) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError(
                "patch_spacing_m must contain three finite positive values."
            )
        if int(self.quadrature_order) not in {1, 4}:
            raise ValueError("quadrature_order must be 1 or 4.")
        nonnegative = {
            "obstacle_height_m": self.obstacle_height_m,
            "l1_weight": self.l1_weight,
            "tv_weight": self.tv_weight,
            "isotope_group_weight": self.isotope_group_weight,
            "nuisance_l1_weight": self.nuisance_l1_weight,
            "nuisance_l2_weight": self.nuisance_l2_weight,
            "tolerance": self.tolerance,
            "objective_tolerance": self.objective_tolerance,
            "continuum_to_peak": self.continuum_to_peak,
            "backscatter_fraction": self.backscatter_fraction,
            "support_threshold_fraction": self.support_threshold_fraction,
            "cluster_threshold_fraction": self.cluster_threshold_fraction,
            "cluster_min_strength_cps_1m": self.cluster_min_strength_cps_1m,
            "held_out_fraction": self.held_out_fraction,
            "held_out_xy_tolerance_m": self.held_out_xy_tolerance_m,
        }
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative.values()):
            raise ValueError(
                "MLE weights, tolerances, and fractions must be finite and non-negative."
            )
        if int(self.max_iterations) < 1 or int(self.check_interval) < 1:
            raise ValueError("Iteration counts must be positive.")
        if int(self.response_chunk_size) < 1:
            raise ValueError("response_chunk_size must be positive.")
        if not 0.0 < float(self.step_safety) < 1.0:
            raise ValueError("step_safety must lie strictly between zero and one.")
        if not 0.0 <= float(self.over_relaxation) <= 1.0:
            raise ValueError("over_relaxation must lie between zero and one.")
        if not np.isfinite(self.min_mean) or self.min_mean <= 0.0:
            raise ValueError("min_mean must be finite and positive.")
        if not 0.0 <= float(self.refinement_fraction) <= 1.0:
            raise ValueError("refinement_fraction must lie between zero and one.")
        if not 0.0 <= float(self.response_correlation_threshold) <= 1.0:
            raise ValueError(
                "response_correlation_threshold must lie between zero and one."
            )
        if not 0.0 <= float(self.held_out_fraction) < 1.0:
            raise ValueError("held_out_fraction must lie in [0, 1).")
        if self.held_out_grouping not in {
            "station_id",
            "same_xy_height",
            "shield_program_block",
            "row",
        }:
            raise ValueError(
                "held_out_grouping must be station_id, same_xy_height, "
                "shield_program_block, or row."
            )
        if (
            not np.isfinite(self.held_out_xy_tolerance_m)
            or float(self.held_out_xy_tolerance_m) <= 0.0
        ):
            raise ValueError("held_out_xy_tolerance_m must be finite and positive.")
        if int(self.coarse_to_fine_levels) < 0:
            raise ValueError("coarse_to_fine_levels must be non-negative.")
        if self.gpu_dtype not in {"float32", "float64"}:
            raise ValueError("gpu_dtype must be float32 or float64.")
        object.__setattr__(self, "isotope_names", names)
        object.__setattr__(self, "patch_spacing_m", spacing)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MLEConfig":
        """Build a configuration from a JSON-like mapping."""
        values = dict(payload)
        if "isotopes" in values and "isotope_names" not in values:
            values["isotope_names"] = values.pop("isotopes")
        if "patch_spacing_m" in values:
            raw_spacing = values["patch_spacing_m"]
            if isinstance(raw_spacing, (int, float)):
                raw_spacing = (float(raw_spacing),) * 3
            values["patch_spacing_m"] = tuple(float(value) for value in raw_spacing)
        if "isotope_names" in values:
            values["isotope_names"] = tuple(
                str(value) for value in values["isotope_names"]
            )
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "MLEConfig":
        """Load a JSON configuration file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("MLE configuration root must be a JSON object.")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write deterministic JSON configuration output."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def build_default_config(
    *,
    mode: Literal["count", "spectral"] = "count",
    isotopes: Sequence[str] | None = None,
) -> MLEConfig:
    """Return a default configuration for programmatic callers."""
    names = (
        MLEConfig().isotope_names
        if isotopes is None
        else tuple(str(value) for value in isotopes)
    )
    return MLEConfig(mode=mode, isotope_names=names)
