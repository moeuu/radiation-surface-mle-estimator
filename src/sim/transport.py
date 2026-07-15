"""Shared pre-spectrum transport objects and helpers for simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from measurement.model import PointSource, inverse_square_scale
from sim.isaacsim_app.materials import (
    composition_mass_attenuation,
    composition_mass_attenuation_at_energy,
    normalize_composition_by_mass,
    resolve_material_preset,
)

DEFAULT_MATERIAL_MU_CM_INV: dict[str, dict[str, float]] = {
    "air": {"Cs-137": 0.0, "Co-60": 0.0, "Eu-154": 0.0},
    "concrete": {"Cs-137": 0.17, "Co-60": 0.11, "Eu-154": 0.18},
    "cebr3": {"Cs-137": 0.27, "Co-60": 0.19, "Eu-154": 0.21},
    "aluminum": {"Cs-137": 0.20, "Co-60": 0.14, "Eu-154": 0.21},
}


@dataclass(frozen=True)
class TransportMaterial:
    """Describe a material used in the shared pre-spectrum transport phase."""

    name: str
    mu_by_isotope: dict[str, float] = field(default_factory=dict)
    density_g_cm3: float | None = None
    mass_att_by_isotope_cm2_g: dict[str, float] = field(default_factory=dict)
    preset_name: str | None = None
    composition_by_mass: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportSegment:
    """Describe a material crossing along the source-to-detector path."""

    material: TransportMaterial
    path_length_cm: float
    is_obstacle: bool = False


@dataclass(frozen=True)
class TransportLineResult:
    """Store pre-spectrum transport statistics for one gamma line."""

    energy_keV: float
    line_intensity: float
    emission_counts: float
    stage_transmission: float
    shield_transmission: float
    total_transmission: float
    primary_counts: float
    scatter_counts: float


@dataclass(frozen=True)
class SourceTransportResult:
    """Collect shared pre-spectrum transport outputs for one source."""

    source: PointSource
    detector_position_xyz: tuple[float, float, float]
    dwell_time_s: float
    geometric_scale: float
    base_source_counts: float
    stage_segments: tuple[TransportSegment, ...]
    fe_segment: TransportSegment
    pb_segment: TransportSegment
    lines: tuple[TransportLineResult, ...]

    @property
    def total_stage_path_cm(self) -> float:
        """Return the total static-material path length in centimeters."""
        return float(sum(segment.path_length_cm for segment in self.stage_segments))

    @property
    def total_obstacle_path_cm(self) -> float:
        """Return the total obstacle-only path length in centimeters."""
        return float(sum(segment.path_length_cm for segment in self.stage_segments if segment.is_obstacle))

    @property
    def total_fe_path_cm(self) -> float:
        """Return the Fe-shield path length in centimeters."""
        return float(self.fe_segment.path_length_cm)

    @property
    def total_pb_path_cm(self) -> float:
        """Return the Pb-shield path length in centimeters."""
        return float(self.pb_segment.path_length_cm)


def coerce_transport_material(material: Any) -> TransportMaterial:
    """Convert a stage/exported material-like object into a shared material payload."""
    composition = normalize_composition_by_mass(getattr(material, "composition_by_mass", {}))
    return TransportMaterial(
        name=str(getattr(material, "name")),
        mu_by_isotope={
            str(key): float(value)
            for key, value in getattr(material, "mu_by_isotope", {}).items()
        },
        density_g_cm3=getattr(material, "density_g_cm3", None),
        mass_att_by_isotope_cm2_g={
            str(key): float(value)
            for key, value in getattr(material, "mass_att_by_isotope_cm2_g", {}).items()
        },
        preset_name=getattr(material, "preset_name", None),
        composition_by_mass=composition,
    )


def make_transport_segment(material: Any, path_length_cm: float, *, is_obstacle: bool = False) -> TransportSegment:
    """Build a shared transport segment from a material-like input."""
    return TransportSegment(
        material=coerce_transport_material(material),
        path_length_cm=float(path_length_cm),
        is_obstacle=bool(is_obstacle),
    )


def material_transmission(
    material: TransportMaterial,
    isotope: str,
    line_energy_keV: float,
    path_length_cm: float,
) -> float:
    """Return the Beer-Lambert transmission through one material segment."""
    path_length_cm = max(0.0, float(path_length_cm))
    if path_length_cm <= 0.0:
        return 1.0
    mu = material.mu_by_isotope.get(isotope)
    density_g_cm3 = material.density_g_cm3
    mass_att = material.mass_att_by_isotope_cm2_g.get(isotope)
    energy_mass_att = None
    if mass_att is None and material.composition_by_mass:
        energy_mass_att = composition_mass_attenuation_at_energy(
            material.composition_by_mass,
            line_energy_keV,
        )
        if energy_mass_att is None:
            mass_att = composition_mass_attenuation(material.composition_by_mass, isotope)
    preset = resolve_material_preset(material.preset_name or material.name)
    if density_g_cm3 is None and preset is not None:
        density_g_cm3 = preset.density_g_cm3
    preset_energy_mass_att = None
    if mass_att is None and preset is not None:
        mass_att = preset.mass_att_by_isotope_cm2_g.get(isotope)
    if mass_att is None and preset is not None and preset.composition_by_mass:
        preset_energy_mass_att = composition_mass_attenuation_at_energy(
            preset.composition_by_mass,
            line_energy_keV,
        )
        if preset_energy_mass_att is None:
            mass_att = composition_mass_attenuation(preset.composition_by_mass, isotope)
    if mu is None and density_g_cm3 is not None and energy_mass_att is not None:
        mu = float(density_g_cm3) * float(energy_mass_att)
    if mu is None and density_g_cm3 is not None and preset_energy_mass_att is not None:
        mu = float(density_g_cm3) * float(preset_energy_mass_att)
    if mu is None and density_g_cm3 is not None and mass_att is not None:
        mu = float(density_g_cm3) * float(mass_att)
    if mu is None:
        table = DEFAULT_MATERIAL_MU_CM_INV.get(material.name.lower(), {})
        mu = float(table.get(isotope, 0.0))
    return float(np.exp(-float(mu) * path_length_cm))


def scatter_scale(*, path_length_cm: float, transmission: float, scatter_gain: float) -> float:
    """Return a coarse scatter multiplier driven by blocking and material depth."""
    blocked_fraction = float(np.clip(1.0 - float(transmission), 0.0, 1.0))
    path_factor = float(np.clip(float(path_length_cm) / 80.0, 0.0, 2.0))
    return float(float(scatter_gain) * blocked_fraction * (0.5 + path_factor))


def build_source_transport_result(
    *,
    source: PointSource,
    detector_position_xyz: Sequence[float],
    dwell_time_s: float,
    stage_segments: Iterable[TransportSegment],
    fe_segment: TransportSegment,
    pb_segment: TransportSegment,
    nuclide_lines: Iterable[tuple[float, float]],
    scatter_gain: float,
) -> SourceTransportResult:
    """Build a shared pre-spectrum transport result for one source."""
    detector_position_xyz = tuple(float(value) for value in detector_position_xyz)
    stage_segments = tuple(stage_segments)
    detector_position = np.asarray(detector_position_xyz, dtype=float)
    geometric_scale = float(inverse_square_scale(detector_position, source))
    base_source_counts = float(dwell_time_s) * float(source.intensity_cps_1m) * geometric_scale
    total_path_cm = (
        sum(segment.path_length_cm for segment in stage_segments)
        + float(fe_segment.path_length_cm)
        + float(pb_segment.path_length_cm)
    )
    lines: list[TransportLineResult] = []
    for energy_keV, line_intensity in nuclide_lines:
        emission_counts = base_source_counts * float(line_intensity)
        stage_transmission = 1.0
        for segment in stage_segments:
            stage_transmission *= material_transmission(
                segment.material,
                source.isotope,
                float(energy_keV),
                segment.path_length_cm,
            )
        shield_transmission = material_transmission(
            fe_segment.material,
            source.isotope,
            float(energy_keV),
            fe_segment.path_length_cm,
        ) * material_transmission(
            pb_segment.material,
            source.isotope,
            float(energy_keV),
            pb_segment.path_length_cm,
        )
        total_transmission = float(stage_transmission * shield_transmission)
        primary_counts = float(emission_counts * total_transmission)
        scatter_counts = float(
            primary_counts
            * scatter_scale(
                path_length_cm=total_path_cm,
                transmission=total_transmission,
                scatter_gain=scatter_gain,
            )
        )
        lines.append(
            TransportLineResult(
                energy_keV=float(energy_keV),
                line_intensity=float(line_intensity),
                emission_counts=float(emission_counts),
                stage_transmission=float(stage_transmission),
                shield_transmission=float(shield_transmission),
                total_transmission=float(total_transmission),
                primary_counts=primary_counts,
                scatter_counts=scatter_counts,
            )
        )
    return SourceTransportResult(
        source=source,
        detector_position_xyz=detector_position_xyz,
        dwell_time_s=float(dwell_time_s),
        geometric_scale=geometric_scale,
        base_source_counts=base_source_counts,
        stage_segments=stage_segments,
        fe_segment=fe_segment,
        pb_segment=pb_segment,
        lines=tuple(lines),
    )
