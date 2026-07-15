"""Line-resolved spectral response tensors built from the local shared kernel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from spectrum.library import default_library
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    cebr3_efficiency,
    compton_continuum_shape,
    default_background_shape,
    default_resolution,
    detector_response_kernel_for_incident_gamma,
)


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without accepting booleans or lossy casts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _isotope_names(isotopes: Sequence[str]) -> tuple[str, ...]:
    """Return unique, non-empty isotope names in their requested order."""
    names = tuple(isotopes)
    if not names:
        raise ValueError("At least one isotope is required.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Isotopes must contain only non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Isotopes must not contain duplicates.")
    return names


def _orientation_indices(
    values: object,
    *,
    name: str,
    measurement_count: int,
    orientation_count: int,
) -> NDArray[np.int64]:
    """Return one in-range integer shield index per measurement."""
    raw = np.asarray(values)
    if raw.shape != (measurement_count,):
        raise ValueError(
            f"{name} must have shape ({measurement_count},), got {raw.shape}."
        )
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(f"{name} must contain integer indices.")
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= orientation_count):
        raise ValueError(
            f"{name} entries must lie in [0, {orientation_count - 1}]."
        )
    return np.ascontiguousarray(indices)


def _validated_observation_geometry(
    observations: object,
    kernel: ContinuousKernel,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return validated geometry, timing, shield pairs, and spectrum edges."""
    detector_positions = np.asarray(
        getattr(observations, "detector_positions_xyz"),
        dtype=np.float64,
    )
    if (
        detector_positions.ndim != 2
        or detector_positions.shape[1:] != (3,)
        or detector_positions.shape[0] == 0
    ):
        raise ValueError("detector_positions_xyz must have non-empty shape (M, 3).")
    if not np.all(np.isfinite(detector_positions)):
        raise ValueError("detector_positions_xyz must contain only finite values.")
    detector_positions = np.ascontiguousarray(detector_positions)
    measurement_count = int(detector_positions.shape[0])

    orientations = np.asarray(kernel.orientations, dtype=float)
    if (
        orientations.ndim != 2
        or orientations.shape[1:] != (3,)
        or orientations.shape[0] == 0
        or not np.all(np.isfinite(orientations))
    ):
        raise ValueError("kernel.orientations must have non-empty finite shape (R, 3).")
    orientation_count = int(orientations.shape[0])
    fe_indices = _orientation_indices(
        getattr(observations, "fe_indices"),
        name="fe_indices",
        measurement_count=measurement_count,
        orientation_count=orientation_count,
    )
    pb_indices = _orientation_indices(
        getattr(observations, "pb_indices"),
        name="pb_indices",
        measurement_count=measurement_count,
        orientation_count=orientation_count,
    )

    live_times = np.asarray(
        getattr(observations, "live_times_s"),
        dtype=np.float64,
    )
    if live_times.shape != (measurement_count,):
        raise ValueError("live_times_s must contain one entry per measurement.")
    if not np.all(np.isfinite(live_times)) or np.any(live_times <= 0.0):
        raise ValueError("live_times_s must contain finite positive values.")
    live_times = np.ascontiguousarray(live_times)

    edges = np.asarray(
        getattr(observations, "energy_bin_edges_keV"),
        dtype=np.float64,
    )
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("energy_bin_edges_keV must be a one-dimensional bin edge array.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_bin_edges_keV must be finite and strictly increasing.")
    edges = np.ascontiguousarray(edges)

    if hasattr(observations, "spectrum_counts"):
        spectrum = np.asarray(getattr(observations, "spectrum_counts"))
        expected_shape = (measurement_count, edges.size - 1)
        if spectrum.shape != expected_shape:
            raise ValueError(
                f"spectrum_counts must have shape {expected_shape}, got {spectrum.shape}."
            )
    return detector_positions, fe_indices, pb_indices, live_times, edges


@dataclass(frozen=True)
class SpectralResponseResult:
    """Store a line-resolved response tensor and its construction diagnostics."""

    response_per_integrated_strength: NDArray[np.float64]
    response_per_density: NDArray[np.float64]
    nuisance_response: NDArray[np.float64]
    nuisance_names: tuple[str, ...]
    line_energies_keV_by_isotope: dict[str, tuple[float, ...]]
    line_weights_by_isotope: dict[str, tuple[float, ...]]


def _validated_patch_quadrature(
    patches: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return validated patch areas and fixed-width quadrature arrays."""
    if hasattr(patches, "quadrature_points_xyz") and hasattr(
        patches, "quadrature_weights"
    ):
        points = np.asarray(getattr(patches, "quadrature_points_xyz"), dtype=float)
        weights = np.asarray(getattr(patches, "quadrature_weights"), dtype=float)
    else:
        if not hasattr(patches, "patches"):
            raise TypeError(
                "patches must provide aggregate quadrature arrays or a patches sequence."
            )
        patch_items = tuple(getattr(patches, "patches"))
        if not patch_items:
            raise ValueError("patches must contain at least one surface patch.")
        rows: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for patch_index, patch in enumerate(patch_items):
            patch_points = np.asarray(patch.quadrature_points_xyz, dtype=float)
            patch_weights = np.asarray(
                patch.quadrature_weights,
                dtype=float,
            ).reshape(-1)
            if (
                patch_points.ndim != 2
                or patch_points.shape[1:] != (3,)
                or patch_points.shape[0] == 0
            ):
                raise ValueError(
                    "Patch "
                    f"{patch_index} quadrature_points_xyz must have shape (Q, 3), Q >= 1."
                )
            if patch_weights.shape != (patch_points.shape[0],):
                raise ValueError(
                    f"Patch {patch_index} quadrature weights must match its points."
                )
            rows.append((patch_points, patch_weights))
        maximum_count = max(int(row[1].size) for row in rows)
        points = np.empty((len(patch_items), maximum_count, 3), dtype=float)
        weights = np.zeros((len(patch_items), maximum_count), dtype=float)
        for patch_index, (patch_points, patch_weights) in enumerate(rows):
            count = int(patch_weights.size)
            points[patch_index, :count] = patch_points
            points[patch_index, count:] = patch_points[-1]
            weights[patch_index, :count] = patch_weights
    areas = np.asarray(getattr(patches, "areas_m2"), dtype=float).reshape(-1)
    if (
        points.ndim != 3
        or points.shape[2:] != (3,)
        or points.shape[0] == 0
        or points.shape[1] == 0
    ):
        raise ValueError(
            "quadrature_points_xyz must have non-empty shape (G, Q, 3)."
        )
    if weights.shape != points.shape[:2]:
        raise ValueError("quadrature_weights must have shape (G, Q).")
    if (
        areas.shape != (points.shape[0],)
        or not np.all(np.isfinite(areas))
        or np.any(areas <= 0.0)
    ):
        raise ValueError("areas_m2 must contain one finite positive value per patch.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        raise ValueError("Patch quadrature must contain only finite values.")
    if np.any(weights < 0.0) or not np.allclose(
        np.sum(weights, axis=1),
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Each patch's quadrature weights must be non-negative and sum to one."
        )
    return (
        np.ascontiguousarray(areas, dtype=np.float64),
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _line_entries(
    kernel: ContinuousKernel,
    isotope: str,
    *,
    require_line_resolved: bool,
) -> tuple[dict[str, float], ...]:
    """Return normalized line energy/weight/mu entries for one isotope."""
    table = kernel.line_mu_by_isotope
    raw: object | None = None
    if isinstance(table, Mapping):
        raw = table.get(isotope)
        if raw is None:
            normalized = {
                "".join(ch for ch in str(key).upper() if ch.isalnum()): value
                for key, value in table.items()
            }
            key = "".join(ch for ch in str(isotope).upper() if ch.isalnum())
            raw = normalized.get(key)
    entries: list[dict[str, float]] = []
    if isinstance(raw, (tuple, list)):
        for raw_index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            energy = float(item.get("energy_keV", np.nan))
            weight = float(item.get("weight", 0.0))
            mu_fe = float(item.get("fe", item.get("mu_fe", np.nan)))
            mu_pb = float(item.get("pb", item.get("mu_pb", np.nan)))
            if (
                np.isfinite(energy)
                and energy > 0.0
                and np.isfinite(weight)
                and weight > 0.0
                and np.isfinite(mu_fe)
                and mu_fe >= 0.0
                and np.isfinite(mu_pb)
                and mu_pb >= 0.0
            ):
                entries.append(
                    {
                        "energy_keV": energy,
                        "weight": weight,
                        "fe": mu_fe,
                        "pb": mu_pb,
                        "transport_line_index": float(raw_index),
                    }
                )
    if not entries:
        if require_line_resolved:
            raise ValueError(
                f"No line-resolved attenuation table is available for {isotope}."
            )
        library = default_library()
        nuclide = library.get(isotope)
        if nuclide is None or not nuclide.lines:
            raise ValueError(f"No gamma-line library entry is available for {isotope}.")
        mu_fe, mu_pb = kernel._mu_values(isotope)  # explicit diagnostic fallback
        entries = []
        for raw_index, line in enumerate(nuclide.lines):
            if float(line.intensity) <= 0.0:
                continue
            entries.append(
                {
                    "energy_keV": float(line.energy_keV),
                    "weight": max(float(line.intensity), 0.0),
                    "fe": float(mu_fe),
                    "pb": float(mu_pb),
                    "transport_line_index": float(raw_index),
                }
            )
    total_weight = float(sum(entry["weight"] for entry in entries))
    if total_weight <= 0.0:
        raise ValueError(f"Gamma-line weights for {isotope} sum to zero.")
    return tuple(
        {**entry, "weight": float(entry["weight"] / total_weight)}
        for entry in entries
    )


def _obstacle_grid_for_line(
    grid: ObstacleGrid | None,
    isotope: str,
    line_index: int,
    *,
    require_line_resolved: bool,
) -> ObstacleGrid | None:
    """Return a grid whose line table contains only the requested gamma line."""
    if grid is None or not grid.has_transport_model:
        return grid
    rows = grid.transport_line_mu_values(isotope)
    if rows is None:
        if require_line_resolved:
            raise ValueError(
                "No line-resolved obstacle attenuation table is available for "
                f"{isotope}; aggregate obstacle attenuation is not valid for "
                "spectral fitting."
            )
        return grid
    if not 0 <= int(line_index) < len(rows):
        raise ValueError(
            "Obstacle line attenuation table for "
            f"{isotope} has {len(rows)} rows but line index {line_index} was requested."
        )
    mu_by_isotope = dict(grid.transport_mu_by_isotope)
    return grid.with_transport_model(
        boxes_m=grid.transport_boxes_m,
        mu_by_isotope=mu_by_isotope,
        line_mu_by_isotope={str(isotope): (rows[int(line_index)],)},
    )


def _kernel_for_line(
    kernel: ContinuousKernel,
    isotope: str,
    line: Mapping[str, float],
    line_index: int,
    *,
    require_line_resolved: bool,
) -> ContinuousKernel:
    """Clone the shared kernel with one line-specific shield and obstacle row."""
    grid = _obstacle_grid_for_line(
        kernel.obstacle_grid,
        isotope,
        line_index,
        require_line_resolved=require_line_resolved,
    )
    return replace(
        kernel,
        line_mu_by_isotope={
            str(isotope): (
                {
                    "energy_keV": float(line["energy_keV"]),
                    "weight": 1.0,
                    "fe": float(line["fe"]),
                    "pb": float(line["pb"]),
                },
            )
        },
        obstacle_grid=grid,
    )


def build_spectral_nuisance_response(
    live_times_s: NDArray[np.float64],
    energy_bin_edges_keV: NDArray[np.float64],
    *,
    include_background: bool = True,
    include_scatter: bool = True,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Build non-negative background-rate and scatter-rate nuisance columns."""
    live_times = np.asarray(live_times_s, dtype=float)
    if live_times.ndim != 1:
        raise ValueError("live_times_s must be one-dimensional.")
    if not np.all(np.isfinite(live_times)) or np.any(live_times < 0.0):
        raise ValueError("live_times_s must contain finite non-negative values.")
    edges = np.asarray(energy_bin_edges_keV, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("energy_bin_edges_keV must be a one-dimensional bin edge array.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_bin_edges_keV must be finite and strictly increasing.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    if include_background:
        shape = default_background_shape(centers)
        shape = shape / max(float(np.sum(shape)), 1.0e-30)
        columns.append(live_times[:, None] * shape[None, :])
        names.append("background_rate_cps")
    if include_scatter:
        incident_energy = max(float(edges[-1]) * 0.85, 200.0)
        shape = compton_continuum_shape(centers, incident_energy, shape="exponential")
        shape = shape / max(float(np.sum(shape)), 1.0e-30)
        columns.append(live_times[:, None] * shape[None, :])
        names.append("scatter_rate_cps")
    if not columns:
        return np.zeros((live_times.size, centers.size, 0), dtype=float), ()
    return np.stack(columns, axis=-1), tuple(names)


def build_spectral_response(
    observations: object,
    patches: object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    *,
    chunk_size: int = 262144,
    continuum_to_peak: float = COMPTON_CONTINUUM_TO_PEAK,
    backscatter_fraction: float = BACKSCATTER_FRACTION,
    require_line_resolved: bool = True,
    include_background_nuisance: bool = True,
    include_scatter_nuisance: bool = True,
) -> SpectralResponseResult:
    """Build ``M x B x G x I`` line-resolved count response tensors."""
    kernel_chunk_size = _positive_integer(chunk_size, name="chunk_size")
    for name, value in {
        "continuum_to_peak": continuum_to_peak,
        "backscatter_fraction": backscatter_fraction,
    }.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    (
        detector_positions,
        fe_indices,
        pb_indices,
        live_times,
        edges,
    ) = _validated_observation_geometry(observations, kernel)
    measurement_count = int(detector_positions.shape[0])
    names = _isotope_names(isotopes)
    areas, quadrature_points, quadrature_weights = _validated_patch_quadrature(patches)
    patch_count, quadrature_count = quadrature_points.shape[:2]
    sources = quadrature_points.reshape(-1, 3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(np.median(np.diff(edges)))
    response = np.zeros(
        (measurement_count, centers.size, patch_count, len(names)), dtype=float
    )
    energies_by_isotope: dict[str, tuple[float, ...]] = {}
    weights_by_isotope: dict[str, tuple[float, ...]] = {}
    resolution = default_resolution()

    for isotope_index, isotope in enumerate(names):
        lines = _line_entries(
            kernel,
            isotope,
            require_line_resolved=require_line_resolved,
        )
        energies_by_isotope[isotope] = tuple(float(line["energy_keV"]) for line in lines)
        weights_by_isotope[isotope] = tuple(float(line["weight"]) for line in lines)
        for line_index, line in enumerate(lines):
            transport_line_index = int(
                line.get("transport_line_index", float(line_index))
            )
            line_kernel = _kernel_for_line(
                kernel,
                isotope,
                line,
                transport_line_index,
                require_line_resolved=require_line_resolved,
            )
            raw_values = np.asarray(
                line_kernel.kernel_values_selected_pairs_for_detectors(
                    isotope=isotope,
                    detector_positions=detector_positions,
                    sources=sources,
                    fe_indices=fe_indices,
                    pb_indices=pb_indices,
                    chunk_size=kernel_chunk_size,
                ),
                dtype=np.float64,
            )
            expected_shape = (
                measurement_count,
                patch_count * quadrature_count,
            )
            if raw_values.shape != expected_shape:
                raise ValueError(
                    "Batched selected-pair kernel returned shape "
                    f"{raw_values.shape}, expected {expected_shape}."
                )
            if not np.all(np.isfinite(raw_values)) or np.any(raw_values < 0.0):
                raise ValueError(
                    "Batched selected-pair kernel must return finite non-negative values."
                )
            values = raw_values.reshape(
                measurement_count,
                patch_count,
                quadrature_count,
            )
            spatial = live_times[:, None] * np.einsum(
                "mgq,gq->mg", values, quadrature_weights, optimize=True
            )
            pulse = detector_response_kernel_for_incident_gamma(
                centers,
                float(line["energy_keV"]),
                resolution,
                cebr3_efficiency,
                bin_width,
                continuum_to_peak=float(continuum_to_peak),
                backscatter_fraction=float(backscatter_fraction),
            )
            if pulse.shape != centers.shape:
                raise ValueError("Detector response returned an incompatible bin shape.")
            if not np.all(np.isfinite(pulse)) or np.any(pulse < 0.0):
                raise ValueError(
                    "Detector response must contain finite non-negative values."
                )
            response[:, :, :, isotope_index] += (
                float(line["weight"])
                * spatial[:, None, :]
                * pulse[None, :, None]
            )

    nuisance, nuisance_names = build_spectral_nuisance_response(
        live_times,
        edges,
        include_background=include_background_nuisance,
        include_scatter=include_scatter_nuisance,
    )
    return SpectralResponseResult(
        response_per_integrated_strength=response,
        response_per_density=response * areas[None, None, :, None],
        nuisance_response=nuisance,
        nuisance_names=nuisance_names,
        line_energies_keV_by_isotope=energies_by_isotope,
        line_weights_by_isotope=weights_by_isotope,
    )
