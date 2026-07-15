"""Build count-domain surface responses from the shared physical kernel.

The response builder deliberately delegates all detector, shield, and obstacle
physics to :class:`measurement.continuous_kernels.ContinuousKernel`.  It only
adds acquisition live time and normalized surface-patch quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel


@runtime_checkable
class ObservationBatchLike(Protocol):
    """Structural input required from an estimator-independent observation batch."""

    detector_positions_xyz: object
    fe_indices: object
    pb_indices: object
    live_times_s: object


@runtime_checkable
class SurfacePatchSetLike(Protocol):
    """Structural input required from an area-aware surface patch set."""

    areas_m2: object
    quadrature_points_xyz: object
    quadrature_weights: object


@dataclass(frozen=True)
class CountResponseMatrices:
    """Count responses for the two supported surface parameterizations.

    ``response_by_integrated_strength`` maps patch-integrated source strength
    in detector cps at 1 m to expected counts.  ``response_by_density`` maps
    source density in detector cps at 1 m per square metre to expected counts.
    Both arrays have shape ``(measurements, patches, isotopes)``.
    """

    response_by_integrated_strength: NDArray[np.float64]
    response_by_density: NDArray[np.float64]
    isotope_names: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedResponseInputs:
    """Validated arrays used by the chunked response calculation."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    patch_areas_m2: NDArray[np.float64]
    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    isotope_names: tuple[str, ...]


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive built-in integer or raise a clear validation error."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _required_attribute(
    value: object,
    name: str,
    *,
    aliases: Sequence[str] = (),
) -> object:
    """Return one required structural attribute, accepting documented aliases."""
    for candidate in (name, *aliases):
        if hasattr(value, candidate):
            return getattr(value, candidate)
    alias_text = "" if not aliases else f" (accepted aliases: {', '.join(aliases)})"
    raise TypeError(f"Input must provide {name!r}{alias_text}.")


def _finite_array(value: object, *, name: str, ndim: int) -> NDArray[np.float64]:
    """Return a finite float64 array with the requested rank."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array.") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _orientation_indices(
    value: object,
    *,
    name: str,
    measurement_count: int,
    orientation_count: int,
) -> NDArray[np.int64]:
    """Return one validated shield-orientation index per measurement."""
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (measurement_count,):
        raise ValueError(
            f"{name} must have shape ({measurement_count},); got {raw.shape}."
        )
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must contain integer indices.")
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= orientation_count):
        raise ValueError(
            f"{name} entries must lie in [0, {orientation_count - 1}]."
        )
    return np.ascontiguousarray(indices)


def _isotope_names(isotopes: Sequence[str]) -> tuple[str, ...]:
    """Return validated, ordered isotope names."""
    names = tuple(isotopes)
    if not names:
        raise ValueError("isotopes must contain at least one isotope name.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("isotopes must contain only non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("isotopes must not contain duplicates.")
    return names


def _patch_quadrature_arrays(
    patches: SurfacePatchSetLike | object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return dense patch quadrature arrays from aggregate or per-patch data.

    Canonical ``SurfacePatchSet`` instances expose aggregate ``(G, Q, 3)`` and
    ``(G, Q)`` arrays.  The per-patch representation is also accepted so this
    builder remains compatible with sets containing a mixture of one- and
    four-point rectangular quadrature rules.  Mixed rules are padded with a
    repeated valid point and zero weight; the physical integral is unchanged.
    """
    if hasattr(patches, "quadrature_points_xyz") and hasattr(
        patches,
        "quadrature_weights",
    ):
        points = _finite_array(
            getattr(patches, "quadrature_points_xyz"),
            name="patches.quadrature_points_xyz",
            ndim=3,
        )
        weights = _finite_array(
            getattr(patches, "quadrature_weights"),
            name="patches.quadrature_weights",
            ndim=2,
        )
        return points, weights

    patch_items = _required_attribute(patches, "patches")
    if not isinstance(patch_items, Sequence) or isinstance(
        patch_items,
        (str, bytes),
    ):
        raise TypeError("patches.patches must be a sequence of surface patches.")
    patch_sequence = tuple(patch_items)
    if not patch_sequence:
        raise ValueError("patches.patches must contain at least one patch.")

    point_rows: list[NDArray[np.float64]] = []
    weight_rows: list[NDArray[np.float64]] = []
    for patch_index, patch in enumerate(patch_sequence):
        points = _finite_array(
            _required_attribute(patch, "quadrature_points_xyz"),
            name=f"patches.patches[{patch_index}].quadrature_points_xyz",
            ndim=2,
        )
        weights = _finite_array(
            _required_attribute(patch, "quadrature_weights"),
            name=f"patches.patches[{patch_index}].quadrature_weights",
            ndim=1,
        )
        if points.shape[0] == 0 or points.shape[1:] != (3,):
            raise ValueError(
                "Each patch quadrature_points_xyz must have non-empty shape (Q, 3)."
            )
        if weights.shape != (points.shape[0],):
            raise ValueError(
                "Each patch quadrature_weights vector must match its quadrature points."
            )
        point_rows.append(points)
        weight_rows.append(weights)

    maximum_count = max(row.shape[0] for row in point_rows)
    dense_points = np.empty((len(point_rows), maximum_count, 3), dtype=np.float64)
    dense_weights = np.zeros((len(point_rows), maximum_count), dtype=np.float64)
    for patch_index, (points, weights) in enumerate(zip(point_rows, weight_rows)):
        point_count = int(points.shape[0])
        dense_points[patch_index, :point_count] = points
        dense_points[patch_index, point_count:] = points[-1]
        dense_weights[patch_index, :point_count] = weights
    return dense_points, dense_weights


def _prepare_inputs(
    observations: ObservationBatchLike | object,
    patches: SurfacePatchSetLike | object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
) -> _PreparedResponseInputs:
    """Validate all response dimensions, physical values, and pair indices."""
    detector_positions = _finite_array(
        _required_attribute(observations, "detector_positions_xyz"),
        name="observations.detector_positions_xyz",
        ndim=2,
    )
    if detector_positions.shape[1:] != (3,) or detector_positions.shape[0] == 0:
        raise ValueError(
            "observations.detector_positions_xyz must have non-empty shape (M, 3)."
        )
    measurement_count = int(detector_positions.shape[0])

    live_times = _finite_array(
        _required_attribute(observations, "live_times_s"),
        name="observations.live_times_s",
        ndim=1,
    )
    if live_times.shape != (measurement_count,):
        raise ValueError(
            "observations.live_times_s must have one entry per measurement."
        )
    if np.any(live_times < 0.0):
        raise ValueError("observations.live_times_s must be non-negative.")

    if not hasattr(kernel, "kernel_values_selected_pairs_for_detectors"):
        raise TypeError(
            "kernel must provide kernel_values_selected_pairs_for_detectors()."
        )
    orientations = _finite_array(
        _required_attribute(kernel, "orientations"),
        name="kernel.orientations",
        ndim=2,
    )
    if orientations.shape[0] == 0 or orientations.shape[1:] != (3,):
        raise ValueError("kernel.orientations must have non-empty shape (R, 3).")
    orientation_count = int(orientations.shape[0])
    fe_indices = _orientation_indices(
        _required_attribute(
            observations,
            "fe_indices",
            aliases=("fe_orientation_indices",),
        ),
        name="observations.fe_indices",
        measurement_count=measurement_count,
        orientation_count=orientation_count,
    )
    pb_indices = _orientation_indices(
        _required_attribute(
            observations,
            "pb_indices",
            aliases=("pb_orientation_indices",),
        ),
        name="observations.pb_indices",
        measurement_count=measurement_count,
        orientation_count=orientation_count,
    )

    points, weights = _patch_quadrature_arrays(patches)
    if points.shape[0] == 0 or points.shape[1] == 0 or points.shape[2:] != (3,):
        raise ValueError(
            "patches.quadrature_points_xyz must have non-empty shape (G, Q, 3)."
        )
    patch_count, quadrature_count = int(points.shape[0]), int(points.shape[1])
    if weights.shape != (patch_count, quadrature_count):
        raise ValueError(
            "patches.quadrature_weights must have shape (G, Q) matching "
            "quadrature_points_xyz."
        )
    if np.any(weights < 0.0):
        raise ValueError("patches.quadrature_weights must be non-negative.")
    weight_sums = np.sum(weights, axis=1)
    if not np.allclose(weight_sums, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(
            "patches.quadrature_weights must sum to one independently for each patch."
        )

    areas = _finite_array(
        _required_attribute(patches, "areas_m2"),
        name="patches.areas_m2",
        ndim=1,
    )
    if areas.shape != (patch_count,):
        raise ValueError("patches.areas_m2 must have one entry per patch.")
    if np.any(areas <= 0.0):
        raise ValueError("patches.areas_m2 must contain only positive values.")

    return _PreparedResponseInputs(
        detector_positions_xyz=detector_positions,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times_s=live_times,
        patch_areas_m2=areas,
        quadrature_points_xyz=points,
        quadrature_weights=weights,
        isotope_names=_isotope_names(isotopes),
    )


def _build_integrated_response(
    inputs: _PreparedResponseInputs,
    kernel: ContinuousKernel,
    *,
    measurement_chunk_size: int,
    patch_chunk_size: int,
    kernel_chunk_size: int,
) -> NDArray[np.float64]:
    """Evaluate live-time-weighted normalized patch quadrature in fixed chunks."""
    measurement_count = int(inputs.detector_positions_xyz.shape[0])
    patch_count, quadrature_count = inputs.quadrature_points_xyz.shape[:2]
    response = np.empty(
        (measurement_count, patch_count, len(inputs.isotope_names)),
        dtype=np.float64,
    )
    evaluate = kernel.kernel_values_selected_pairs_for_detectors

    for isotope_index, isotope in enumerate(inputs.isotope_names):
        for measurement_start in range(0, measurement_count, measurement_chunk_size):
            measurement_stop = min(
                measurement_start + measurement_chunk_size,
                measurement_count,
            )
            measurement_slice = slice(measurement_start, measurement_stop)
            detectors = inputs.detector_positions_xyz[measurement_slice]
            fe_indices = inputs.fe_indices[measurement_slice]
            pb_indices = inputs.pb_indices[measurement_slice]
            live_times = inputs.live_times_s[measurement_slice]

            for patch_start in range(0, patch_count, patch_chunk_size):
                patch_stop = min(patch_start + patch_chunk_size, patch_count)
                patch_slice = slice(patch_start, patch_stop)
                points = np.ascontiguousarray(
                    inputs.quadrature_points_xyz[patch_slice].reshape(-1, 3)
                )
                kernel_values = np.asarray(
                    evaluate(
                        isotope=isotope,
                        detector_positions=detectors,
                        sources=points,
                        fe_indices=fe_indices,
                        pb_indices=pb_indices,
                        chunk_size=kernel_chunk_size,
                    ),
                    dtype=np.float64,
                )
                expected_shape = (
                    measurement_stop - measurement_start,
                    (patch_stop - patch_start) * quadrature_count,
                )
                if kernel_values.shape != expected_shape:
                    raise ValueError(
                        "Batched selected-pair kernel returned shape "
                        f"{kernel_values.shape}, expected {expected_shape}."
                    )
                if not np.all(np.isfinite(kernel_values)):
                    raise ValueError("Batched selected-pair kernel returned non-finite values.")
                if np.any(kernel_values < 0.0):
                    raise ValueError("Batched selected-pair kernel returned negative values.")

                shaped_values = kernel_values.reshape(
                    measurement_stop - measurement_start,
                    patch_stop - patch_start,
                    quadrature_count,
                )
                weighted_values = np.sum(
                    shaped_values
                    * inputs.quadrature_weights[patch_slice][None, :, :],
                    axis=2,
                )
                response[
                    measurement_slice,
                    patch_slice,
                    isotope_index,
                ] = live_times[:, None] * weighted_values
    return response


def build_count_responses(
    observations: ObservationBatchLike | object,
    patches: SurfacePatchSetLike | object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    *,
    measurement_chunk_size: int = 128,
    patch_chunk_size: int = 512,
    kernel_chunk_size: int = 262_144,
) -> CountResponseMatrices:
    """Build solver-ready integrated-strength and area-scaled density responses.

    Quadrature weights are normalized within each patch.  Consequently the
    integrated-strength response is ``live_time * sum(weight * kernel)`` and
    the density response is that value multiplied by ``patch area``.
    Chunk boundaries affect memory use only and never alter output ordering.
    """
    measurement_chunk = _positive_integer(
        measurement_chunk_size,
        name="measurement_chunk_size",
    )
    patch_chunk = _positive_integer(patch_chunk_size, name="patch_chunk_size")
    kernel_chunk = _positive_integer(kernel_chunk_size, name="kernel_chunk_size")
    inputs = _prepare_inputs(observations, patches, isotopes, kernel)
    integrated = _build_integrated_response(
        inputs,
        kernel,
        measurement_chunk_size=measurement_chunk,
        patch_chunk_size=patch_chunk,
        kernel_chunk_size=kernel_chunk,
    )
    density = integrated * inputs.patch_areas_m2[None, :, None]
    return CountResponseMatrices(
        response_by_integrated_strength=integrated,
        response_by_density=density,
        isotope_names=inputs.isotope_names,
    )


def build_count_response(
    observations: ObservationBatchLike | object,
    patches: SurfacePatchSetLike | object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    *,
    measurement_chunk_size: int = 128,
    patch_chunk_size: int = 512,
    kernel_chunk_size: int = 262_144,
) -> NDArray[np.float64]:
    """Return the ``M x G x I`` unit-integrated-strength response for the solver."""
    return build_count_responses(
        observations,
        patches,
        isotopes,
        kernel,
        measurement_chunk_size=measurement_chunk_size,
        patch_chunk_size=patch_chunk_size,
        kernel_chunk_size=kernel_chunk_size,
    ).response_by_integrated_strength


def build_density_response(
    observations: ObservationBatchLike | object,
    patches: SurfacePatchSetLike | object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    *,
    measurement_chunk_size: int = 128,
    patch_chunk_size: int = 512,
    kernel_chunk_size: int = 262_144,
) -> NDArray[np.float64]:
    """Return the ``M x G x I`` response for cps@1m/m2 patch densities."""
    return build_count_responses(
        observations,
        patches,
        isotopes,
        kernel,
        measurement_chunk_size=measurement_chunk_size,
        patch_chunk_size=patch_chunk_size,
        kernel_chunk_size=kernel_chunk_size,
    ).response_by_density


__all__ = [
    "CountResponseMatrices",
    "ObservationBatchLike",
    "SurfacePatchSetLike",
    "build_count_response",
    "build_count_responses",
    "build_density_response",
]
