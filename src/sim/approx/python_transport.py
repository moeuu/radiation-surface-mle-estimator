"""Approximate Python transport for diagnostics and non-Geant4 debug modes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from measurement.continuous_kernels import segment_box_intersection_length_m
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    OctantShield,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    path_length_cm,
    resolve_mu_values,
    rotated_positive_octant_blocks_direction,
    spherical_shell_path_length_cm,
)
from sim.protocol import SimulationCommand, SimulationObservation
from sim.transport import (
    SourceTransportResult,
    TransportMaterial,
    TransportSegment,
    build_source_transport_result,
    make_transport_segment,
)
from spectrum.dead_time import non_paralyzable_correction
from spectrum.pipeline import BACKGROUND_COUNTS_PER_SECOND, BACKGROUND_RATE_CPS, SpectralDecomposer
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    backscatter_energy,
    compton_continuum_shape,
    gaussian_peak,
)

StageSegmentsProvider = Callable[
    [PointSource, tuple[float, float, float]],
    Sequence[TransportSegment],
]
ShieldPathProvider = Callable[
    [PointSource, tuple[float, float, float], SimulationCommand],
    tuple[float, float],
]


@dataclass
class PythonTransportScene:
    """Store scene inputs used by the shared Python transport model."""

    sources: list[PointSource] = field(default_factory=list)
    obstacle_grid: ObstacleGrid | None = None
    obstacle_material: str = "concrete"


def energy_bin_edges_keV(decomposer: SpectralDecomposer) -> np.ndarray:
    """Return energy bin edges compatible with the decomposer energy axis."""
    energy = np.asarray(decomposer.energy_axis, dtype=float)
    if energy.size == 0:
        return energy.copy()
    step = float(np.median(np.diff(energy))) if energy.size > 1 else 1.0
    return np.concatenate([energy, [energy[-1] + step]])


def point_sources_from_payload(payload: Mapping[str, Any]) -> list[PointSource]:
    """Parse point sources from a simulator reset payload."""
    sources_payload = payload.get("sources", [])
    if not isinstance(sources_payload, list):
        raise ValueError("sources must be a list.")
    sources: list[PointSource] = []
    for index, entry in enumerate(sources_payload):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Source entry {index} must be an object.")
        position_payload = entry.get("position", (0.0, 0.0, 0.0))
        if not isinstance(position_payload, (list, tuple)) or len(position_payload) != 3:
            raise ValueError(f"Source entry {index} position must have three values.")
        sources.append(
            PointSource(
                isotope=str(entry.get("isotope", f"source_{index}")),
                position=(
                    float(position_payload[0]),
                    float(position_payload[1]),
                    float(position_payload[2]),
                ),
                intensity_cps_1m=float(entry.get("intensity_cps_1m", 0.0)),
            )
        )
    return sources


def obstacle_grid_from_payload(payload: Mapping[str, Any]) -> ObstacleGrid | None:
    """Parse an obstacle grid from a simulator reset payload."""
    shape_payload = payload.get("obstacle_grid_shape", (0, 0))
    if not isinstance(shape_payload, (list, tuple)) or len(shape_payload) != 2:
        raise ValueError("obstacle_grid_shape must have two values.")
    grid_shape = (int(shape_payload[0]), int(shape_payload[1]))
    if grid_shape[0] <= 0 or grid_shape[1] <= 0:
        return None
    origin_payload = payload.get("obstacle_origin_xy", (0.0, 0.0))
    if not isinstance(origin_payload, (list, tuple)) or len(origin_payload) != 2:
        raise ValueError("obstacle_origin_xy must have two values.")
    cells_payload = payload.get("obstacle_cells", [])
    if not isinstance(cells_payload, list):
        raise ValueError("obstacle_cells must be a list.")
    return ObstacleGrid(
        origin=(float(origin_payload[0]), float(origin_payload[1])),
        cell_size=float(payload.get("obstacle_cell_size_m", 1.0)),
        grid_shape=grid_shape,
        blocked_cells=tuple((int(cell[0]), int(cell[1])) for cell in cells_payload),
    )


class PythonTransportSpectrumModel:
    """Generate spectra with shared Python geometry and detector-response logic."""

    def __init__(
        self,
        *,
        sources: Iterable[PointSource] = (),
        decomposer: SpectralDecomposer | None = None,
        mu_by_isotope: Mapping[str, object] | None = None,
        shield_params: ShieldParams | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_material: str = "concrete",
        scatter_gain: float = 0.03,
        rng_seed: int = 123,
        dead_time_s: float = 0.0,
        detector_model: Mapping[str, Any] | None = None,
    ) -> None:
        """Store model configuration and scene state."""
        self.decomposer = decomposer or SpectralDecomposer()
        self.mu_by_isotope = dict(mu_by_isotope or {})
        self.shield_params = shield_params or ShieldParams()
        self.obstacle_height_m = float(obstacle_height_m)
        self.scatter_gain = float(scatter_gain)
        self.rng_seed = int(rng_seed)
        self.dead_time_s = float(dead_time_s)
        self.detector_model = dict(detector_model or {})
        self.octant_shield = OctantShield()
        self.orientations = generate_octant_orientations()
        self.scene = PythonTransportScene(
            sources=list(sources),
            obstacle_grid=obstacle_grid,
            obstacle_material=str(obstacle_material),
        )
        self._line_response_cache: dict[float, np.ndarray] = {}
        self._scatter_response_cache: dict[float, np.ndarray] = {}

    def reset_from_payload(self, payload: Mapping[str, Any] | None) -> None:
        """Reset sources, obstacle geometry, and optional detector metadata."""
        if payload is None:
            return
        self.scene = PythonTransportScene(
            sources=point_sources_from_payload(payload),
            obstacle_grid=obstacle_grid_from_payload(payload),
            obstacle_material=str(payload.get("obstacle_material", self.scene.obstacle_material)),
        )
        detector_model = payload.get("detector_model")
        if detector_model is not None:
            if not isinstance(detector_model, Mapping):
                raise ValueError("detector_model must be an object.")
            self.detector_model = dict(detector_model)

    def reset_scene(
        self,
        *,
        sources: Iterable[PointSource],
        obstacle_grid: ObstacleGrid | None,
        obstacle_material: str,
    ) -> None:
        """Reset the active scene from already parsed scene objects."""
        self.scene = PythonTransportScene(
            sources=list(sources),
            obstacle_grid=obstacle_grid,
            obstacle_material=str(obstacle_material),
        )

    def observe(
        self,
        command: SimulationCommand,
        *,
        detector_pose_xyz: tuple[float, float, float],
        detector_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        backend_label: str,
        sources: Iterable[PointSource] | None = None,
        stage_segments_provider: StageSegmentsProvider | None = None,
        shield_path_provider: ShieldPathProvider | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> SimulationObservation:
        """Generate one sampled spectrum observation."""
        transport_results = self.source_transport_results(
            command,
            detector_pose_xyz=detector_pose_xyz,
            sources=self.scene.sources if sources is None else sources,
            stage_segments_provider=stage_segments_provider,
            shield_path_provider=shield_path_provider,
        )
        expected = self.expected_spectrum_from_transport_results(transport_results, command.dwell_time_s)
        spectrum = self.sample_spectrum(expected, command.step_id)
        metadata = self.metadata_from_transport_results(
            transport_results,
            backend_label=backend_label,
            expected_spectrum=expected,
            extra_metadata=extra_metadata,
        )
        num_orientations = max(1, int(len(self.orientations)))
        metadata.setdefault("fe_orientation_index", int(command.fe_orientation_index))
        metadata.setdefault("pb_orientation_index", int(command.pb_orientation_index))
        metadata.setdefault("shield_num_orientations", int(num_orientations))
        metadata.setdefault(
            "shield_pair_id",
            int(command.fe_orientation_index) * num_orientations
            + int(command.pb_orientation_index),
        )
        metadata.setdefault(
            "shield_thickness_fe_cm",
            float(self.shield_params.thickness_fe_cm),
        )
        metadata.setdefault(
            "shield_thickness_pb_cm",
            float(self.shield_params.thickness_pb_cm),
        )
        metadata.setdefault(
            "shield_thickness_scale",
            0.0
            if (
                float(self.shield_params.thickness_fe_cm) <= 0.0
                and float(self.shield_params.thickness_pb_cm) <= 0.0
            )
            else 1.0,
        )
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=tuple(float(value) for value in detector_pose_xyz),
            detector_quat_wxyz=tuple(float(value) for value in detector_quat_wxyz),
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.asarray(spectrum, dtype=float).tolist(),
            energy_bin_edges_keV=energy_bin_edges_keV(self.decomposer).tolist(),
            metadata=metadata,
        )

    def source_transport_results(
        self,
        command: SimulationCommand,
        *,
        detector_pose_xyz: tuple[float, float, float],
        sources: Iterable[PointSource],
        stage_segments_provider: StageSegmentsProvider | None = None,
        shield_path_provider: ShieldPathProvider | None = None,
    ) -> tuple[SourceTransportResult, ...]:
        """Build transport results for all source-detector pairs."""
        detector_position = tuple(float(value) for value in detector_pose_xyz)
        results: list[SourceTransportResult] = []
        for source in sources:
            if source.isotope not in self.decomposer.library:
                continue
            if stage_segments_provider is None:
                stage_segments = self.obstacle_stage_segments(source, detector_position)
            else:
                stage_segments = tuple(stage_segments_provider(source, detector_position))
            if shield_path_provider is None:
                fe_path_cm, pb_path_cm = self.shield_path_lengths_cm(
                    source,
                    detector_position,
                    command.fe_orientation_index,
                    command.pb_orientation_index,
                )
            else:
                fe_path_cm, pb_path_cm = shield_path_provider(source, detector_position, command)
            nuclide = self.decomposer.library[source.isotope]
            nuclide_lines = tuple((float(line.energy_keV), float(line.intensity)) for line in nuclide.lines)
            results.append(
                build_source_transport_result(
                    source=source,
                    detector_position_xyz=detector_position,
                    dwell_time_s=float(command.dwell_time_s),
                    stage_segments=stage_segments,
                    fe_segment=self._shield_segment(source.isotope, "fe", fe_path_cm),
                    pb_segment=self._shield_segment(source.isotope, "pb", pb_path_cm),
                    nuclide_lines=nuclide_lines,
                    scatter_gain=self.scatter_gain,
                )
            )
        return tuple(results)

    def obstacle_stage_segments(
        self,
        source: PointSource,
        detector_pose_xyz: tuple[float, float, float],
    ) -> tuple[TransportSegment, ...]:
        """Return material segments through reset-payload obstacle boxes."""
        obstacle_grid = self.scene.obstacle_grid
        if obstacle_grid is None:
            return ()
        source_position = np.asarray(source.position, dtype=float)
        detector_position = np.asarray(detector_pose_xyz, dtype=float)
        material = TransportMaterial(name=self.scene.obstacle_material)
        segments: list[TransportSegment] = []
        for box in obstacle_grid.blocked_boxes(z_min=0.0, z_max=self.obstacle_height_m):
            path_length_cm = 100.0 * segment_box_intersection_length_m(
                source_position,
                detector_position,
                np.asarray(box, dtype=float),
            )
            if path_length_cm <= 0.0:
                continue
            segments.append(
                make_transport_segment(
                    material,
                    path_length_cm,
                    is_obstacle=True,
                )
            )
        return tuple(segments)

    def shield_path_lengths_cm(
        self,
        source: PointSource,
        detector_pose_xyz: tuple[float, float, float],
        fe_orientation_index: int,
        pb_orientation_index: int,
    ) -> tuple[float, float]:
        """Return Fe and Pb spherical-octant shell path lengths."""
        source_pos = source.position_array()
        detector_pos = np.asarray(detector_pose_xyz, dtype=float)
        direction = detector_pos - source_pos
        fe_index = int(fe_orientation_index) % len(self.orientations)
        pb_index = int(pb_orientation_index) % len(self.orientations)
        detector_to_source = source_pos - detector_pos
        fe_blocked = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -self.orientations[fe_index],
        )
        pb_blocked = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -self.orientations[pb_index],
        )
        fe_path = self._shield_path_length_cm(
            direction_m=direction,
            normal=self.orientations[fe_index],
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=fe_blocked,
        )
        pb_path = self._shield_path_length_cm(
            direction_m=direction,
            normal=self.orientations[pb_index],
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=pb_blocked,
        )
        return float(fe_path), float(pb_path)

    def expected_spectrum_from_transport_results(
        self,
        transport_results: Iterable[SourceTransportResult],
        dwell_time_s: float,
    ) -> np.ndarray:
        """Return the expected detector spectrum from transport results."""
        expected = np.zeros_like(self.decomposer.energy_axis, dtype=float)
        for transport_result in transport_results:
            expected += self.source_expected_spectrum(transport_result)
        background_rate = BACKGROUND_RATE_CPS
        if BACKGROUND_COUNTS_PER_SECOND != BACKGROUND_RATE_CPS:
            background_rate = BACKGROUND_COUNTS_PER_SECOND
        if background_rate > 0.0:
            expected += self.decomposer._background_shape * float(background_rate) * float(dwell_time_s)
        return np.clip(expected, a_min=0.0, a_max=None)

    def source_expected_spectrum(
        self,
        transport_result: SourceTransportResult,
    ) -> np.ndarray:
        """Return the expected spectrum contribution for one transported source."""
        if not transport_result.lines or transport_result.base_source_counts <= 0.0:
            return np.zeros_like(self.decomposer.energy_axis, dtype=float)
        expected = np.zeros_like(self.decomposer.energy_axis, dtype=float)
        for line in transport_result.lines:
            expected += float(line.primary_counts) * self.line_response_template(line.energy_keV)
            if line.scatter_counts > 0.0:
                expected += float(line.scatter_counts) * self.scatter_response_template(line.energy_keV)
        return expected

    def sample_spectrum(self, expected_spectrum: np.ndarray, step_id: int) -> np.ndarray:
        """Sample Poisson counting noise and apply optional dead-time correction."""
        rng = np.random.default_rng(self.rng_seed + int(step_id))
        sampled = rng.poisson(np.clip(expected_spectrum, a_min=0.0, a_max=None))
        if self.dead_time_s > 0.0:
            return non_paralyzable_correction(sampled, dead_time_s=self.dead_time_s)
        return np.asarray(sampled, dtype=float)

    def metadata_from_transport_results(
        self,
        transport_results: Sequence[SourceTransportResult],
        *,
        backend_label: str,
        expected_spectrum: np.ndarray,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build common observation metadata for Python transport observations."""
        metadata: dict[str, Any] = {
            "backend": str(backend_label),
            "transport_backend": "python",
            "python_transport_model": "line_response_attenuation_scatter",
            "num_sources": int(len(transport_results)),
            "total_obstacle_path_cm": float(
                sum(result.total_obstacle_path_cm for result in transport_results)
            ),
            "total_stage_path_cm": float(sum(result.total_stage_path_cm for result in transport_results)),
            "total_fe_path_cm": float(sum(result.total_fe_path_cm for result in transport_results)),
            "total_pb_path_cm": float(sum(result.total_pb_path_cm for result in transport_results)),
            "expected_total_counts": float(np.sum(expected_spectrum)),
            "scatter_gain": float(self.scatter_gain),
            "dead_time_s": float(self.dead_time_s),
        }
        if self.detector_model:
            metadata["detector_model"] = dict(self.detector_model)
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        return metadata

    def line_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return the detector response template for a unit-intensity gamma line."""
        cache_key = float(line_energy_keV)
        cached = self._line_response_cache.get(cache_key)
        if cached is not None:
            return cached
        energy_axis = np.asarray(self.decomposer.energy_axis, dtype=float)
        bin_width_keV = float(self.decomposer.config.bin_width_keV)
        sigma = float(self.decomposer.resolution_fn(cache_key))
        peak = gaussian_peak(energy_axis, center=cache_key, sigma=sigma)
        peak_area = float(peak.sum() * bin_width_keV)
        efficiency = (
            float(self.decomposer.efficiency_fn(cache_key))
            if self.decomposer.efficiency_fn is not None
            else 1.0
        )
        response = peak * efficiency * bin_width_keV
        cont_shape = compton_continuum_shape(energy_axis, cache_key, shape="exponential")
        if cont_shape.sum() > 0.0:
            cont_shape = cont_shape / float(cont_shape.sum())
            response += COMPTON_CONTINUUM_TO_PEAK * peak_area * cont_shape * efficiency
        if cache_key > 200.0:
            e_back = backscatter_energy(cache_key)
            sigma_back = float(self.decomposer.resolution_fn(e_back))
            back = gaussian_peak(energy_axis, center=e_back, sigma=sigma_back)
            back_norm = float(back.sum() * bin_width_keV)
            if back_norm > 0.0:
                area_back = BACKSCATTER_FRACTION * peak_area
                back_efficiency = (
                    float(self.decomposer.efficiency_fn(e_back))
                    if self.decomposer.efficiency_fn is not None
                    else 1.0
                )
                response += back * (area_back / back_norm) * back_efficiency
        self._line_response_cache[cache_key] = response
        return response

    def scatter_response_template(self, line_energy_keV: float) -> np.ndarray:
        """Return a scatter-dominated low-energy response for one gamma line."""
        cache_key = float(line_energy_keV)
        cached = self._scatter_response_cache.get(cache_key)
        if cached is not None:
            return cached
        energy_axis = np.asarray(self.decomposer.energy_axis, dtype=float)
        response = compton_continuum_shape(energy_axis, cache_key, shape="exponential")
        if float(np.sum(response)) > 0.0:
            response = response / float(np.sum(response))
        if cache_key > 200.0:
            e_back = backscatter_energy(cache_key)
            sigma_back = float(self.decomposer.resolution_fn(e_back))
            response += 0.25 * gaussian_peak(energy_axis, center=e_back, sigma=sigma_back)
        if float(np.sum(response)) > 0.0:
            response = response / float(np.sum(response))
        self._scatter_response_cache[cache_key] = response
        return response

    def _shield_path_length_cm(
        self,
        *,
        direction_m: np.ndarray,
        normal: np.ndarray,
        thickness_cm: float,
        inner_radius_cm: float,
        blocked: bool,
    ) -> float:
        """Return the configured shield-geometry path length."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            return spherical_shell_path_length_cm(
                direction_m=direction_m,
                inner_radius_cm=float(inner_radius_cm),
                outer_radius_cm=float(inner_radius_cm) + float(thickness_cm),
                blocked=blocked,
            )
        return path_length_cm(
            direction_m,
            normal,
            float(thickness_cm),
            blocked=blocked,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )

    def _shield_segment(
        self,
        isotope: str,
        material_name: str,
        path_length_cm: float,
    ) -> TransportSegment:
        """Build a shield segment using the PF TVL coefficients for this isotope."""
        mu_fe, mu_pb = resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )
        mu = mu_fe if material_name == "fe" else mu_pb
        material = TransportMaterial(
            name=material_name,
            mu_by_isotope={str(isotope): float(mu)},
        )
        return make_transport_segment(material, float(path_length_cm))
