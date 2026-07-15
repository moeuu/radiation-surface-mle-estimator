"""Build lightweight radiation visualization samples for simulator sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from measurement.model import PointSource, inverse_square_scale
from sim.isaacsim_app.geometry import (
    OrientedBox,
    Sphere,
    quaternion_wxyz_to_matrix,
    segment_path_length_through_box,
    segment_path_length_through_sphere,
)
from sim.transport import SourceTransportResult, make_transport_segment, material_transmission, scatter_scale


@dataclass(frozen=True)
class RadiationVisualizationConfig:
    """Control representative radiation track and hit sampling."""

    enabled: bool = True
    max_tracks: int = 96
    max_hits: int = 24
    track_length_m: float = 12.0
    source_jitter_m: float = 0.03
    detector_jitter_m: float = 0.08
    miss_spread_m: float = 0.75
    playback_enabled: bool = True
    playback_time_scale: float = 0.0
    playback_fps: float = 12.0
    track_flight_time_s: float = 0.55
    track_persistence_s: float = 1.25
    max_live_tracks: int = 48
    scatter_visual_gain_scale: float = 1.0
    scatter_direction_jitter: float = 0.55
    obstacle_path_prefixes: tuple[str, ...] = ()
    obstacle_focus_fraction: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RadiationVisualizationConfig":
        """Create a visualization config from a JSON-like dictionary."""
        payload = {} if data is None else dict(data)
        obstacle_path_prefixes = payload.get("obstacle_path_prefixes", ())
        if obstacle_path_prefixes is None:
            obstacle_path_prefixes = ()
        elif isinstance(obstacle_path_prefixes, (str, bytes)):
            obstacle_path_prefixes = (str(obstacle_path_prefixes),)
        return cls(
            enabled=bool(payload.get("enabled", True)),
            max_tracks=int(payload.get("max_tracks", 96)),
            max_hits=int(payload.get("max_hits", 24)),
            track_length_m=float(payload.get("track_length_m", 12.0)),
            source_jitter_m=float(payload.get("source_jitter_m", 0.03)),
            detector_jitter_m=float(payload.get("detector_jitter_m", 0.08)),
            miss_spread_m=float(payload.get("miss_spread_m", 0.75)),
            playback_enabled=bool(payload.get("playback_enabled", True)),
            playback_time_scale=float(payload.get("playback_time_scale", 0.0)),
            playback_fps=float(payload.get("playback_fps", 12.0)),
            track_flight_time_s=float(payload.get("track_flight_time_s", 0.55)),
            track_persistence_s=float(payload.get("track_persistence_s", 1.25)),
            max_live_tracks=int(payload.get("max_live_tracks", 48)),
            scatter_visual_gain_scale=float(payload.get("scatter_visual_gain_scale", 1.0)),
            scatter_direction_jitter=float(payload.get("scatter_direction_jitter", 0.55)),
            obstacle_path_prefixes=tuple(str(value) for value in obstacle_path_prefixes),
            obstacle_focus_fraction=float(payload.get("obstacle_focus_fraction", 0.0)),
        )


@dataclass(frozen=True)
class _LineVisualizationRecord:
    """Store one gamma-line contribution used for visual sampling."""

    source_xyz: tuple[float, float, float]
    detector_xyz: tuple[float, float, float]
    isotope: str
    energy_keV: float
    emission_counts: float
    primary_counts: float
    scatter_counts: float
    attenuated_counts: float = 0.0
    scatter_anchor_xyz: tuple[float, float, float] | None = None
    obstacle_volumes: tuple[Any, ...] = ()
    scatter_gain: float = 0.03


@dataclass(frozen=True)
class _ObstacleInteraction:
    """Store one obstacle crossing used to anchor representative scatter."""

    anchor_xyz: tuple[float, float, float]
    path_length_cm: float
    material: Any


def build_visualization_metadata_from_transport(
    transport_results: tuple[SourceTransportResult, ...],
    *,
    seed: int,
    config: RadiationVisualizationConfig,
    mode: str,
) -> dict[str, Any]:
    """Return representative radiation metadata from transport results."""
    records: list[_LineVisualizationRecord] = []
    for result in transport_results:
        source_xyz = tuple(float(value) for value in result.source.position)
        detector_xyz = tuple(float(value) for value in result.detector_position_xyz)
        for line in result.lines:
            records.append(
                _LineVisualizationRecord(
                    source_xyz=source_xyz,
                    detector_xyz=detector_xyz,
                    isotope=str(result.source.isotope),
                    energy_keV=float(line.energy_keV),
                    emission_counts=float(line.emission_counts),
                    primary_counts=float(line.primary_counts),
                    scatter_counts=float(line.scatter_counts),
                    attenuated_counts=float(
                        max(0.0, line.emission_counts - line.primary_counts - line.scatter_counts)
                    ),
                )
            )
    dwell_time_s = max((float(result.dwell_time_s) for result in transport_results), default=0.0)
    return _build_visualization_metadata(
        records,
        seed=seed,
        config=config,
        mode=mode,
        dwell_time_s=dwell_time_s,
    )


def build_visualization_metadata_from_scene(
    scene: Any,
    request: Any,
    *,
    seed: int,
    config: RadiationVisualizationConfig,
    library: dict[str, Any],
    mode: str,
    scatter_gain: float = 0.03,
) -> dict[str, Any]:
    """Return representative radiation metadata from a scene and request."""
    records: list[_LineVisualizationRecord] = []
    detector_xyz = tuple(float(value) for value in request.detector_pose_xyz)
    for source in getattr(scene, "sources", ()):
        point_source = PointSource(
            isotope=str(source.isotope),
            position=tuple(float(value) for value in source.position_xyz),
            intensity_cps_1m=float(source.intensity_cps_1m),
        )
        geometric_scale = float(inverse_square_scale(np.asarray(detector_xyz, dtype=float), point_source))
        base_counts = float(request.dwell_time_s) * float(source.intensity_cps_1m) * geometric_scale
        nuclide = library.get(str(source.isotope))
        if nuclide is None:
            continue
        source_xyz = tuple(float(value) for value in source.position_xyz)
        obstacle_volumes = _obstacle_volumes(scene, config=config)
        obstacle_interactions = _obstacle_interactions_for_volumes(
            obstacle_volumes,
            source_xyz,
            detector_xyz,
        )
        scatter_anchor_xyz = (
            obstacle_interactions[0].anchor_xyz if obstacle_interactions else None
        )
        for line in nuclide.lines:
            emission_counts = base_counts * float(line.intensity)
            stage_transmission = _obstacle_transmission(
                obstacle_interactions,
                isotope=str(source.isotope),
                line_energy_keV=float(line.energy_keV),
            )
            obstacle_path_cm = float(sum(item.path_length_cm for item in obstacle_interactions))
            primary_counts = float(emission_counts * stage_transmission)
            scatter_counts = float(
                primary_counts
                * scatter_scale(
                    path_length_cm=obstacle_path_cm,
                    transmission=stage_transmission,
                    scatter_gain=float(scatter_gain),
                )
            )
            records.append(
                _LineVisualizationRecord(
                    source_xyz=source_xyz,
                    detector_xyz=detector_xyz,
                    isotope=str(source.isotope),
                    energy_keV=float(line.energy_keV),
                    emission_counts=float(emission_counts),
                    primary_counts=primary_counts,
                    scatter_counts=scatter_counts,
                    attenuated_counts=float(max(0.0, emission_counts - primary_counts - scatter_counts)),
                    scatter_anchor_xyz=scatter_anchor_xyz,
                    obstacle_volumes=obstacle_volumes,
                    scatter_gain=float(scatter_gain),
                )
            )
    return _build_visualization_metadata(
        records,
        seed=seed,
        config=config,
        mode=mode,
        dwell_time_s=float(request.dwell_time_s),
    )


def _build_visualization_metadata(
    records: list[_LineVisualizationRecord],
    *,
    seed: int,
    config: RadiationVisualizationConfig,
    mode: str,
    dwell_time_s: float,
) -> dict[str, Any]:
    """Build JSON-serializable radiation tracks and hit points."""
    if not config.enabled or config.max_tracks <= 0 or not records:
        return {
            "radiation_tracks": [],
            "radiation_hits": [],
            "radiation_visualization": {
                "enabled": bool(config.enabled),
                "mode": str(mode),
                "sampled_track_count": 0,
                "sampled_hit_count": 0,
                "playback_enabled": bool(config.playback_enabled),
                "playback_duration_s": max(0.0, float(dwell_time_s)),
            },
        }
    rng = np.random.default_rng(int(seed))
    counts = _allocate_track_counts(records, int(config.max_tracks), rng)
    tracks: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for record, count in zip(records, counts, strict=True):
        for _ in range(int(count)):
            track, hit = _sample_track(record, config, rng, allow_hit=len(hits) < int(config.max_hits))
            tracks.append(track)
            if hit is not None:
                hits.append(hit)
    _assign_track_timing(tracks, config=config, dwell_time_s=dwell_time_s, rng=rng)
    return {
        "radiation_tracks": tracks,
        "radiation_hits": hits,
        "radiation_visualization": {
            "enabled": True,
            "mode": str(mode),
            "sampled_track_count": len(tracks),
            "sampled_hit_count": len(hits),
            "max_tracks": int(config.max_tracks),
            "max_hits": int(config.max_hits),
            "playback_enabled": bool(config.playback_enabled),
            "playback_duration_s": max(0.0, float(dwell_time_s)),
            "playback_time_scale": max(0.0, float(config.playback_time_scale)),
            "playback_fps": max(1.0, float(config.playback_fps)),
            "track_flight_time_s": max(1.0e-3, float(config.track_flight_time_s)),
            "track_persistence_s": max(0.0, float(config.track_persistence_s)),
            "max_live_tracks": max(1, int(config.max_live_tracks)),
        },
    }


def _assign_track_timing(
    tracks: list[dict[str, Any]],
    *,
    config: RadiationVisualizationConfig,
    dwell_time_s: float,
    rng: np.random.Generator,
) -> None:
    """Assign visual emission and lifetime timing to sampled tracks."""
    duration_s = max(0.0, float(dwell_time_s))
    if not tracks or duration_s <= 0.0:
        for track in tracks:
            track["emission_time_s"] = 0.0
            track["flight_time_s"] = max(1.0e-3, float(config.track_flight_time_s))
            track["persistence_s"] = max(0.0, float(config.track_persistence_s))
        return
    emission_times = rng.uniform(0.0, duration_s, size=len(tracks))
    emission_times.sort()
    for track, emission_time_s in zip(tracks, emission_times, strict=True):
        track["emission_time_s"] = float(emission_time_s)
        track["flight_time_s"] = max(1.0e-3, float(config.track_flight_time_s))
        track["persistence_s"] = max(0.0, float(config.track_persistence_s))


def _allocate_track_counts(
    records: list[_LineVisualizationRecord],
    max_tracks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate a finite track budget across gamma-line records."""
    weights = np.asarray(
        [
            max(
                record.primary_counts + record.scatter_counts + record.attenuated_counts,
                record.emission_counts,
                0.0,
            )
            for record in records
        ],
        dtype=float,
    )
    if max_tracks <= 0 or float(np.sum(weights)) <= 0.0:
        return np.zeros(len(records), dtype=int)
    probabilities = weights / float(np.sum(weights))
    return rng.multinomial(int(max_tracks), probabilities)


def _sample_track(
    record: _LineVisualizationRecord,
    config: RadiationVisualizationConfig,
    rng: np.random.Generator,
    *,
    allow_hit: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Sample one representative gamma track and optional detector hit."""
    source = np.asarray(record.source_xyz, dtype=float)
    detector = np.asarray(record.detector_xyz, dtype=float)
    start = source + rng.normal(0.0, float(config.source_jitter_m), size=3)
    detected = _sample_detector_hit(record, rng, allow_hit=allow_hit)
    if detected:
        visual_kind = "detected"
        end = detector + rng.normal(0.0, float(config.detector_jitter_m), size=3)
        points = [start, end]
        scatter_anchor = (
            None
            if record.scatter_anchor_xyz is None
            else np.asarray(record.scatter_anchor_xyz, dtype=float)
        )
        primary_fraction = _fraction(record.primary_counts, record.emission_counts)
        scatter_fraction = _fraction(record.scatter_counts, record.emission_counts)
        attenuated_fraction = _fraction(record.attenuated_counts, record.emission_counts)
    else:
        ray_end = _sample_non_detected_ray_end(source, record.obstacle_volumes, config, rng)
        direction = _unit_vector(ray_end - source)
        length = float(np.linalg.norm(ray_end - source))
        ray_interactions = _obstacle_interactions_for_volumes(
            record.obstacle_volumes,
            tuple(float(value) for value in source),
            tuple(float(value) for value in ray_end),
        )
        (
            visual_kind,
            primary_fraction,
            scatter_fraction,
            attenuated_fraction,
        ) = _sample_ray_visual_kind(
            ray_interactions,
            isotope=record.isotope,
            line_energy_keV=record.energy_keV,
            scatter_gain=record.scatter_gain,
            scatter_visual_gain_scale=config.scatter_visual_gain_scale,
            rng=rng,
        )
        scatter_anchor = None
        if ray_interactions:
            scatter_anchor = np.asarray(ray_interactions[0].anchor_xyz, dtype=float)
        if visual_kind == "scatter" and scatter_anchor is not None:
            mid = scatter_anchor + rng.normal(0.0, 0.06, size=3)
            bend = _unit_vector(
                direction + rng.normal(0.0, max(0.0, float(config.scatter_direction_jitter)), size=3)
            )
            end = mid + bend * rng.uniform(0.45, 1.1) * length
            points = [start, mid, end]
        elif visual_kind == "attenuated" and scatter_anchor is not None:
            end = scatter_anchor + rng.normal(0.0, 0.05, size=3)
            points = [start, end]
        else:
            visual_kind = "primary"
            end = start + direction * length
            points = [start, end]
    track = {
        "points_xyz": [_to_xyz_list(point) for point in points],
        "source_xyz": _to_xyz_list(source),
        "end_xyz": _to_xyz_list(end),
        "isotope": record.isotope,
        "energy_keV": float(record.energy_keV),
        "detected": detected,
        "visual_kind": visual_kind,
        "weight": float(max(record.emission_counts, 0.0)),
        "primary_fraction": primary_fraction,
        "scatter_fraction": scatter_fraction,
        "attenuated_fraction": attenuated_fraction,
    }
    if scatter_anchor is not None:
        track["scatter_anchor_xyz"] = [float(value) for value in scatter_anchor]
    hit = None
    if detected:
        hit = {
            "position_xyz": _to_xyz_list(end),
            "isotope": record.isotope,
            "energy_keV": float(record.energy_keV),
            "weight": float(max(record.primary_counts, 0.0)),
        }
    return track, hit


def _sample_detector_hit(
    record: _LineVisualizationRecord,
    rng: np.random.Generator,
    *,
    allow_hit: bool,
) -> bool:
    """Return whether a representative visual track is a detector hit."""
    if not allow_hit:
        return False
    return bool(rng.random() < _detection_probability(record))


def _detection_probability(record: _LineVisualizationRecord) -> float:
    """Return a bounded visual detection probability for one line record."""
    if record.emission_counts <= 0.0:
        return 0.0
    transmission = float(record.primary_counts) / max(float(record.emission_counts), 1.0e-12)
    return float(np.clip(transmission, 0.0, 0.95))


def _sample_ray_visual_kind(
    interactions: tuple[_ObstacleInteraction, ...],
    *,
    isotope: str,
    line_energy_keV: float,
    scatter_gain: float,
    scatter_visual_gain_scale: float,
    rng: np.random.Generator,
) -> tuple[str, float, float, float]:
    """Sample a visual class for an isotropically emitted ray."""
    if not interactions:
        return "primary", 1.0, 0.0, 0.0
    path_length_cm = float(sum(interaction.path_length_cm for interaction in interactions))
    transmission = _obstacle_transmission(
        interactions,
        isotope=isotope,
        line_energy_keV=line_energy_keV,
    )
    scatter_fraction = float(
        transmission
        * scatter_scale(
            path_length_cm=path_length_cm,
            transmission=transmission,
            scatter_gain=float(scatter_gain),
        )
        * max(0.0, float(scatter_visual_gain_scale))
    )
    primary_fraction = float(np.clip(transmission, 0.0, 1.0))
    scatter_fraction = float(np.clip(scatter_fraction, 0.0, max(0.0, 1.0 - primary_fraction)))
    attenuated_fraction = float(max(0.0, 1.0 - primary_fraction - scatter_fraction))
    roll = rng.random()
    if roll < scatter_fraction:
        return "scatter", primary_fraction, scatter_fraction, attenuated_fraction
    if roll < scatter_fraction + attenuated_fraction:
        return "attenuated", primary_fraction, scatter_fraction, attenuated_fraction
    return "primary", primary_fraction, scatter_fraction, attenuated_fraction


def _fraction(numerator: float, denominator: float) -> float:
    """Return a bounded non-negative ratio."""
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(float(numerator) / max(float(denominator), 1.0e-12), 0.0, 1.0))


def _obstacle_interactions(
    scene: Any,
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
) -> tuple[_ObstacleInteraction, ...]:
    """Return obstacle crossings between source and detector for visualization."""
    return _obstacle_interactions_for_volumes(_obstacle_volumes(scene), source_xyz, detector_xyz)


def _obstacle_volumes(
    scene: Any,
    *,
    config: RadiationVisualizationConfig | None = None,
) -> tuple[Any, ...]:
    """Return obstacle volumes exported in a scene payload."""
    allowed_prefixes = () if config is None else tuple(config.obstacle_path_prefixes)
    return tuple(
        volume
        for volume in getattr(scene, "static_volumes", ())
        if _is_obstacle_volume(scene, volume, allowed_prefixes=allowed_prefixes)
    )


def _obstacle_interactions_for_volumes(
    volumes: tuple[Any, ...],
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
) -> tuple[_ObstacleInteraction, ...]:
    """Return obstacle crossings for a segment through known obstacle volumes."""
    interactions: list[_ObstacleInteraction] = []
    for volume in volumes:
        interaction = _volume_interaction(volume, source_xyz, detector_xyz)
        if interaction is not None:
            interactions.append(interaction)
    interactions.sort(key=lambda item: item.path_length_cm, reverse=True)
    return tuple(interactions)


def _is_obstacle_volume(
    scene: Any,
    volume: Any,
    *,
    allowed_prefixes: tuple[str, ...] = (),
) -> bool:
    """Return whether an exported volume should drive obstacle visualization."""
    prim_paths = getattr(scene, "prim_paths", None)
    obstacles_root = str(getattr(prim_paths, "obstacles_root", "/World/SimBridge/Obstacles"))
    path = str(getattr(volume, "path", ""))
    if allowed_prefixes:
        return any(path.startswith(prefix) for prefix in allowed_prefixes)
    if path.startswith(obstacles_root):
        return True
    if not path.startswith("/World/Environment"):
        return False
    material = getattr(volume, "material", None)
    material_name = "" if material is None else str(getattr(material, "name", "")).strip().lower()
    return material_name not in {"", "air", "vacuum"}


def _volume_interaction(
    volume: Any,
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
) -> _ObstacleInteraction | None:
    """Return one obstacle interaction if the source-detector segment crosses a volume."""
    path_length_m = _volume_path_length_m(volume, source_xyz, detector_xyz)
    if path_length_m <= 0.0:
        return None
    anchor = _volume_intersection_anchor(volume, source_xyz, detector_xyz)
    material = getattr(volume, "material", None) or SimpleNamespace(name="concrete")
    return _ObstacleInteraction(
        anchor_xyz=anchor,
        path_length_cm=100.0 * float(path_length_m),
        material=material,
    )


def _volume_path_length_m(
    volume: Any,
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
) -> float:
    """Return segment path length through an exported visualization obstacle."""
    shape = str(getattr(volume, "shape", ""))
    if shape == "box" and getattr(volume, "size_xyz", None) is not None:
        box = OrientedBox(
            center_xyz=tuple(float(value) for value in getattr(volume, "translation_xyz")),
            size_xyz=tuple(float(value) for value in getattr(volume, "size_xyz")),
            rotation_matrix=quaternion_wxyz_to_matrix(
                tuple(float(value) for value in getattr(volume, "orientation_wxyz", (1.0, 0.0, 0.0, 0.0)))
            ),
        )
        return float(segment_path_length_through_box(source_xyz, detector_xyz, box))
    if shape == "sphere" and getattr(volume, "radius_m", None) is not None:
        sphere = Sphere(
            center_xyz=tuple(float(value) for value in getattr(volume, "translation_xyz")),
            radius_m=float(getattr(volume, "radius_m")),
        )
        return float(segment_path_length_through_sphere(source_xyz, detector_xyz, sphere))
    return 0.0


def _volume_intersection_anchor(
    volume: Any,
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a representative point inside the crossed obstacle volume."""
    if str(getattr(volume, "shape", "")) == "box" and getattr(volume, "size_xyz", None) is not None:
        midpoint = _box_intersection_midpoint(volume, source_xyz, detector_xyz)
        if midpoint is not None:
            return midpoint
    return tuple(float(value) for value in getattr(volume, "translation_xyz", detector_xyz))


def _box_intersection_midpoint(
    volume: Any,
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Return the midpoint of a segment's interval inside an oriented box."""
    center = np.asarray(getattr(volume, "translation_xyz"), dtype=float)
    size = np.asarray(getattr(volume, "size_xyz"), dtype=float)
    rotation = quaternion_wxyz_to_matrix(
        tuple(float(value) for value in getattr(volume, "orientation_wxyz", (1.0, 0.0, 0.0, 0.0)))
    )
    local_start = rotation.T @ (np.asarray(source_xyz, dtype=float) - center)
    local_end = rotation.T @ (np.asarray(detector_xyz, dtype=float) - center)
    delta = local_end - local_start
    half = 0.5 * size
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        if abs(float(delta[axis])) <= 1.0e-12:
            if local_start[axis] < -half[axis] or local_start[axis] > half[axis]:
                return None
            continue
        t0 = (-half[axis] - local_start[axis]) / delta[axis]
        t1 = (half[axis] - local_start[axis]) / delta[axis]
        near = float(min(t0, t1))
        far = float(max(t0, t1))
        t_enter = max(t_enter, near)
        t_exit = min(t_exit, far)
        if t_enter > t_exit:
            return None
    t_mid = 0.5 * (t_enter + t_exit)
    local_mid = local_start + t_mid * delta
    world_mid = center + rotation @ local_mid
    return (float(world_mid[0]), float(world_mid[1]), float(world_mid[2]))


def _obstacle_transmission(
    interactions: tuple[_ObstacleInteraction, ...],
    *,
    isotope: str,
    line_energy_keV: float,
) -> float:
    """Return the visual direct-transmission factor through crossed obstacles."""
    transmission = 1.0
    for interaction in interactions:
        segment = make_transport_segment(interaction.material, interaction.path_length_cm, is_obstacle=True)
        transmission *= material_transmission(
            segment.material,
            isotope,
            float(line_energy_keV),
            interaction.path_length_cm,
        )
    return float(np.clip(transmission, 0.0, 1.0))


def _sample_non_detected_ray_end(
    source: np.ndarray,
    obstacle_volumes: tuple[Any, ...],
    config: RadiationVisualizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the endpoint for a non-detected representative emitted ray."""
    focus_fraction = float(np.clip(config.obstacle_focus_fraction, 0.0, 1.0))
    if obstacle_volumes and focus_fraction > 0.0 and rng.random() < focus_fraction:
        return _sample_obstacle_focused_ray_end(source, obstacle_volumes, config, rng)
    direction = _sample_isotropic_direction(rng)
    length = rng.uniform(0.35, 1.0) * max(float(config.track_length_m), 0.5)
    return source + direction * length


def _sample_obstacle_focused_ray_end(
    source: np.ndarray,
    obstacle_volumes: tuple[Any, ...],
    config: RadiationVisualizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a representative ray that crosses a selected obstacle volume."""
    volume = obstacle_volumes[int(rng.integers(0, len(obstacle_volumes)))]
    anchor = _sample_volume_anchor(volume, rng)
    direction = _unit_vector(anchor - source)
    length = max(float(config.track_length_m), 0.5)
    return anchor + direction * rng.uniform(0.35, 1.0) * length


def _sample_volume_anchor(volume: Any, rng: np.random.Generator) -> np.ndarray:
    """Sample a stable point inside a visualized obstacle volume."""
    shape = str(getattr(volume, "shape", ""))
    center = np.asarray(getattr(volume, "translation_xyz", (0.0, 0.0, 0.0)), dtype=float)
    if shape == "box" and getattr(volume, "size_xyz", None) is not None:
        size = np.asarray(getattr(volume, "size_xyz"), dtype=float)
        rotation = quaternion_wxyz_to_matrix(
            tuple(float(value) for value in getattr(volume, "orientation_wxyz", (1.0, 0.0, 0.0, 0.0)))
        )
        local = rng.uniform(-0.35, 0.35, size=3) * size
        return center + rotation @ local
    if shape == "sphere" and getattr(volume, "radius_m", None) is not None:
        direction = _sample_isotropic_direction(rng)
        radius = float(getattr(volume, "radius_m")) * float(np.cbrt(rng.random()))
        return center + direction * radius
    return center


def _sample_isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    """Sample a unit direction uniformly over the sphere."""
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    radius = np.sqrt(max(0.0, 1.0 - z * z))
    return np.asarray(
        (
            float(radius * np.cos(phi)),
            float(radius * np.sin(phi)),
            float(z),
        ),
        dtype=float,
    )


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, falling back to +X for near-zero input."""
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.asarray((1.0, 0.0, 0.0), dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _to_xyz_list(values: np.ndarray) -> list[float]:
    """Convert a 3-vector into a JSON-serializable float list."""
    return [float(values[0]), float(values[1]), float(values[2])]
