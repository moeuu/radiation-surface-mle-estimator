"""Author lightweight radiation visualization prims into an Isaac Sim stage."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sim.isaacsim_app.stage_backend import StageBackend
from sim.protocol import SimulationObservation


GEANT4_PRIMARY_GREEN = (0.0, 1.0, 0.0)
GEANT4_DETECTED_YELLOW = (1.0, 0.95, 0.0)
GEANT4_SCATTER_MAGENTA = (0.85, 0.0, 1.0)
GEANT4_ATTENUATED_GRAY = (0.48, 0.48, 0.48)


class RadiationSceneVisualizer:
    """Render sampled radiation tracks and detector hits in a stage."""

    def __init__(
        self,
        stage_backend: StageBackend,
        *,
        root_path: str = "/World/SimBridge/Radiation",
    ) -> None:
        """Store the backend and generated prim root."""
        self.stage_backend = stage_backend
        self.root_path = str(root_path)
        self.tracks_root = f"{self.root_path}/Tracks"
        self.hits_root = f"{self.root_path}/Hits"
        self.live_root = f"{self.root_path}/Live"
        self.live_tracks_root = f"{self.live_root}/Tracks"
        self.live_particles_root = f"{self.live_root}/Particles"

    def update_from_observation(self, observation: SimulationObservation) -> None:
        """Replace radiation visualization prims using observation metadata."""
        metadata = dict(observation.metadata)
        if "radiation_tracks" not in metadata and "radiation_hits" not in metadata:
            return
        tracks = _coerce_list(metadata.get("radiation_tracks", []))
        hits = _coerce_list(metadata.get("radiation_hits", []))
        visualization = _coerce_dict(metadata.get("radiation_visualization", {}))
        self.stage_backend.remove_prim(self.root_path)
        self.stage_backend.ensure_xform(self.root_path)
        self.stage_backend.ensure_xform(self.tracks_root)
        self.stage_backend.ensure_xform(self.hits_root)
        if _playback_enabled(visualization, tracks):
            self._play_tracks(tracks, visualization)
            self.stage_backend.remove_prim(self.live_root)
        self._author_tracks(tracks)
        self._author_hits(hits)
        self._author_detector_pulse(observation)
        self.stage_backend.step()

    def _play_tracks(self, tracks: list[Any], visualization: dict[str, Any]) -> None:
        """Animate sampled radiation tracks over the configured measurement time."""
        duration_s = max(0.0, float(visualization.get("playback_duration_s", 0.0)))
        fps = float(np.clip(float(visualization.get("playback_fps", 12.0)), 1.0, 60.0))
        time_scale = max(0.0, float(visualization.get("playback_time_scale", 0.0)))
        max_live_tracks = max(1, int(visualization.get("max_live_tracks", 48)))
        frame_count = max(1, int(np.ceil(duration_s * fps)))
        frame_dt_s = duration_s / float(frame_count)
        for frame_index in range(frame_count + 1):
            sim_time_s = min(duration_s, float(frame_index) * frame_dt_s)
            active_tracks = _active_tracks(
                tracks,
                sim_time_s=sim_time_s,
                max_live_tracks=max_live_tracks,
            )
            self.stage_backend.remove_prim(self.live_root)
            self.stage_backend.ensure_xform(self.live_root)
            self.stage_backend.ensure_xform(self.live_tracks_root)
            self.stage_backend.ensure_xform(self.live_particles_root)
            self._author_live_tracks(active_tracks, sim_time_s)
            self.stage_backend.step()
            if time_scale > 0.0 and frame_index < frame_count:
                time.sleep(frame_dt_s * time_scale)

    def _author_tracks(self, tracks: list[Any]) -> None:
        """Author sampled gamma tracks as thin stage curves."""
        for index, payload in enumerate(tracks):
            if not isinstance(payload, dict):
                continue
            points = _points_from_track(payload)
            if len(points) < 2:
                continue
            color = _track_color(payload)
            width_m = _track_width(payload, detected_width=0.032, base_width=0.02)
            self.stage_backend.ensure_polyline(
                f"{self.tracks_root}/Track_{index:04d}",
                points_xyz=tuple(points),
                color_rgb=color,
                width_m=width_m,
            )

    def _author_live_tracks(self, tracks: list[dict[str, Any]], sim_time_s: float) -> None:
        """Author active partial tracks and moving particle markers."""
        for index, payload in enumerate(tracks):
            points = _points_from_track(payload)
            if len(points) < 2:
                continue
            partial_points, particle_position = _partial_track(points, payload, sim_time_s)
            if len(partial_points) < 2:
                continue
            isotope = str(payload.get("isotope", ""))
            detected = bool(payload.get("detected", False))
            color = _track_color(payload)
            self.stage_backend.ensure_polyline(
                f"{self.live_tracks_root}/Track_{index:04d}",
                points_xyz=tuple(partial_points),
                color_rgb=color,
                width_m=_track_width(payload, detected_width=0.035, base_width=0.025),
            )
            self.stage_backend.ensure_sphere(
                f"{self.live_particles_root}/Particle_{index:04d}",
                radius_m=0.035 if detected else 0.026,
                translation_xyz=particle_position,
                color_rgb=_hit_color(isotope),
                material="air",
            )

    def _author_hits(self, hits: list[Any]) -> None:
        """Author sampled detector hit markers."""
        for index, payload in enumerate(hits):
            if not isinstance(payload, dict):
                continue
            position = payload.get("position_xyz")
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                continue
            isotope = str(payload.get("isotope", ""))
            self.stage_backend.ensure_sphere(
                f"{self.hits_root}/Hit_{index:04d}",
                radius_m=0.045,
                translation_xyz=tuple(float(value) for value in position),
                color_rgb=_hit_color(isotope),
                material="air",
            )

    def _author_detector_pulse(self, observation: SimulationObservation) -> None:
        """Author a compact pulse marker scaled by total measured counts."""
        total_counts = float(np.sum(np.asarray(observation.spectrum_counts, dtype=float)))
        radius_m = float(np.clip(0.06 + 0.018 * np.log10(total_counts + 1.0), 0.06, 0.18))
        self.stage_backend.ensure_sphere(
            f"{self.root_path}/DetectorPulse",
            radius_m=radius_m,
            translation_xyz=tuple(float(value) for value in observation.detector_pose_xyz),
            color_rgb=(0.0, 0.9, 1.0),
            material="air",
        )


def _coerce_list(value: Any) -> list[Any]:
    """Return a list payload or an empty list for malformed metadata."""
    return list(value) if isinstance(value, list) else []


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary payload or an empty dictionary for malformed metadata."""
    return dict(value) if isinstance(value, dict) else {}


def _playback_enabled(visualization: dict[str, Any], tracks: list[Any]) -> bool:
    """Return whether the observation requests live playback."""
    if not tracks or not bool(visualization.get("playback_enabled", False)):
        return False
    duration_s = float(visualization.get("playback_duration_s", 0.0))
    return duration_s > 0.0


def _active_tracks(
    tracks: list[Any],
    *,
    sim_time_s: float,
    max_live_tracks: int,
) -> list[dict[str, Any]]:
    """Return tracks that should be visible at a playback timestamp."""
    active: list[dict[str, Any]] = []
    for payload in tracks:
        if not isinstance(payload, dict):
            continue
        emission_time_s = float(payload.get("emission_time_s", 0.0))
        flight_time_s = max(1.0e-3, float(payload.get("flight_time_s", 0.55)))
        persistence_s = max(0.0, float(payload.get("persistence_s", 1.25)))
        if emission_time_s <= sim_time_s <= emission_time_s + flight_time_s + persistence_s:
            active.append(payload)
    if len(active) <= max_live_tracks:
        return active
    return active[-max_live_tracks:]


def _points_from_track(payload: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Extract a polyline point list from a track payload."""
    raw_points = payload.get("points_xyz")
    if not isinstance(raw_points, list):
        source = payload.get("source_xyz")
        end = payload.get("end_xyz")
        raw_points = [source, end]
    points: list[tuple[float, float, float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            continue
        points.append(tuple(float(value) for value in point))
    return points


def _partial_track(
    points: list[tuple[float, float, float]],
    payload: dict[str, Any],
    sim_time_s: float,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    """Return the currently visible prefix of a moving track."""
    emission_time_s = float(payload.get("emission_time_s", 0.0))
    flight_time_s = max(1.0e-3, float(payload.get("flight_time_s", 0.55)))
    progress = float(np.clip((sim_time_s - emission_time_s) / flight_time_s, 0.0, 1.0))
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)
    particle = start + progress * (end - start)
    partial = [points[0], (float(particle[0]), float(particle[1]), float(particle[2]))]
    if progress >= 1.0:
        partial = points
    return partial, (float(particle[0]), float(particle[1]), float(particle[2]))


def _track_color(payload: dict[str, Any]) -> tuple[float, float, float]:
    """Return a Geant4-like trajectory color."""
    visual_kind = str(payload.get("visual_kind", ""))
    if bool(payload.get("detected", False)) or visual_kind == "detected":
        return GEANT4_DETECTED_YELLOW
    if visual_kind == "scatter":
        return GEANT4_SCATTER_MAGENTA
    if visual_kind == "attenuated":
        return GEANT4_ATTENUATED_GRAY
    return GEANT4_PRIMARY_GREEN


def _track_width(
    payload: dict[str, Any],
    *,
    detected_width: float,
    base_width: float,
) -> float:
    """Return the display width for one trajectory."""
    visual_kind = str(payload.get("visual_kind", ""))
    if bool(payload.get("detected", False)) or visual_kind == "detected":
        return float(detected_width)
    if visual_kind == "attenuated":
        return float(base_width * 1.3)
    if visual_kind == "scatter":
        return float(base_width * 1.15)
    return float(base_width)


def _hit_color(isotope: str) -> tuple[float, float, float]:
    """Return a bright hit marker color for an isotope."""
    return GEANT4_DETECTED_YELLOW
