"""File-based I/O helpers for the external Geant4 executable."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sim.geant4_app.scene_export import ExportedGeant4Scene

if TYPE_CHECKING:
    from sim.geant4_app.engine import Geant4StepRequest


def write_scene_file(scene: ExportedGeant4Scene, path: str | Path) -> None:
    """Write an exported Geant4 scene into a line-oriented text format."""
    output_path = Path(path)
    lines: list[str] = [
        _format_line(
            "SCENE",
            scene_hash=scene.scene_hash,
            usd_path=scene.usd_path or "-",
            room_x=scene.room_size_xyz[0],
            room_y=scene.room_size_xyz[1],
            room_z=scene.room_size_xyz[2],
        ),
        _format_line(
            "PRIM_PATHS",
            world_root=scene.prim_paths.world_root,
            generated_root=scene.prim_paths.generated_root,
            obstacles_root=scene.prim_paths.obstacles_root,
            sources_root=scene.prim_paths.sources_root,
            robot_root=scene.prim_paths.robot_root,
            detector_path=scene.prim_paths.detector_path,
            fe_shield_path=scene.prim_paths.fe_shield_path,
            pb_shield_path=scene.prim_paths.pb_shield_path,
        ),
        _format_line(
            "DETECTOR",
            crystal_radius_m=scene.detector_model.crystal_radius_m,
            crystal_length_m=scene.detector_model.crystal_length_m,
            housing_thickness_m=scene.detector_model.housing_thickness_m,
            crystal_shape=scene.detector_model.crystal_shape,
            crystal_material=scene.detector_model.crystal_material,
            housing_material=scene.detector_model.housing_material,
        ),
    ]
    for shield_name, shield in (("fe", scene.fe_shield), ("pb", scene.pb_shield)):
        lines.append(
            _format_line(
                "SHIELD",
                kind=shield_name,
                path=shield.path,
                shape=shield.shape,
                inner_radius_m=shield.inner_radius_m,
                outer_radius_m=shield.outer_radius_m,
                thickness_m=shield.thickness_cm / 100.0,
                thickness_cm=shield.thickness_cm,
                use_angle_attenuation=int(bool(shield.use_angle_attenuation)),
                sx="-" if shield.size_xyz is None else shield.size_xyz[0],
                sy="-" if shield.size_xyz is None else shield.size_xyz[1],
                sz="-" if shield.size_xyz is None else shield.size_xyz[2],
                material_name=shield.material.name,
                density_g_cm3=shield.material.density_g_cm3 if shield.material.density_g_cm3 is not None else "-",
                preset_name=shield.material.preset_name or "-",
            )
        )
        lines.extend(_material_detail_lines(shield.path, shield.material))
    for source in scene.sources:
        lines.append(
            _format_line(
                "SOURCE",
                isotope=source.isotope,
                x=source.position_xyz[0],
                y=source.position_xyz[1],
                z=source.position_xyz[2],
                intensity_cps_1m=source.intensity_cps_1m,
            )
        )
    for volume in scene.static_volumes:
        material_name = "-" if volume.material is None else volume.material.name
        density_g_cm3 = "-"
        preset_name = "-"
        if volume.material is not None:
            density_g_cm3 = volume.material.density_g_cm3 if volume.material.density_g_cm3 is not None else "-"
            preset_name = volume.material.preset_name or "-"
        lines.append(
            _format_line(
                "VOLUME",
                path=volume.path,
                shape=volume.shape,
                tx=volume.translation_xyz[0],
                ty=volume.translation_xyz[1],
                tz=volume.translation_xyz[2],
                qw=volume.orientation_wxyz[0],
                qx=volume.orientation_wxyz[1],
                qy=volume.orientation_wxyz[2],
                qz=volume.orientation_wxyz[3],
                material_name=material_name,
                density_g_cm3=density_g_cm3,
                preset_name=preset_name,
                transport_group=volume.transport_group or "-",
                transport_mode=volume.transport_mode,
                sx="-" if volume.size_xyz is None else volume.size_xyz[0],
                sy="-" if volume.size_xyz is None else volume.size_xyz[1],
                sz="-" if volume.size_xyz is None else volume.size_xyz[2],
                radius_m="-" if volume.radius_m is None else volume.radius_m,
            )
        )
        if volume.material is not None:
            lines.extend(_material_detail_lines(volume.path, volume.material))
        if volume.triangles_xyz is not None:
            for triangle in volume.triangles_xyz:
                lines.append(
                    _format_line(
                        "TRI",
                        path=volume.path,
                        ax=triangle[0][0],
                        ay=triangle[0][1],
                        az=triangle[0][2],
                        bx=triangle[1][0],
                        by=triangle[1][1],
                        bz=triangle[1][2],
                        cx=triangle[2][0],
                        cy=triangle[2][1],
                        cz=triangle[2][2],
                    )
                )
    output_path.write_text("".join(lines), encoding="utf-8")


def write_request_file(request: Geant4StepRequest, path: str | Path) -> None:
    """Write one Geant4 step request into a line-oriented text format."""
    output_path = Path(path)
    lines = [
        _format_line(
            "STEP",
            step_id=request.step_id,
            dwell_time_s=request.dwell_time_s,
            seed=request.seed,
        ),
        _format_line(
            "POSE",
            kind="detector",
            x=request.detector_pose_xyz[0],
            y=request.detector_pose_xyz[1],
            z=request.detector_pose_xyz[2],
            qw=request.detector_quat_wxyz[0],
            qx=request.detector_quat_wxyz[1],
            qy=request.detector_quat_wxyz[2],
            qz=request.detector_quat_wxyz[3],
        ),
        _format_line(
            "POSE",
            kind="fe",
            x=request.fe_shield_pose_xyz[0],
            y=request.fe_shield_pose_xyz[1],
            z=request.fe_shield_pose_xyz[2],
            qw=request.fe_shield_quat_wxyz[0],
            qx=request.fe_shield_quat_wxyz[1],
            qy=request.fe_shield_quat_wxyz[2],
            qz=request.fe_shield_quat_wxyz[3],
        ),
        _format_line(
            "POSE",
            kind="pb",
            x=request.pb_shield_pose_xyz[0],
            y=request.pb_shield_pose_xyz[1],
            z=request.pb_shield_pose_xyz[2],
            qw=request.pb_shield_quat_wxyz[0],
            qx=request.pb_shield_quat_wxyz[1],
            qy=request.pb_shield_quat_wxyz[2],
            qz=request.pb_shield_quat_wxyz[3],
        ),
    ]
    output_path.write_text("".join(lines), encoding="utf-8")


def read_response_file(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a line-oriented response file written by the external executable."""
    response_path = Path(path)
    spectrum: np.ndarray | None = None
    spectrum_variance: np.ndarray | None = None
    metadata: dict[str, Any] = {}
    for line in response_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("META "):
            fields = _parse_key_values(stripped.split()[1:])
            for key, value in fields.items():
                metadata[str(key)] = _coerce_value(value)
            continue
        if stripped.startswith("SPECTRUM "):
            _, payload = stripped.split(" ", 1)
            if payload.strip():
                spectrum = np.asarray([float(part) for part in payload.split(",") if part], dtype=float)
            else:
                spectrum = np.zeros(0, dtype=float)
            continue
        if stripped.startswith("SPECTRUM_VARIANCE "):
            _, payload = stripped.split(" ", 1)
            if payload.strip():
                spectrum_variance = np.asarray(
                    [float(part) for part in payload.split(",") if part],
                    dtype=float,
                )
            else:
                spectrum_variance = np.zeros(0, dtype=float)
    if spectrum is None:
        raise RuntimeError("External Geant4 response did not contain a SPECTRUM record.")
    if spectrum_variance is not None:
        metadata["spectrum_count_variance"] = spectrum_variance.tolist()
        metadata["spectrum_count_variance_total"] = float(np.sum(spectrum_variance))
    return spectrum, metadata


def _material_detail_lines(path: str, material: object) -> list[str]:
    """Return exported material-detail records for one material."""
    lines: list[str] = []
    mu_by_isotope = getattr(material, "mu_by_isotope", {})
    for isotope, value in sorted(mu_by_isotope.items()):
        lines.append(_format_line("MU", path=path, isotope=isotope, value=value))
    mass_att = getattr(material, "mass_att_by_isotope_cm2_g", {})
    for isotope, value in sorted(mass_att.items()):
        lines.append(_format_line("MASS_ATT", path=path, isotope=isotope, value=value))
    composition = getattr(material, "composition_by_mass", {})
    for element, fraction in sorted(composition.items()):
        lines.append(_format_line("COMP", path=path, element=element, fraction=fraction))
    return lines


def _format_line(record_type: str, **fields: Any) -> str:
    """Format one line-oriented record."""
    items = [record_type]
    for key, value in fields.items():
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        else:
            encoded = str(value)
        encoded = encoded.replace(" ", "%20")
        items.append(f"{key}={encoded}")
    return " ".join(items) + "\n"


def _parse_key_values(parts: list[str]) -> dict[str, str]:
    """Parse `key=value` tokens from a line-oriented record."""
    result: dict[str, str] = {}
    for token in parts:
        key, value = token.split("=", 1)
        result[key] = value.replace("%20", " ")
    return result


def _coerce_value(raw_value: str) -> float | int | str | bool:
    """Coerce a string token back into a Python scalar."""
    normalized = str(raw_value)
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    try:
        integer_value = int(normalized)
        if str(integer_value) == normalized:
            return integer_value
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        return normalized
