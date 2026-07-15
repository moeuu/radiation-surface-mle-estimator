"""Generate a random environment USD file from a manifest using Blender."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _build_parser() -> argparse.ArgumentParser:
    """Build the Blender-side argument parser."""
    parser = argparse.ArgumentParser(description="Generate a Blender-authored USD environment.")
    parser.add_argument("--input", required=True, help="Input environment manifest JSON.")
    parser.add_argument("--output", required=True, help="Output USD/USDA file path.")
    return parser


def _parse_blender_args() -> argparse.Namespace:
    """Parse arguments after Blender's -- separator."""
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    return _build_parser().parse_args(script_args)


def _load_manifest(path: Path) -> dict:
    """Load the JSON manifest that defines the generated environment."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Environment manifest must be a JSON object.")
    return payload


def _clear_scene() -> None:
    """Remove all objects from the active Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_base_usd(payload: dict) -> bool:
    """Import the optional base USD environment and return whether it was used."""
    base_usd = payload.get("base_usd_path")
    if base_usd in (None, ""):
        return False
    base_path = Path(str(base_usd)).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base USD environment not found: {base_path}")
    bpy.ops.wm.usd_import(filepath=base_path.as_posix())
    return True


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    """Create a simple diffuse material."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    if color[3] < 1.0:
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        if shader is not None and "Alpha" in shader.inputs:
            shader.inputs["Alpha"].default_value = color[3]
        mat.blend_method = "BLEND"
        if hasattr(mat, "show_transparent_back"):
            mat.show_transparent_back = True
    return mat


def _material_library() -> dict[str, bpy.types.Material]:
    """Create reusable materials with names understood by the Geant4 exporter."""
    return {
        "concrete": _material("ConcreteMaterial", (0.55, 0.58, 0.60, 1.0)),
        "steel": _material("SteelMaterial", (0.32, 0.34, 0.36, 1.0)),
        "stainless_steel": _material("StainlessSteelMaterial", (0.52, 0.54, 0.55, 1.0)),
        "aluminum": _material("AluminumMaterial", (0.72, 0.74, 0.76, 1.0)),
        "water": _material("WaterMaterial", (0.25, 0.45, 0.85, 0.55)),
        "air": _material("AirMaterial", (0.88, 0.93, 1.0, 0.15)),
        "ceiling": _material("TransparentCeilingMaterial", (0.88, 0.93, 1.0, 0.04)),
    }


def _resolve_material(
    materials: dict[str, bpy.types.Material],
    material_name: str,
) -> bpy.types.Material:
    """Return a known material, falling back to concrete."""
    key = str(material_name).strip().lower().replace("-", "_").replace(" ", "_")
    return materials.get(key, materials["concrete"])


def _ensure_empty(name: str, *, transport_group: str | None = None) -> bpy.types.Object:
    """Create or reuse an empty object used as a semantic USD group."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(obj)
    if transport_group not in (None, ""):
        obj["simbridge_transport_group"] = str(transport_group)
        obj["simbridge_category"] = str(transport_group)
    return obj


def _add_cube(
    name: str,
    *,
    size_xyz: tuple[float, float, float],
    center_xyz: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object | None = None,
    transport_group: str | None = None,
    material_name: str | None = None,
) -> bpy.types.Object:
    """Add a cube with the requested dimensions and center."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center_xyz)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size_xyz
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    obj["simbridge_material"] = str(material_name or material.name)
    obj["simbridge_material_preset"] = str(material_name or material.name)
    if parent is not None:
        obj.parent = parent
    if transport_group not in (None, ""):
        obj["simbridge_transport_group"] = str(transport_group)
        obj["simbridge_category"] = str(transport_group)
    return obj


def _author_room(
    payload: dict,
    concrete: bpy.types.Material,
    ceiling_material: bpy.types.Material,
) -> None:
    """Create the room shell objects in Blender."""
    size_x, size_y, size_z = (float(value) for value in payload["room_size_xyz"])
    wall_height = max(0.1, size_z)
    wall_thickness = 0.1
    wall_group = _ensure_empty("Wall", transport_group="wall")
    _author_room_boundaries(
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        wall_height=wall_height,
        wall_thickness=wall_thickness,
        wall_group=wall_group,
        concrete=concrete,
        ceiling_material=ceiling_material,
        include_floor=True,
        include_walls=True,
    )


def _author_imported_room_shell(
    payload: dict,
    concrete: bpy.types.Material,
    ceiling_material: bpy.types.Material,
) -> None:
    """Create full-height wall and ceiling boundaries for imported base rooms."""
    size_x, size_y, size_z = (float(value) for value in payload["room_size_xyz"])
    wall_height = max(0.1, size_z)
    wall_thickness = 0.1
    wall_group = _ensure_empty("Wall", transport_group="wall")
    _author_room_boundaries(
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        wall_height=wall_height,
        wall_thickness=wall_thickness,
        wall_group=wall_group,
        concrete=concrete,
        ceiling_material=ceiling_material,
        include_floor=False,
        include_walls=True,
    )


def _author_room_boundaries(
    *,
    size_x: float,
    size_y: float,
    size_z: float,
    wall_height: float,
    wall_thickness: float,
    wall_group: bpy.types.Object,
    concrete: bpy.types.Material,
    ceiling_material: bpy.types.Material,
    include_floor: bool,
    include_walls: bool,
) -> None:
    """Create reusable room boundary meshes."""
    if include_floor:
        _add_cube(
            "Floor",
            size_xyz=(size_x, size_y, 0.1),
            center_xyz=(0.5 * size_x, 0.5 * size_y, -0.05),
            material=concrete,
            parent=wall_group,
            transport_group="wall",
        )
    if include_walls:
        _add_cube(
            "NorthWall",
            size_xyz=(size_x, wall_thickness, wall_height),
            center_xyz=(0.5 * size_x, size_y + 0.5 * wall_thickness, 0.5 * wall_height),
            material=concrete,
            parent=wall_group,
            transport_group="wall",
        )
        _add_cube(
            "SouthWall",
            size_xyz=(size_x, wall_thickness, wall_height),
            center_xyz=(0.5 * size_x, -0.5 * wall_thickness, 0.5 * wall_height),
            material=concrete,
            parent=wall_group,
            transport_group="wall",
        )
        _add_cube(
            "EastWall",
            size_xyz=(wall_thickness, size_y, wall_height),
            center_xyz=(size_x + 0.5 * wall_thickness, 0.5 * size_y, 0.5 * wall_height),
            material=concrete,
            parent=wall_group,
            transport_group="wall",
        )
        _add_cube(
            "WestWall",
            size_xyz=(wall_thickness, size_y, wall_height),
            center_xyz=(-0.5 * wall_thickness, 0.5 * size_y, 0.5 * wall_height),
            material=concrete,
            parent=wall_group,
            transport_group="wall",
        )
    _add_cube(
        "Ceiling",
        size_xyz=(size_x, size_y, wall_thickness),
        center_xyz=(0.5 * size_x, 0.5 * size_y, size_z + 0.5 * wall_thickness),
        material=ceiling_material,
        parent=wall_group,
        transport_group="wall",
        material_name="concrete",
    )


def _author_obstacles(payload: dict, concrete: bpy.types.Material) -> None:
    """Create obstacle box objects in Blender from blocked grid cells."""
    materials = _material_library()
    if payload.get("obstacle_instances"):
        _author_known_obstacle_instances(payload, materials)
        return
    origin_x, origin_y = (float(value) for value in payload["obstacle_origin_xy"])
    cell_size = float(payload["obstacle_cell_size_m"])
    height = float(payload.get("obstacle_height_m", 2.0))
    obstacle_group = _ensure_empty("Obstacles", transport_group="obstacle")
    for index, cell in enumerate(payload.get("obstacle_cells", [])):
        ix, iy = (int(value) for value in cell)
        x0 = origin_x + float(ix) * cell_size
        y0 = origin_y + float(iy) * cell_size
        _add_cube(
            f"Obstacle_{index:04d}",
            size_xyz=(cell_size, cell_size, height),
            center_xyz=(x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, 0.5 * height),
            material=concrete,
            parent=obstacle_group,
            transport_group="obstacle",
            material_name=str(payload.get("obstacle_material", "concrete")),
        )


def _author_known_obstacle_instances(
    payload: dict,
    materials: dict[str, bpy.types.Material],
) -> None:
    """Create composite known obstacle assets from the manifest."""
    obstacle_group = _ensure_empty("Obstacles", transport_group="obstacle")
    for instance_index, instance in enumerate(payload.get("obstacle_instances", [])):
        instance_name = str(instance.get("name", f"KnownObstacle_{instance_index:04d}"))
        instance_group = _ensure_empty(instance_name, transport_group="obstacle")
        instance_group.parent = obstacle_group
        instance_group["simbridge_template"] = str(instance.get("template", "unknown"))
        for component_index, component in enumerate(instance.get("components", [])):
            material_name = str(component.get("material", "concrete"))
            center = tuple(float(value) for value in component["center_xyz"])
            size = tuple(float(value) for value in component["size_xyz"])
            if len(center) != 3 or len(size) != 3:
                raise ValueError("Known obstacle component center_xyz and size_xyz must be 3D.")
            _add_cube(
                str(component.get("name", f"{instance_name}_component_{component_index:02d}")),
                size_xyz=(size[0], size[1], size[2]),
                center_xyz=(center[0], center[1], center[2]),
                material=_resolve_material(materials, material_name),
                parent=instance_group,
                transport_group="obstacle",
                material_name=material_name,
            )


def _export_usd(output_path: Path) -> None:
    """Export the Blender scene to USD under /World/Environment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.usd_export(
            filepath=output_path.as_posix(),
            export_materials=True,
            selected_objects_only=False,
            root_prim_path="/World/Environment",
        )
    except TypeError:
        bpy.ops.wm.usd_export(
            filepath=output_path.as_posix(),
            export_materials=True,
            selected_objects_only=False,
        )


def _object_world_bounds(
    obj: bpy.types.Object,
) -> tuple[float, float, float, float, float, float] | None:
    """Return the world-space axis-aligned bounds for a mesh object."""
    if obj.type != "MESH" or not obj.bound_box:
        return None
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [float(point.x) for point in points]
    ys = [float(point.y) for point in points]
    zs = [float(point.z) for point in points]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _disk_intersects_rect(
    center_xy: tuple[float, float],
    radius_m: float,
    rect: tuple[float, float, float, float],
) -> bool:
    """Return True when a disk intersects an axis-aligned rectangle."""
    x, y = center_xy
    x0, x1, y0, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return dx * dx + dy * dy <= radius_m * radius_m


def _point_to_cell(
    point_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    cell_size: float,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the grid cell containing a point, or None outside the grid."""
    rel_x = float(point_xy[0]) - origin_xy[0]
    rel_y = float(point_xy[1]) - origin_xy[1]
    if rel_x < 0.0 or rel_y < 0.0:
        return None
    ix = int(rel_x // cell_size)
    iy = int(rel_y // cell_size)
    if ix < 0 or iy < 0 or ix >= grid_shape[0] or iy >= grid_shape[1]:
        return None
    return ix, iy


def _reachable_cells(
    free_cells: set[tuple[int, int]],
    start: tuple[int, int],
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    """Return the connected free component reachable from start."""
    if start not in free_cells:
        return set()
    visited = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        ix, iy = queue.popleft()
        for neighbor in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
            if neighbor in visited:
                continue
            if neighbor[0] < 0 or neighbor[1] < 0:
                continue
            if neighbor[0] >= grid_shape[0] or neighbor[1] >= grid_shape[1]:
                continue
            if neighbor not in free_cells:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _write_traversability_map(payload: dict) -> None:
    """Write a 2D robot traversability map from the generated 3D scene."""
    output_value = payload.get("traversability_output_path")
    if output_value in (None, ""):
        return
    output_path = Path(str(output_value)).expanduser().resolve()
    origin_x, origin_y = (float(value) for value in payload["obstacle_origin_xy"])
    cell_size = float(payload.get("traversability_cell_size_m", payload["obstacle_cell_size_m"]))
    nx, ny = (int(value) for value in payload["obstacle_grid_shape"])
    radius = max(0.0, float(payload.get("traversability_robot_radius_m", 0.35)))
    z_min, z_max = (
        float(value)
        for value in payload.get("traversability_blocking_z_range_m", (0.05, 2.0))
    )
    obstacle_rects: list[tuple[float, float, float, float]] = []
    for rect in payload.get("traversability_rects_xy", []):
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError("traversability_rects_xy entries must be [x0, x1, y0, y1].")
        obstacle_rects.append(tuple(float(value) for value in rect))
    for obj in bpy.context.scene.objects:
        bounds = _object_world_bounds(obj)
        if bounds is None:
            continue
        x0, x1, y0, y1, obj_z0, obj_z1 = bounds
        if obj_z1 < z_min or obj_z0 > z_max:
            continue
        obstacle_rects.append((x0, x1, y0, y1))
    free_cells: set[tuple[int, int]] = set()
    for ix in range(nx):
        for iy in range(ny):
            center = (origin_x + (ix + 0.5) * cell_size, origin_y + (iy + 0.5) * cell_size)
            blocked = any(_disk_intersects_rect(center, radius, rect) for rect in obstacle_rects)
            if not blocked:
                free_cells.add((ix, iy))
    reachable = payload.get("traversability_reachable_from_xy")
    if reachable not in (None, ""):
        start = _point_to_cell(
            (float(reachable[0]), float(reachable[1])),
            (origin_x, origin_y),
            cell_size,
            (nx, ny),
        )
        free_cells = set() if start is None else _reachable_cells(free_cells, start, (nx, ny))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_payload = {
        "version": 1,
        "source": "blender_projected_3d_environment",
        "origin": [origin_x, origin_y],
        "cell_size": cell_size,
        "grid_shape": [nx, ny],
        "robot_radius_m": radius,
        "blocking_z_range_m": [z_min, z_max],
        "traversable_fraction": float(len(free_cells)) / float(max(nx * ny, 1)),
        "traversable_cells": [list(cell) for cell in sorted(free_cells)],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(map_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    """Generate the Blender scene and export it to USD."""
    args = _parse_blender_args()
    payload = _load_manifest(Path(args.input).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()
    _clear_scene()
    materials = _material_library()
    concrete = materials["concrete"]
    base_imported = _import_base_usd(payload)
    if not base_imported:
        _author_room(payload, concrete, concrete)
    else:
        _author_imported_room_shell(payload, concrete, concrete)
    _author_obstacles(payload, concrete)
    _write_traversability_map(payload)
    _export_usd(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
