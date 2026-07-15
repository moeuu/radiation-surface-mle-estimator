"""Area-aware surface patches for distribution reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import _blocked_mask_xy, _exposed_side_arrays


@dataclass(frozen=True)
class SurfacePatchDictionary:
    """Describe finite surface patches and their graph-TV adjacency."""

    centers_xyz: NDArray[np.float64]
    areas_m2: NDArray[np.float64]
    kinds: tuple[str, ...]
    face_ids: tuple[str, ...]
    normals_xyz: NDArray[np.float64]
    local_uv_m: NDArray[np.float64]
    adjacency_edges: NDArray[np.int64]
    shared_edge_lengths_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate array shapes and physical patch measures."""
        centers = np.asarray(self.centers_xyz, dtype=float)
        areas = np.asarray(self.areas_m2, dtype=float).reshape(-1)
        normals = np.asarray(self.normals_xyz, dtype=float)
        local_uv = np.asarray(self.local_uv_m, dtype=float)
        edges = np.asarray(self.adjacency_edges, dtype=np.int64)
        edge_lengths = np.asarray(self.shared_edge_lengths_m, dtype=float).reshape(-1)
        count = int(areas.size)
        if centers.shape != (count, 3):
            raise ValueError("centers_xyz must be shaped (C, 3).")
        if normals.shape != (count, 3):
            raise ValueError("normals_xyz must be shaped (C, 3).")
        if local_uv.shape != (count, 2):
            raise ValueError("local_uv_m must be shaped (C, 2).")
        if len(self.kinds) != count or len(self.face_ids) != count:
            raise ValueError("kinds and face_ids must have one entry per patch.")
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("areas_m2 must contain finite positive values.")
        if edges.size == 0:
            edges = np.zeros((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("adjacency_edges must be shaped (E, 2).")
        if edge_lengths.size != edges.shape[0]:
            raise ValueError("shared_edge_lengths_m must have E entries.")
        if edges.size and (np.min(edges) < 0 or np.max(edges) >= count):
            raise ValueError("adjacency_edges contains an invalid patch index.")
        if np.any(~np.isfinite(edge_lengths)) or np.any(edge_lengths <= 0.0):
            raise ValueError("shared edge lengths must be finite and positive.")
        object.__setattr__(self, "centers_xyz", centers)
        object.__setattr__(self, "areas_m2", areas)
        object.__setattr__(self, "normals_xyz", normals)
        object.__setattr__(self, "local_uv_m", local_uv)
        object.__setattr__(self, "adjacency_edges", edges)
        object.__setattr__(self, "shared_edge_lengths_m", edge_lengths)

    @property
    def patch_count(self) -> int:
        """Return the number of finite surface patches."""
        return int(self.areas_m2.size)


@dataclass(frozen=True)
class _PatchBlock:
    """Store one construction block before global index offsets are applied."""

    centers: NDArray[np.float64]
    areas: NDArray[np.float64]
    kinds: NDArray[np.str_]
    face_ids: NDArray[np.str_]
    normals: NDArray[np.float64]
    local_uv: NDArray[np.float64]
    edges: NDArray[np.int64]
    edge_lengths: NDArray[np.float64]


def _axis_edges(length_m: float, target_spacing_m: float) -> NDArray[np.float64]:
    """Return exact interval edges with cells no wider than target spacing."""
    length = float(length_m)
    spacing = float(target_spacing_m)
    if length <= 0.0 or spacing <= 0.0:
        raise ValueError("Surface lengths and spacing must be positive.")
    cells = max(1, int(np.ceil(length / spacing)))
    return np.linspace(0.0, length, num=cells + 1, dtype=float)


def _face_block(
    *,
    origin_xyz: Sequence[float],
    u_axis_xyz: Sequence[float],
    v_axis_xyz: Sequence[float],
    u_edges_m: NDArray[np.float64],
    v_edges_m: NDArray[np.float64],
    normal_xyz: Sequence[float],
    kind: str,
    face_id: str,
) -> _PatchBlock:
    """Build a rectangular face as finite patches with grid adjacency."""
    origin = np.asarray(origin_xyz, dtype=float).reshape(3)
    u_axis = np.asarray(u_axis_xyz, dtype=float).reshape(3)
    v_axis = np.asarray(v_axis_xyz, dtype=float).reshape(3)
    normal = np.asarray(normal_xyz, dtype=float).reshape(3)
    u_edges = np.asarray(u_edges_m, dtype=float).reshape(-1)
    v_edges = np.asarray(v_edges_m, dtype=float).reshape(-1)
    du = np.diff(u_edges)
    dv = np.diff(v_edges)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(u_centers, v_centers, indexing="ij")
    centers = (
        origin[None, :]
        + uu.reshape(-1, 1) * u_axis[None, :]
        + vv.reshape(-1, 1) * v_axis[None, :]
    )
    scale = float(np.linalg.norm(np.cross(u_axis, v_axis)))
    areas = (du[:, None] * dv[None, :] * scale).reshape(-1)
    ids = np.arange(areas.size, dtype=np.int64).reshape(du.size, dv.size)
    edge_parts: list[NDArray[np.int64]] = []
    length_parts: list[NDArray[np.float64]] = []
    if du.size > 1:
        edge_parts.append(np.column_stack([ids[:-1, :].ravel(), ids[1:, :].ravel()]))
        length_parts.append(
            np.broadcast_to(dv[None, :] * np.linalg.norm(v_axis), (du.size - 1, dv.size)).ravel()
        )
    if dv.size > 1:
        edge_parts.append(np.column_stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()]))
        length_parts.append(
            np.broadcast_to(du[:, None] * np.linalg.norm(u_axis), (du.size, dv.size - 1)).ravel()
        )
    edges = (
        np.vstack(edge_parts).astype(np.int64, copy=False)
        if edge_parts
        else np.zeros((0, 2), dtype=np.int64)
    )
    edge_lengths = (
        np.concatenate(length_parts).astype(float, copy=False)
        if length_parts
        else np.zeros(0, dtype=float)
    )
    return _PatchBlock(
        centers=centers,
        areas=areas,
        kinds=np.repeat(np.asarray([str(kind)], dtype=str), areas.size),
        face_ids=np.repeat(np.asarray([str(face_id)], dtype=str), areas.size),
        normals=np.repeat(normal.reshape(1, 3), areas.size, axis=0),
        local_uv=np.column_stack([uu.ravel(), vv.ravel()]),
        edges=edges,
        edge_lengths=edge_lengths,
    )


def _filter_patch_block(block: _PatchBlock, keep_mask: NDArray[np.bool_]) -> _PatchBlock:
    """Filter patches and remap graph edges in batch."""
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if keep.size != block.areas.size:
        raise ValueError("keep_mask must have one entry per patch.")
    remap = np.full(keep.size, -1, dtype=np.int64)
    remap[keep] = np.arange(np.count_nonzero(keep), dtype=np.int64)
    if block.edges.size:
        edge_keep = keep[block.edges[:, 0]] & keep[block.edges[:, 1]]
        edges = remap[block.edges[edge_keep]]
        edge_lengths = block.edge_lengths[edge_keep]
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_lengths = np.zeros(0, dtype=float)
    return _PatchBlock(
        centers=block.centers[keep],
        areas=block.areas[keep],
        kinds=block.kinds[keep],
        face_ids=block.face_ids[keep],
        normals=block.normals[keep],
        local_uv=block.local_uv[keep],
        edges=edges,
        edge_lengths=edge_lengths,
    )


def _repeated_face_blocks(
    *,
    origins_xyz: NDArray[np.float64],
    face_ids: NDArray[np.str_],
    u_axis_xyz: Sequence[float],
    v_axis_xyz: Sequence[float],
    u_edges_m: NDArray[np.float64],
    v_edges_m: NDArray[np.float64],
    normal_xyz: Sequence[float],
    kind: str,
) -> _PatchBlock:
    """Build equal rectangular faces for many obstacle components in batch."""
    origins = np.asarray(origins_xyz, dtype=float).reshape(-1, 3)
    names = np.asarray(face_ids, dtype=str).reshape(-1)
    if origins.shape[0] != names.size:
        raise ValueError("origins_xyz and face_ids must have matching lengths.")
    if origins.shape[0] == 0:
        return _empty_block()
    template = _face_block(
        origin_xyz=(0.0, 0.0, 0.0),
        u_axis_xyz=u_axis_xyz,
        v_axis_xyz=v_axis_xyz,
        u_edges_m=u_edges_m,
        v_edges_m=v_edges_m,
        normal_xyz=normal_xyz,
        kind=kind,
        face_id="template",
    )
    face_count = int(origins.shape[0])
    patch_count = int(template.areas.size)
    centers = (
        origins[:, None, :] + template.centers[None, :, :]
    ).reshape(-1, 3)
    offsets = np.arange(face_count, dtype=np.int64) * patch_count
    edges = (
        template.edges[None, :, :] + offsets[:, None, None]
    ).reshape(-1, 2)
    return _PatchBlock(
        centers=centers,
        areas=np.tile(template.areas, face_count),
        kinds=np.repeat(
            np.asarray([str(kind)], dtype=str),
            face_count * patch_count,
        ),
        face_ids=np.repeat(names, patch_count),
        normals=np.tile(template.normals, (face_count, 1)),
        local_uv=np.tile(template.local_uv, (face_count, 1)),
        edges=edges.astype(np.int64, copy=False),
        edge_lengths=np.tile(template.edge_lengths, face_count),
    )


def _empty_block() -> _PatchBlock:
    """Return an empty patch-construction block."""
    return _PatchBlock(
        centers=np.zeros((0, 3), dtype=float),
        areas=np.zeros(0, dtype=float),
        kinds=np.zeros(0, dtype=str),
        face_ids=np.zeros(0, dtype=str),
        normals=np.zeros((0, 3), dtype=float),
        local_uv=np.zeros((0, 2), dtype=float),
        edges=np.zeros((0, 2), dtype=np.int64),
        edge_lengths=np.zeros(0, dtype=float),
    )


def _combine_patch_blocks(blocks: Sequence[_PatchBlock]) -> SurfacePatchDictionary:
    """Combine construction blocks into one globally indexed dictionary."""
    active = [block for block in blocks if block.areas.size]
    if not active:
        raise ValueError("Surface patch dictionary is empty.")
    offsets = np.cumsum([0, *[int(block.areas.size) for block in active[:-1]]])
    edges = [
        block.edges + int(offset)
        for block, offset in zip(active, offsets)
        if block.edges.size
    ]
    return SurfacePatchDictionary(
        centers_xyz=np.vstack([block.centers for block in active]),
        areas_m2=np.concatenate([block.areas for block in active]),
        kinds=tuple(np.concatenate([block.kinds for block in active]).tolist()),
        face_ids=tuple(np.concatenate([block.face_ids for block in active]).tolist()),
        normals_xyz=np.vstack([block.normals for block in active]),
        local_uv_m=np.vstack([block.local_uv for block in active]),
        adjacency_edges=(
            np.vstack(edges).astype(np.int64, copy=False)
            if edges
            else np.zeros((0, 2), dtype=np.int64)
        ),
        shared_edge_lengths_m=np.concatenate(
            [block.edge_lengths for block in active if block.edges.size]
        )
        if edges
        else np.zeros(0, dtype=float),
    )


def build_surface_patch_dictionary(
    env: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None,
    spacing: float | Sequence[float],
    *,
    obstacle_height_m: float = 2.0,
) -> SurfacePatchDictionary:
    """Build area-aware room and obstacle patches with face-local adjacency."""
    spacing_arr = np.asarray(spacing, dtype=float).reshape(-1)
    if spacing_arr.size == 1:
        spacing_arr = np.repeat(spacing_arr, 3)
    if spacing_arr.shape != (3,) or np.any(spacing_arr <= 0.0):
        raise ValueError("spacing must be a positive scalar or 3-vector.")
    x_edges = _axis_edges(float(env.size_x), float(spacing_arr[0]))
    y_edges = _axis_edges(float(env.size_y), float(spacing_arr[1]))
    z_edges = _axis_edges(float(env.size_z), float(spacing_arr[2]))
    floor = _face_block(
        origin_xyz=(0.0, 0.0, 0.0),
        u_axis_xyz=(1.0, 0.0, 0.0),
        v_axis_xyz=(0.0, 1.0, 0.0),
        u_edges_m=x_edges,
        v_edges_m=y_edges,
        normal_xyz=(0.0, 0.0, 1.0),
        kind="floor",
        face_id="room_floor",
    )
    if obstacle_grid is not None and obstacle_grid.blocked_cells:
        floor = _filter_patch_block(
            floor,
            ~_blocked_mask_xy(floor.centers[:, :2], obstacle_grid),
        )
    blocks: list[_PatchBlock] = [
        floor,
        _face_block(
            origin_xyz=(0.0, 0.0, float(env.size_z)),
            u_axis_xyz=(1.0, 0.0, 0.0),
            v_axis_xyz=(0.0, 1.0, 0.0),
            u_edges_m=x_edges,
            v_edges_m=y_edges,
            normal_xyz=(0.0, 0.0, -1.0),
            kind="ceiling",
            face_id="room_ceiling",
        ),
        _face_block(
            origin_xyz=(0.0, 0.0, 0.0),
            u_axis_xyz=(0.0, 1.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=y_edges,
            v_edges_m=z_edges,
            normal_xyz=(1.0, 0.0, 0.0),
            kind="wall",
            face_id="room_wall_x0",
        ),
        _face_block(
            origin_xyz=(float(env.size_x), 0.0, 0.0),
            u_axis_xyz=(0.0, 1.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=y_edges,
            v_edges_m=z_edges,
            normal_xyz=(-1.0, 0.0, 0.0),
            kind="wall",
            face_id="room_wall_x1",
        ),
        _face_block(
            origin_xyz=(0.0, 0.0, 0.0),
            u_axis_xyz=(1.0, 0.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=x_edges,
            v_edges_m=z_edges,
            normal_xyz=(0.0, 1.0, 0.0),
            kind="wall",
            face_id="room_wall_y0",
        ),
        _face_block(
            origin_xyz=(0.0, float(env.size_y), 0.0),
            u_axis_xyz=(1.0, 0.0, 0.0),
            v_axis_xyz=(0.0, 0.0, 1.0),
            u_edges_m=x_edges,
            v_edges_m=z_edges,
            normal_xyz=(0.0, -1.0, 0.0),
            kind="wall",
            face_id="room_wall_y1",
        ),
    ]
    if obstacle_grid is not None and obstacle_grid.blocked_cells:
        cells = np.asarray(obstacle_grid.blocked_cells, dtype=np.int64).reshape(-1, 2)
        cell_size = float(obstacle_grid.cell_size)
        obstacle_height = min(
            max(float(obstacle_height_m), 0.0),
            float(env.size_z),
        )
        origins_xy = (
            np.asarray(obstacle_grid.origin, dtype=float)[None, :]
            + cells.astype(float) * cell_size
        )
        top_origins = np.column_stack(
            [origins_xy, np.full(cells.shape[0], obstacle_height, dtype=float)]
        )
        top_ids = np.char.add(
            np.char.add("obstacle_top_", cells[:, 0].astype(str)),
            np.char.add("_", cells[:, 1].astype(str)),
        )
        blocks.append(
            _repeated_face_blocks(
                origins_xyz=top_origins,
                face_ids=top_ids,
                u_axis_xyz=(1.0, 0.0, 0.0),
                v_axis_xyz=(0.0, 1.0, 0.0),
                u_edges_m=_axis_edges(cell_size, float(spacing_arr[0])),
                v_edges_m=_axis_edges(cell_size, float(spacing_arr[1])),
                normal_xyz=(0.0, 0.0, 1.0),
                kind="obstacle_top",
            )
        )
        side_cells, side_ids = _exposed_side_arrays(obstacle_grid)
        side_names = np.asarray(["west", "east", "south", "north"], dtype=str)
        for side_id in range(4):
            selected = side_ids == side_id
            if not np.any(selected):
                continue
            selected_cells = side_cells[selected]
            selected_xy = (
                np.asarray(obstacle_grid.origin, dtype=float)[None, :]
                + selected_cells.astype(float) * cell_size
            )
            if side_id == 0:
                origins = np.column_stack(
                    [selected_xy[:, 0], selected_xy[:, 1], np.zeros(selected_xy.shape[0])]
                )
                u_axis = (0.0, 1.0, 0.0)
                normal = (-1.0, 0.0, 0.0)
                horizontal_spacing = float(spacing_arr[1])
            elif side_id == 1:
                origins = np.column_stack(
                    [selected_xy[:, 0] + cell_size, selected_xy[:, 1], np.zeros(selected_xy.shape[0])]
                )
                u_axis = (0.0, 1.0, 0.0)
                normal = (1.0, 0.0, 0.0)
                horizontal_spacing = float(spacing_arr[1])
            elif side_id == 2:
                origins = np.column_stack(
                    [selected_xy[:, 0], selected_xy[:, 1], np.zeros(selected_xy.shape[0])]
                )
                u_axis = (1.0, 0.0, 0.0)
                normal = (0.0, -1.0, 0.0)
                horizontal_spacing = float(spacing_arr[0])
            else:
                origins = np.column_stack(
                    [selected_xy[:, 0], selected_xy[:, 1] + cell_size, np.zeros(selected_xy.shape[0])]
                )
                u_axis = (1.0, 0.0, 0.0)
                normal = (0.0, 1.0, 0.0)
                horizontal_spacing = float(spacing_arr[0])
            prefix = np.char.add(
                np.char.add("obstacle_side_", side_names[side_id]),
                "_",
            )
            side_face_ids = np.char.add(
                np.char.add(prefix, selected_cells[:, 0].astype(str)),
                np.char.add("_", selected_cells[:, 1].astype(str)),
            )
            blocks.append(
                _repeated_face_blocks(
                    origins_xyz=origins,
                    face_ids=side_face_ids,
                    u_axis_xyz=u_axis,
                    v_axis_xyz=(0.0, 0.0, 1.0),
                    u_edges_m=_axis_edges(cell_size, horizontal_spacing),
                    v_edges_m=_axis_edges(
                        max(obstacle_height, 1.0e-12),
                        float(spacing_arr[2]),
                    ),
                    normal_xyz=normal,
                    kind="obstacle_side",
                )
            )
    return _combine_patch_blocks(blocks)
