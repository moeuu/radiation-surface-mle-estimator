"""Generate candidate measurement poses for online exploration."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _resolve_bounds(bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (lo, hi) bounds for candidate generation."""
    if bounds_xyz is None:
        lo = np.array([0.0, 0.0, 0.0], dtype=float)
        hi = np.array([10.0, 10.0, 10.0], dtype=float)
    else:
        lo = np.asarray(bounds_xyz[0], dtype=float)
        hi = np.asarray(bounds_xyz[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bounds_xyz must contain two (3,) arrays.")
    hi = np.maximum(hi, lo)
    return lo, hi


def _resolve_free_space_checker(map_api: object | None) -> Callable[[NDArray[np.float64]], bool]:
    """Return a callable that checks whether a point is in free space."""
    if map_api is None:
        return lambda _: True
    if callable(map_api):
        return map_api
    for attr in ("is_free", "is_free_space", "is_free_cell"):
        fn = getattr(map_api, attr, None)
        if callable(fn):
            return fn
    return lambda _: True


def _cell_center_from_map(
    map_api: object,
    cell: tuple[int, int],
    z_value: float,
) -> NDArray[np.float64]:
    """Return a 3D cell-center point for a map cell."""
    center_fn = getattr(map_api, "cell_center", None)
    if callable(center_fn):
        x_value, y_value = center_fn(cell)
        return np.array([float(x_value), float(y_value), float(z_value)], dtype=float)
    origin = getattr(map_api, "origin", (0.0, 0.0))
    cell_size = float(getattr(map_api, "cell_size", 1.0))
    return np.array(
        [
            float(origin[0]) + (float(cell[0]) + 0.5) * cell_size,
            float(origin[1]) + (float(cell[1]) + 0.5) * cell_size,
            float(z_value),
        ],
        dtype=float,
    )


def _map_free_cell_centers(
    map_api: object | None,
    *,
    z_value: float,
) -> NDArray[np.float64]:
    """Return deterministic free-cell centers from the planning map."""
    if map_api is None:
        return np.zeros((0, 3), dtype=float)
    grid_shape = getattr(map_api, "grid_shape", None)
    if grid_shape is None:
        return np.zeros((0, 3), dtype=float)
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is not None:
        cells = [tuple(cell) for cell in traversable_cells]
    else:
        is_free_cell = getattr(map_api, "is_free_cell", None)
        if not callable(is_free_cell):
            is_free_cell = getattr(map_api, "is_cell_free", None)
        if not callable(is_free_cell):
            return np.zeros((0, 3), dtype=float)
        cells = [
            (ix, iy)
            for ix in range(int(grid_shape[0]))
            for iy in range(int(grid_shape[1]))
            if bool(is_free_cell((ix, iy)))
        ]
    if not cells:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(
        [_cell_center_from_map(map_api, tuple(cell), z_value) for cell in cells]
    ).astype(float)


def _filter_candidates(
    candidates: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    min_dist_from_visited: float,
    is_free_fn: Callable[[NDArray[np.float64]], bool],
) -> NDArray[np.float64]:
    """Filter candidate poses by minimum distance and free-space checks."""
    if candidates.size == 0:
        return candidates
    mask = np.ones(candidates.shape[0], dtype=bool)
    if visited_poses_xyz is not None and visited_poses_xyz.size and min_dist_from_visited > 0.0:
        diffs = candidates[:, None, :] - visited_poses_xyz[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask &= np.all(dists >= min_dist_from_visited, axis=1)
    if is_free_fn is not None:
        mask &= np.array([bool(is_free_fn(pt)) for pt in candidates], dtype=bool)
    return candidates[mask]


def _sample_uniform(
    rng: np.random.Generator,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_samples: int,
) -> NDArray[np.float64]:
    """Sample points uniformly within bounds."""
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=float)
    span = hi - lo
    return lo + rng.random((n_samples, 3)) * span


def _sample_sobol(
    rng: np.random.Generator,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_samples: int,
) -> NDArray[np.float64]:
    """
    Sample points using a Sobol sequence; fall back to uniform if unavailable.

    Degenerate dimensions (lo == hi) are kept fixed at their bound value.
    """
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=float)
    try:
        from scipy.stats import qmc
    except ImportError:
        return _sample_uniform(rng, lo, hi, n_samples)
    active_dims = hi > lo
    if not np.any(active_dims):
        return np.repeat(lo[None, :], n_samples, axis=0)
    seed = int(rng.integers(0, 2**32 - 1))
    sampler = qmc.Sobol(d=int(np.sum(active_dims)), scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_samples, 1))))
    sample = sampler.random_base2(m)
    sample = sample[:n_samples]
    scaled = qmc.scale(sample, lo[active_dims], hi[active_dims])
    out = np.repeat(lo[None, :], n_samples, axis=0)
    out[:, active_dims] = scaled
    return out


def _generate_ring_candidates(
    current_pose_xyz: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_candidates: int,
    min_dist_from_visited: float,
) -> NDArray[np.float64]:
    """Generate candidates on concentric rings around the current pose."""
    if n_candidates <= 0:
        return np.zeros((0, 3), dtype=float)
    max_dx = min(current_pose_xyz[0] - lo[0], hi[0] - current_pose_xyz[0])
    max_dy = min(current_pose_xyz[1] - lo[1], hi[1] - current_pose_xyz[1])
    max_radius = max(0.0, min(max_dx, max_dy))
    min_radius = max(0.1, min_dist_from_visited)
    if max_radius < min_radius:
        max_radius = min_radius
    num_rings = max(1, int(np.sqrt(n_candidates)))
    num_angles = max(4, int(np.ceil(n_candidates / num_rings)))
    radii = np.linspace(min_radius, max_radius, num=num_rings)
    angles = np.linspace(0.0, 2.0 * np.pi, num=num_angles, endpoint=False)
    points = []
    for r in radii:
        for theta in angles:
            x = current_pose_xyz[0] + r * np.cos(theta)
            y = current_pose_xyz[1] + r * np.sin(theta)
            z = current_pose_xyz[2]
            points.append([x, y, z])
    points_arr = np.array(points, dtype=float)
    points_arr = points_arr[:n_candidates]
    return points_arr


def generate_candidate_poses(
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None = None,
    n_candidates: int = 1024,
    strategy: str = "free_space_sobol",
    min_dist_from_visited: float = 1.0,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Return (L, 3) candidate poses in free space for the given strategy."""
    rng = np.random.default_rng() if rng is None else rng
    current_pose_xyz = np.asarray(current_pose_xyz, dtype=float)
    if current_pose_xyz.shape != (3,):
        raise ValueError("current_pose_xyz must be shape (3,).")
    visited = None
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim != 2 or visited.shape[1] != 3:
            raise ValueError("visited_poses_xyz must be shape (N, 3).")

    lo, hi = _resolve_bounds(bounds_xyz)
    is_free_fn = _resolve_free_space_checker(map_api)

    if strategy == "ring":
        raw = _generate_ring_candidates(
            current_pose_xyz=current_pose_xyz,
            lo=lo,
            hi=hi,
            n_candidates=max(n_candidates, 1),
            min_dist_from_visited=min_dist_from_visited,
        )
    elif strategy == "free_space_sobol":
        raw = _sample_sobol(rng, lo, hi, max(n_candidates * 3, n_candidates))
    elif strategy == "gaussian":
        raw = rng.normal(loc=current_pose_xyz, scale=0.75, size=(max(n_candidates * 3, n_candidates), 3))
        raw = np.clip(raw, lo, hi)
    else:
        raise ValueError(f"Unknown candidate generation strategy: {strategy}")

    filtered = _filter_candidates(raw, visited, min_dist_from_visited, is_free_fn)
    if filtered.shape[0] < n_candidates:
        map_centers = _map_free_cell_centers(
            map_api,
            z_value=float(current_pose_xyz[2]),
        )
        if map_centers.size:
            if visited is not None and visited.size:
                distances = np.linalg.norm(
                    map_centers[:, None, :2] - visited[None, :, :2],
                    axis=2,
                )
                order = np.argsort(np.min(distances, axis=1))[::-1]
                map_centers = map_centers[order]
            map_centers = _filter_candidates(
                map_centers,
                visited,
                min_dist_from_visited,
                is_free_fn,
            )
            if map_centers.size:
                filtered = np.vstack([filtered, map_centers])
    if filtered.shape[0] < n_candidates:
        extra = _sample_uniform(rng, lo, hi, max(n_candidates, 1))
        extra = _filter_candidates(extra, visited, min_dist_from_visited, is_free_fn)
        if extra.size:
            filtered = np.vstack([filtered, extra])
    return filtered[:n_candidates]
