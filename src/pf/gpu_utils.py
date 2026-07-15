"""GPU helpers for batched continuous-kernel computations."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    TORCH_AVAILABLE = False

from measurement.shielding import (
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
)
from pf.state import IsotopeState

_LEGACY_CONTINUOUS_ISOTOPE = "__legacy__"


def _numpy_optional(value: np.ndarray | "torch.Tensor" | None) -> np.ndarray | None:
    """Return an optional array as a CPU numpy array."""
    if value is None:
        return None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _legacy_line_mu_by_isotope(
    line_weights: np.ndarray | "torch.Tensor" | None,
    line_mu_fe: np.ndarray | "torch.Tensor" | None,
    line_mu_pb: np.ndarray | "torch.Tensor" | None,
) -> dict[str, tuple[dict[str, float], ...]] | None:
    """Return ContinuousKernel line-mu payload for legacy GPU helper inputs."""
    weights = _numpy_optional(line_weights)
    mu_fe = _numpy_optional(line_mu_fe)
    mu_pb = _numpy_optional(line_mu_pb)
    if weights is None or mu_fe is None or mu_pb is None:
        return None
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mu_fe = np.asarray(mu_fe, dtype=float).reshape(-1)
    mu_pb = np.asarray(mu_pb, dtype=float).reshape(-1)
    if weights.size != mu_fe.size or weights.size != mu_pb.size or weights.size == 0:
        return None
    rows = tuple(
        {
            "weight": float(weight),
            "fe": float(fe_value),
            "pb": float(pb_value),
        }
        for weight, fe_value, pb_value in zip(weights, mu_fe, mu_pb)
        if float(weight) > 0.0
    )
    if not rows:
        return None
    return {_LEGACY_CONTINUOUS_ISOTOPE: rows}


def _legacy_obstacle_grid(
    obstacle_boxes_m: np.ndarray | "torch.Tensor" | None,
    obstacle_mu_cm_inv: float,
    obstacle_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None,
    obstacle_line_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None,
):
    """Return a ContinuousKernel obstacle grid for legacy GPU helper inputs."""
    boxes = _numpy_optional(obstacle_boxes_m)
    if boxes is None or np.asarray(boxes).size == 0:
        return None
    from measurement.obstacles import ObstacleGrid

    boxes_arr = np.asarray(boxes, dtype=float).reshape(-1, 6)
    mu_values = _numpy_optional(obstacle_mu_cm_inv_by_box)
    if mu_values is None:
        mu_values = np.full(boxes_arr.shape[0], float(obstacle_mu_cm_inv), dtype=float)
    else:
        mu_values = np.asarray(mu_values, dtype=float).reshape(-1)
    line_mu = _numpy_optional(obstacle_line_mu_cm_inv_by_box)
    line_payload = None
    if line_mu is not None:
        line_payload = {
            _LEGACY_CONTINUOUS_ISOTOPE: tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(line_mu, dtype=float).reshape(-1, boxes_arr.shape[0])
            )
        }
    return ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(0, 0),
        blocked_cells=(),
    ).with_transport_model(
        boxes_m=tuple(tuple(float(value) for value in row) for row in boxes_arr),
        mu_by_isotope={
            _LEGACY_CONTINUOUS_ISOTOPE: tuple(float(value) for value in mu_values)
        },
        line_mu_by_isotope=line_payload,
    )


def _legacy_continuous_kernel(
    *,
    mu_fe: float,
    mu_pb: float,
    thickness_fe_cm: float,
    thickness_pb_cm: float,
    use_angle_attenuation: bool,
    inner_radius_fe_cm: float,
    inner_radius_pb_cm: float,
    shield_geometry_model: str,
    obstacle_boxes_m: np.ndarray | "torch.Tensor" | None,
    obstacle_mu_cm_inv: float,
    obstacle_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None,
    obstacle_line_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None,
    line_weights: np.ndarray | "torch.Tensor" | None,
    line_mu_fe: np.ndarray | "torch.Tensor" | None,
    line_mu_pb: np.ndarray | "torch.Tensor" | None,
    detector_radius_m: float,
    detector_aperture_radius_m: float | None,
    detector_aperture_samples: int,
    buildup_fe_coeff: float,
    buildup_pb_coeff: float,
    obstacle_buildup_coeff: float,
    device: "torch.device",
    dtype: "torch.dtype",
):
    """Build a ContinuousKernel matching the legacy GPU helper arguments."""
    from measurement.continuous_kernels import ContinuousKernel
    from measurement.kernels import ShieldParams

    return ContinuousKernel(
        mu_by_isotope={
            _LEGACY_CONTINUOUS_ISOTOPE: {
                "fe": float(mu_fe),
                "pb": float(mu_pb),
            }
        },
        shield_params=ShieldParams(
            mu_fe=float(mu_fe),
            mu_pb=float(mu_pb),
            thickness_fe_cm=float(thickness_fe_cm),
            thickness_pb_cm=float(thickness_pb_cm),
            inner_radius_fe_cm=float(inner_radius_fe_cm),
            inner_radius_pb_cm=float(inner_radius_pb_cm),
            buildup_fe_coeff=float(buildup_fe_coeff),
            buildup_pb_coeff=float(buildup_pb_coeff),
            shield_geometry_model=str(shield_geometry_model),
            use_angle_attenuation=bool(use_angle_attenuation),
        ),
        obstacle_grid=_legacy_obstacle_grid(
            obstacle_boxes_m,
            obstacle_mu_cm_inv,
            obstacle_mu_cm_inv_by_box,
            obstacle_line_mu_cm_inv_by_box,
        ),
        obstacle_buildup_coeff=float(obstacle_buildup_coeff),
        detector_radius_m=float(detector_radius_m),
        detector_aperture_radius_m=detector_aperture_radius_m,
        detector_aperture_samples=int(detector_aperture_samples),
        line_mu_by_isotope=_legacy_line_mu_by_isotope(
            line_weights,
            line_mu_fe,
            line_mu_pb,
        ),
        use_gpu=True,
        gpu_device=str(device),
        gpu_dtype="float64" if dtype is torch.float64 else "float32",
    )


def torch_available() -> bool:
    """Return True if torch is available and CUDA is usable."""
    return bool(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def torch_installed() -> bool:
    """Return True if torch is available (CUDA not required)."""
    return bool(TORCH_AVAILABLE and torch is not None)


def torch_device_available(device: str | None = None) -> bool:
    """Return True when torch can run on the requested device."""
    if not torch_installed():
        return False
    device_name = "cuda" if device is None else str(device)
    if device_name.startswith("cuda"):
        return bool(torch is not None and torch.cuda.is_available())
    return True


def resolve_device(device: str | None) -> "torch.device":
    """Resolve a torch device string with CUDA fallback."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if device is None:
        device = "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return torch.device(device)


def resolve_dtype(dtype: str) -> "torch.dtype":
    """Map a dtype string to a torch dtype."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def pack_states(
    states: Iterable[IsotopeState],
    device: "torch.device",
    dtype: "torch.dtype",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Pack IsotopeState list into padded tensors.

    Returns (positions, strengths, backgrounds, mask).
    """
    states_list = list(states)
    num_particles = len(states_list)
    max_r = max((st.num_sources for st in states_list), default=0)
    positions = np.zeros((num_particles, max_r, 3), dtype=float)
    strengths = np.zeros((num_particles, max_r), dtype=float)
    mask = np.zeros((num_particles, max_r), dtype=float)
    backgrounds = np.zeros(num_particles, dtype=float)
    for i, st in enumerate(states_list):
        r = st.num_sources
        if r > 0:
            positions[i, :r] = st.positions
            strengths[i, :r] = st.strengths
            mask[i, :r] = 1.0
        backgrounds[i] = st.background
    pos_t = torch.as_tensor(positions, device=device, dtype=dtype)
    str_t = torch.as_tensor(strengths, device=device, dtype=dtype)
    mask_t = torch.as_tensor(mask, device=device, dtype=dtype)
    bg_t = torch.as_tensor(backgrounds, device=device, dtype=dtype)
    return pos_t, str_t, bg_t, mask_t


def expected_counts_pair_torch(
    detector_pos: np.ndarray,
    positions: "torch.Tensor",
    strengths: "torch.Tensor",
    backgrounds: "torch.Tensor",
    mask: "torch.Tensor",
    fe_index: int,
    pb_index: int,
    mu_fe: float,
    mu_pb: float,
    thickness_fe_cm: float,
    thickness_pb_cm: float,
    live_time_s: float,
    device: "torch.device",
    dtype: "torch.dtype",
    use_angle_attenuation: bool = False,
    source_scale: float = 1.0,
    inner_radius_fe_cm: float = DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    inner_radius_pb_cm: float = DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    shield_geometry_model: str = SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    obstacle_boxes_m: np.ndarray | "torch.Tensor" | None = None,
    obstacle_mu_cm_inv: float = 0.0,
    obstacle_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None = None,
    obstacle_line_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None = None,
    line_weights: np.ndarray | "torch.Tensor" | None = None,
    line_mu_fe: np.ndarray | "torch.Tensor" | None = None,
    line_mu_pb: np.ndarray | "torch.Tensor" | None = None,
    detector_radius_m: float = 0.0,
    detector_aperture_radius_m: float | None = None,
    detector_aperture_samples: int = 1,
    buildup_fe_coeff: float = 0.0,
    buildup_pb_coeff: float = 0.0,
    obstacle_buildup_coeff: float = 0.0,
    obstacle_box_chunk_size: int = 64,
    tol: float = 1e-6,
) -> "torch.Tensor":
    """
    Compute Λ for all particles at a Fe/Pb orientation pair on GPU.

    When use_angle_attenuation is False, the shield path length is treated as a
    constant thickness for blocked rays (no 1/cos(theta) scaling).
    """
    if torch is None:
        raise RuntimeError("torch is not available")
    kernel = _legacy_continuous_kernel(
        mu_fe=mu_fe,
        mu_pb=mu_pb,
        thickness_fe_cm=thickness_fe_cm,
        thickness_pb_cm=thickness_pb_cm,
        use_angle_attenuation=use_angle_attenuation,
        inner_radius_fe_cm=inner_radius_fe_cm,
        inner_radius_pb_cm=inner_radius_pb_cm,
        shield_geometry_model=shield_geometry_model,
        obstacle_boxes_m=obstacle_boxes_m,
        obstacle_mu_cm_inv=obstacle_mu_cm_inv,
        obstacle_mu_cm_inv_by_box=obstacle_mu_cm_inv_by_box,
        obstacle_line_mu_cm_inv_by_box=obstacle_line_mu_cm_inv_by_box,
        line_weights=line_weights,
        line_mu_fe=line_mu_fe,
        line_mu_pb=line_mu_pb,
        detector_radius_m=detector_radius_m,
        detector_aperture_radius_m=detector_aperture_radius_m,
        detector_aperture_samples=detector_aperture_samples,
        buildup_fe_coeff=buildup_fe_coeff,
        buildup_pb_coeff=buildup_pb_coeff,
        obstacle_buildup_coeff=obstacle_buildup_coeff,
        device=device,
        dtype=dtype,
    )
    return kernel.expected_counts_pair_for_packed_states_torch(
        isotope=_LEGACY_CONTINUOUS_ISOTOPE,
        detector_pos=np.asarray(detector_pos, dtype=float),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=int(fe_index),
        pb_index=int(pb_index),
        live_time_s=float(live_time_s),
        source_scale=source_scale,
        device=device,
        dtype=dtype,
    )


def expected_counts_all_pairs_torch(
    detector_pos: np.ndarray,
    positions: "torch.Tensor",
    strengths: "torch.Tensor",
    backgrounds: "torch.Tensor",
    mask: "torch.Tensor",
    mu_fe: float,
    mu_pb: float,
    thickness_fe_cm: float,
    thickness_pb_cm: float,
    live_time_s: float,
    device: "torch.device",
    dtype: "torch.dtype",
    use_angle_attenuation: bool = False,
    source_scale: float | np.ndarray | "torch.Tensor" = 1.0,
    inner_radius_fe_cm: float = DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    inner_radius_pb_cm: float = DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    shield_geometry_model: str = SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    obstacle_boxes_m: np.ndarray | "torch.Tensor" | None = None,
    obstacle_mu_cm_inv: float = 0.0,
    obstacle_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None = None,
    obstacle_line_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None = None,
    line_weights: np.ndarray | "torch.Tensor" | None = None,
    line_mu_fe: np.ndarray | "torch.Tensor" | None = None,
    line_mu_pb: np.ndarray | "torch.Tensor" | None = None,
    detector_radius_m: float = 0.0,
    detector_aperture_radius_m: float | None = None,
    detector_aperture_samples: int = 1,
    buildup_fe_coeff: float = 0.0,
    buildup_pb_coeff: float = 0.0,
    obstacle_buildup_coeff: float = 0.0,
    obstacle_box_chunk_size: int = 64,
    tol: float = 1e-6,
) -> "torch.Tensor":
    """Compute expected counts for all Fe/Pb orientation pairs in one batch."""
    if torch is None:
        raise RuntimeError("torch is not available")
    kernel = _legacy_continuous_kernel(
        mu_fe=mu_fe,
        mu_pb=mu_pb,
        thickness_fe_cm=thickness_fe_cm,
        thickness_pb_cm=thickness_pb_cm,
        use_angle_attenuation=use_angle_attenuation,
        inner_radius_fe_cm=inner_radius_fe_cm,
        inner_radius_pb_cm=inner_radius_pb_cm,
        shield_geometry_model=shield_geometry_model,
        obstacle_boxes_m=obstacle_boxes_m,
        obstacle_mu_cm_inv=obstacle_mu_cm_inv,
        obstacle_mu_cm_inv_by_box=obstacle_mu_cm_inv_by_box,
        obstacle_line_mu_cm_inv_by_box=obstacle_line_mu_cm_inv_by_box,
        line_weights=line_weights,
        line_mu_fe=line_mu_fe,
        line_mu_pb=line_mu_pb,
        detector_radius_m=detector_radius_m,
        detector_aperture_radius_m=detector_aperture_radius_m,
        detector_aperture_samples=detector_aperture_samples,
        buildup_fe_coeff=buildup_fe_coeff,
        buildup_pb_coeff=buildup_pb_coeff,
        obstacle_buildup_coeff=obstacle_buildup_coeff,
        device=device,
        dtype=dtype,
    )
    return kernel.expected_counts_all_pairs_for_packed_states_torch(
        isotope=_LEGACY_CONTINUOUS_ISOTOPE,
        detector_pos=np.asarray(detector_pos, dtype=float),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        live_time_s=float(live_time_s),
        source_scale=source_scale,
        device=device,
        dtype=dtype,
    )


def expected_counts_selected_pairs_torch(
    detector_pos: np.ndarray,
    positions: "torch.Tensor",
    strengths: "torch.Tensor",
    backgrounds: "torch.Tensor",
    mask: "torch.Tensor",
    fe_indices: np.ndarray | "torch.Tensor",
    pb_indices: np.ndarray | "torch.Tensor",
    mu_fe: float,
    mu_pb: float,
    thickness_fe_cm: float,
    thickness_pb_cm: float,
    live_time_s: float,
    device: "torch.device",
    dtype: "torch.dtype",
    use_angle_attenuation: bool = False,
    source_scale: float | np.ndarray | "torch.Tensor" = 1.0,
    inner_radius_fe_cm: float = DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    inner_radius_pb_cm: float = DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    shield_geometry_model: str = SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    obstacle_boxes_m: np.ndarray | "torch.Tensor" | None = None,
    obstacle_mu_cm_inv: float = 0.0,
    obstacle_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None = None,
    obstacle_line_mu_cm_inv_by_box: np.ndarray | "torch.Tensor" | None = None,
    line_weights: np.ndarray | "torch.Tensor" | None = None,
    line_mu_fe: np.ndarray | "torch.Tensor" | None = None,
    line_mu_pb: np.ndarray | "torch.Tensor" | None = None,
    detector_radius_m: float = 0.0,
    detector_aperture_radius_m: float | None = None,
    detector_aperture_samples: int = 1,
    buildup_fe_coeff: float = 0.0,
    buildup_pb_coeff: float = 0.0,
    obstacle_buildup_coeff: float = 0.0,
    obstacle_box_chunk_size: int = 64,
    tol: float = 1e-6,
) -> "torch.Tensor":
    """Compute expected counts only for the requested Fe/Pb orientation pairs."""
    if torch is None:
        raise RuntimeError("torch is not available")
    kernel = _legacy_continuous_kernel(
        mu_fe=mu_fe,
        mu_pb=mu_pb,
        thickness_fe_cm=thickness_fe_cm,
        thickness_pb_cm=thickness_pb_cm,
        use_angle_attenuation=use_angle_attenuation,
        inner_radius_fe_cm=inner_radius_fe_cm,
        inner_radius_pb_cm=inner_radius_pb_cm,
        shield_geometry_model=shield_geometry_model,
        obstacle_boxes_m=obstacle_boxes_m,
        obstacle_mu_cm_inv=obstacle_mu_cm_inv,
        obstacle_mu_cm_inv_by_box=obstacle_mu_cm_inv_by_box,
        obstacle_line_mu_cm_inv_by_box=obstacle_line_mu_cm_inv_by_box,
        line_weights=line_weights,
        line_mu_fe=line_mu_fe,
        line_mu_pb=line_mu_pb,
        detector_radius_m=detector_radius_m,
        detector_aperture_radius_m=detector_aperture_radius_m,
        detector_aperture_samples=detector_aperture_samples,
        buildup_fe_coeff=buildup_fe_coeff,
        buildup_pb_coeff=buildup_pb_coeff,
        obstacle_buildup_coeff=obstacle_buildup_coeff,
        device=device,
        dtype=dtype,
    )
    return kernel.expected_counts_selected_pairs_for_packed_states_torch(
        isotope=_LEGACY_CONTINUOUS_ISOTOPE,
        detector_pos=np.asarray(detector_pos, dtype=float),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=np.asarray(fe_indices, dtype=np.int64),
        pb_indices=np.asarray(pb_indices, dtype=np.int64),
        live_time_s=float(live_time_s),
        source_scale=source_scale,
        device=device,
        dtype=dtype,
    )
