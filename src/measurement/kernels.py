"""Legacy grid-based kernel calculations (Chapter 3, Sec. 3.2/3.4).

Note: This module uses discrete candidate_sources indices. New continuous PF code
uses measurement.continuous_kernels instead.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import (
    CS137_TVL_FE_MM,
    CS137_TVL_PB_MM,
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_FE_SHIELD_THICKNESS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_THICKNESS_CM,
    LOCAL_POSITIVE_OCTANT_CENTER,
    OctantShield,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    mu_from_tvl_mm,
    resolve_mu_values,
    rotation_matrix_between_vectors,
    spherical_shell_path_length_cm_torch,
)

if TYPE_CHECKING:
    import torch

CS137_MU_PB_CM_INV = mu_from_tvl_mm(CS137_TVL_PB_MM)
CS137_MU_FE_CM_INV = mu_from_tvl_mm(CS137_TVL_FE_MM)


def _torch_available() -> bool:
    """Return True if torch with CUDA is available."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _torch_installed() -> bool:
    """Return True if torch is installed (CUDA not required)."""
    return importlib.util.find_spec("torch") is not None


def _resolve_device(device: str) -> "torch.device":
    """Resolve a torch device string with CUDA fallback."""
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> "torch.dtype":
    """Map a dtype string to a torch dtype."""
    import torch

    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


@dataclass(frozen=True)
class ShieldParams:
    """Lightweight shield material and thickness parameters."""

    mu_pb: float = CS137_MU_PB_CM_INV  # 1/cm based on Cs-137 TVL.
    mu_fe: float = CS137_MU_FE_CM_INV  # 1/cm based on Cs-137 TVL.
    thickness_pb_cm: float = DEFAULT_PB_SHIELD_THICKNESS_CM
    thickness_fe_cm: float = DEFAULT_FE_SHIELD_THICKNESS_CM
    inner_radius_fe_cm: float = DEFAULT_FE_SHIELD_INNER_RADIUS_CM
    inner_radius_pb_cm: float = DEFAULT_PB_SHIELD_INNER_RADIUS_CM
    buildup_fe_coeff: float = 0.0
    buildup_pb_coeff: float = 0.0
    shield_geometry_model: str = SHIELD_GEOMETRY_SPHERICAL_OCTANT
    use_angle_attenuation: bool = False  # Legacy planar-slab mode; spherical shells use exact radial length.


class KernelPrecomputer:
    """
    Precompute geometric + shielding kernels for poses, orientations, and sources.

    This follows the model in Sec. 3.2 and 3.4.
    """

    def __init__(
        self,
        candidate_sources: NDArray[np.float64],
        poses: NDArray[np.float64],
        orientations: NDArray[np.float64],
        shield_params: ShieldParams,
        mu_by_isotope: Dict[str, object],
        use_gpu: bool = True,
        gpu_device: str = "cuda",
        gpu_dtype: str = "float32",
    ) -> None:
        """
        Args:
            candidate_sources: (J,3) candidate source positions.
            poses: (K,3) detector poses.
            orientations: (R,3) shield normal vectors.
            shield_params: ShieldParams instance.
            mu_by_isotope: Per-isotope linear attenuation (1/cm): float, (fe, pb), or {"fe","pb"}.
            use_gpu: Enable CUDA acceleration for kernel computation.
            gpu_device: Torch device string (e.g., "cuda" or "cpu").
            gpu_dtype: Torch dtype string ("float32" or "float64").
        """
        self.sources = candidate_sources
        self.poses = poses
        self.orientations = orientations
        self.shield_params = shield_params
        self.mu_by_isotope = mu_by_isotope
        self.num_sources = candidate_sources.shape[0]
        self.num_poses = poses.shape[0]
        self.num_orient = orientations.shape[0]
        self.octant_shield = OctantShield()
        self._theta_phi_ranges = self.octant_shield.theta_phi_ranges
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.gpu_dtype = gpu_dtype
        self._torch_sources_cache: dict[tuple[str, str], "torch.Tensor"] = {}

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        if not self.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu for KernelPrecomputer.")
        if not _torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

    def _sources_torch(self, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
        """Return sources as a cached torch tensor on the requested device/dtype."""
        import torch

        key = (str(device), str(dtype))
        cached = self._torch_sources_cache.get(key)
        if cached is not None:
            return cached
        src_t = torch.as_tensor(self.sources, device=device, dtype=dtype)
        self._torch_sources_cache[key] = src_t
        return src_t

    def _kernel_gpu(self, isotope: str, pose_idx: int, orient_idx: int) -> NDArray[np.float64]:
        """Compute the kernel using torch on the configured device."""
        import torch

        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        pose = torch.as_tensor(self.poses[pose_idx], device=device, dtype=dtype)
        sources = self._sources_torch(device, dtype)
        direction = pose - sources
        dist = torch.linalg.norm(direction, dim=1)
        tol = torch.as_tensor(1e-6, device=device, dtype=dtype)
        dist = torch.where(dist <= tol, tol, dist)
        dir_unit = direction / dist.unsqueeze(-1)
        geom = 1.0 / (dist**2)

        physical_normal = -np.asarray(self.orientations[orient_idx], dtype=float)
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            physical_normal,
        )
        rotation = torch.as_tensor(rotation_np, device=device, dtype=dtype)
        detector_to_source_unit = -dir_unit
        blocked = torch.all(detector_to_source_unit @ rotation >= -float(tol), dim=-1)

        normal = torch.as_tensor(self.orientations[orient_idx], device=device, dtype=dtype)
        cos_theta = torch.clamp(torch.sum(dir_unit * normal, dim=1), 0.0, 1.0)
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            L_fe = spherical_shell_path_length_cm_torch(
                direction,
                self.shield_params.inner_radius_fe_cm,
                self.shield_params.inner_radius_fe_cm + self.shield_params.thickness_fe_cm,
                blocked,
            )
            L_pb = spherical_shell_path_length_cm_torch(
                direction,
                self.shield_params.inner_radius_pb_cm,
                self.shield_params.inner_radius_pb_cm + self.shield_params.thickness_pb_cm,
                blocked,
            )
        elif self.shield_params.use_angle_attenuation:
            L_fe = torch.where(
                blocked & (cos_theta > tol),
                torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype) / cos_theta,
                torch.zeros_like(cos_theta),
            )
            L_pb = torch.where(
                blocked & (cos_theta > tol),
                torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype) / cos_theta,
                torch.zeros_like(cos_theta),
            )
        else:
            L_fe = torch.where(
                blocked,
                torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype),
                torch.zeros_like(cos_theta),
            )
            L_pb = torch.where(
                blocked,
                torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype),
                torch.zeros_like(cos_theta),
            )
        mu_fe, mu_pb = resolve_mu_values(
            self.mu_by_isotope, isotope, default_fe=self.shield_params.mu_fe, default_pb=self.shield_params.mu_pb
        )
        tau_fe = mu_fe * L_fe
        tau_pb = mu_pb * L_pb
        buildup = (
            1.0
            + max(float(self.shield_params.buildup_fe_coeff), 0.0) * (1.0 - torch.exp(-tau_fe))
            + max(float(self.shield_params.buildup_pb_coeff), 0.0) * (1.0 - torch.exp(-tau_pb))
        )
        att = torch.clamp(torch.exp(-(tau_fe + tau_pb)) * buildup, max=1.0)
        kernels = geom * att
        return kernels.detach().cpu().numpy()

    def geometric_term(self, pose: NDArray[np.float64], source: NDArray[np.float64]) -> float:
        """Inverse-square geometric term 1/d^2 (cps@1m scaling)."""
        d = np.linalg.norm(pose - source)
        if d == 0:
            d = 1e-6
        return float(1.0 / (d**2))

    def kernel(
        self,
        isotope: str,
        pose_idx: int,
        orient_idx: int,
    ) -> NDArray[np.float64]:
        """
        Return the expected-count kernel (J,) for unit source strength.

        Includes geometric term and exponential attenuation based on shield thickness
        and per-isotope linear attenuation coefficients.
        """
        self._gpu_enabled()
        return self._kernel_gpu(isotope, pose_idx, orient_idx)

    def expected_counts(
        self,
        isotope: str,
        pose_idx: int,
        orient_idx: int,
        source_strengths: NDArray[np.float64],
        background: float = 0.0,
        live_time_s: float = 1.0,
    ) -> float:
        """
        Compute expected counts from source strengths (simulation/test helper).

        PF updates always use isotope-wise counts from spectrum unfolding (Sec. 2.5.7);
        this helper is not used directly for PF weight updates.
        """
        kvec = self.kernel(isotope, pose_idx, orient_idx)
        return float(live_time_s * (np.dot(kvec, source_strengths) + background))
