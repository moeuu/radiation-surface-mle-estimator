"""Standalone surface maximum-likelihood radiation estimation."""

from .config import MLEConfig, build_default_config
from .estimator import SurfaceMLEEstimator, fit_surface_mle
from .estimator_backend import SurfaceMLEBackend
from .observation_batch import (
    observation_batch_from_log,
    observation_batch_from_records,
)
from .replay import ReplayContext, ReplayResult, prepare_replay, run_replay
from .reporting import MLEReportPaths, load_mle_estimate, save_mle_estimate
from .response_builder import (
    CountResponseMatrices,
    build_count_response,
    build_count_responses,
    build_density_response,
)
from .solver import (
    SurfaceMapConfig,
    SurfaceMapObjective,
    SurfaceMapResult,
    evaluate_surface_map_objective,
    fit_surface_map_poisson,
)
from .types import (
    MLEEstimate,
    ObservationBatch,
    SurfacePatch,
    SurfacePatchSet,
)
from .spectral_response_builder import (
    SpectralResponseResult,
    build_spectral_nuisance_response,
    build_spectral_response,
)
from .surface_patches import build_surface_patches, refine_surface_patches

__all__ = [
    "MLEConfig",
    "MLEEstimate",
    "MLEReportPaths",
    "ObservationBatch",
    "ReplayContext",
    "ReplayResult",
    "SurfaceMapConfig",
    "SurfaceMapObjective",
    "SurfaceMapResult",
    "SurfaceMLEBackend",
    "SurfaceMLEEstimator",
    "SurfacePatch",
    "SurfacePatchSet",
    "CountResponseMatrices",
    "SpectralResponseResult",
    "build_count_response",
    "build_count_responses",
    "build_default_config",
    "build_density_response",
    "build_spectral_nuisance_response",
    "build_spectral_response",
    "build_surface_patches",
    "evaluate_surface_map_objective",
    "fit_surface_map_poisson",
    "fit_surface_mle",
    "load_mle_estimate",
    "observation_batch_from_log",
    "observation_batch_from_records",
    "prepare_replay",
    "run_replay",
    "refine_surface_patches",
    "save_mle_estimate",
]
