"""Translate runtime simulation dictionaries into spectrum pipeline settings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spectrum.pipeline import SpectrumConfig


def background_rate_cps_from_runtime_config(
    runtime_config: Mapping[str, Any],
) -> float | None:
    """Return configured Geant4 background cps when present."""
    for key in ("background_cps", "background_rate_cps"):
        value = runtime_config.get(key)
        if value is not None:
            return max(float(value), 0.0)
    args = runtime_config.get("executable_args", ())
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        values = list(args)
        for idx, item in enumerate(values):
            if str(item) != "--background-cps":
                continue
            if idx + 1 >= len(values):
                return None
            return max(float(values[idx + 1]), 0.0)
    return None


def spectrum_config_from_runtime_config(
    runtime_config: Mapping[str, Any],
) -> SpectrumConfig:
    """Build a SpectrumConfig from the shared runtime dictionary semantics."""
    config = SpectrumConfig()
    scoring_mode = str(runtime_config.get("detector_scoring_mode", "")).strip().lower()
    source_rate_model = (
        str(runtime_config.get("source_rate_model", "")).strip().lower()
    )
    if (
        scoring_mode == "incident_gamma_energy"
        and "response_efficiency_model" not in runtime_config
    ):
        config.response_efficiency_model = "unit"
    if scoring_mode == "incident_gamma_energy":
        config.use_incident_gamma_response_matrix = True
    if source_rate_model == "detector_cps_1m":
        config.normalize_line_intensities = True
    background_rate = background_rate_cps_from_runtime_config(runtime_config)
    if background_rate is not None:
        config.response_poisson_background_rate_cps = float(background_rate)

    field_names = set(SpectrumConfig.__dataclass_fields__.keys())
    for key, value in runtime_config.items():
        if key not in field_names or value is None:
            continue
        current = getattr(config, key)
        if isinstance(current, bool):
            setattr(config, key, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(config, key, int(value))
        elif isinstance(current, float):
            setattr(config, key, float(value))
        else:
            setattr(config, key, value)
    if (
        background_rate is not None
        and "response_poisson_background_rate_cps" not in runtime_config
    ):
        config.response_poisson_background_rate_cps = float(background_rate)
    config.__post_init__()
    return config
