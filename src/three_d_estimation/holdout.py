"""Fail-closed execution of a final MLE on an unseen Geant4 environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from runtime.discrepancy_calibration import load_discrepancy_calibration
from runtime.measurement_log import MeasurementLog, load_measurement_log
from runtime.records import canonical_json_sha256

from .config import MLEConfig
from .replay import run_replay


@dataclass(frozen=True, slots=True)
class RALHoldoutResult:
    """Describe one strictly separated final-environment replay."""

    tuning_run_id: str
    holdout_run_id: str
    holdout_output_dir: Path
    tuning_environment_id: str
    holdout_environment_id: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe evaluation manifest summary."""
        return {
            "schema_version": 1,
            "evaluation": "unseen_geant4_environment",
            "tuning_run_id": self.tuning_run_id,
            "holdout_run_id": self.holdout_run_id,
            "tuning_environment_id": self.tuning_environment_id,
            "holdout_environment_id": self.holdout_environment_id,
            "holdout_output_dir": self.holdout_output_dir.as_posix(),
            "truth_used_by_estimator": False,
        }


def _environment_id(log: MeasurementLog) -> str:
    """Return the required non-secret environment realization identifier."""
    metadata = log.run_manifest.get("metadata", {})
    value = (
        metadata.get("environment_realization_id")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Holdout logs require metadata.environment_realization_id.")
    return value.strip()


def validate_holdout_separation(
    tuning_log: MeasurementLog,
    holdout_log: MeasurementLog,
    config: MLEConfig,
) -> tuple[str, str]:
    """Prove that tuning, calibration, and final evaluation are disjoint."""
    if not isinstance(tuning_log, MeasurementLog) or not isinstance(
        holdout_log,
        MeasurementLog,
    ):
        raise TypeError("tuning_log and holdout_log must be MeasurementLog objects.")
    if not isinstance(config, MLEConfig):
        raise TypeError("config must be MLEConfig.")
    if tuning_log.run_id == holdout_log.run_id:
        raise ValueError("Tuning and holdout run IDs must differ.")
    tuning_environment = _environment_id(tuning_log)
    holdout_environment = _environment_id(holdout_log)
    if tuning_environment == holdout_environment:
        raise ValueError("Tuning and holdout environment IDs must differ.")
    if canonical_json_sha256(tuning_log.environment) == canonical_json_sha256(
        holdout_log.environment
    ):
        raise ValueError("Tuning and holdout environment manifests must differ.")
    if config.regularization_selection != "fixed":
        raise ValueError(
            "Final holdout replay cannot tune regularization on holdout data."
        )
    if config.tuning_environment_id not in {None, tuning_environment}:
        raise ValueError(
            "MLEConfig tuning_environment_id disagrees with the tuning log."
        )
    if config.final_holdout_environment_id not in {None, holdout_environment}:
        raise ValueError(
            "MLEConfig final_holdout_environment_id disagrees with the holdout log."
        )
    if config.discrepancy_calibration_path is not None:
        calibration = load_discrepancy_calibration(config.discrepancy_calibration_path)
        if holdout_environment in calibration.independent_environment_ids:
            raise ValueError(
                "Final holdout environment was used by discrepancy calibration."
            )
    return tuning_environment, holdout_environment


def run_ral_holdout(
    tuning_run_dir: str | Path,
    holdout_run_dir: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> RALHoldoutResult:
    """Run one authoritative spectral MLE only after separation validation."""
    tuning = load_measurement_log(Path(tuning_run_dir).expanduser().resolve())
    holdout = load_measurement_log(Path(holdout_run_dir).expanduser().resolve())
    config = MLEConfig.load(config_path)
    if config.mode != "spectral":
        raise ValueError("RAL holdout requires the authoritative spectral MLE.")
    tuning_environment, holdout_environment = validate_holdout_separation(
        tuning,
        holdout,
        config,
    )
    target = Path(output_dir).expanduser().resolve()
    run_replay(holdout.path, config=config, output_dir=target)
    result = RALHoldoutResult(
        tuning_run_id=tuning.run_id,
        holdout_run_id=holdout.run_id,
        holdout_output_dir=target,
        tuning_environment_id=tuning_environment,
        holdout_environment_id=holdout_environment,
    )
    (target / "holdout_manifest.json").write_text(
        json.dumps(result.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "RALHoldoutResult",
    "run_ral_holdout",
    "validate_holdout_separation",
]
