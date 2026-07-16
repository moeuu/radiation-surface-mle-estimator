"""Repository and estimator provenance for standalone surface MLE outputs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_commit(root: str | Path = _REPOSITORY_ROOT) -> str:
    """Return the local Git commit without invoking Git or another repository."""
    repository = Path(root)
    git_entry = repository / ".git"
    if git_entry.is_file():
        text = git_entry.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            git_entry = (repository / text.removeprefix("gitdir:").strip()).resolve()
    if git_entry.is_dir():
        head = (git_entry / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            reference = head.removeprefix("ref:").strip()
            loose = git_entry / reference
            if loose.is_file():
                value = loose.read_text(encoding="utf-8").strip()
                if value:
                    return value
            packed = git_entry / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("#", "^")) or not line.strip():
                        continue
                    value, name = line.split(" ", 1)
                    if name.strip() == reference:
                        return value.strip()
        elif head:
            return head
    return "unknown-standalone-build"


def resolved_mapping_sha256(payload: Mapping[str, object]) -> str:
    """Return a compact canonical SHA-256 for a resolved JSON mapping."""
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def estimator_provenance(
    *,
    variant: str,
    measurement_log_schema_version: int | None = None,
    measurement_run_id: str | None = None,
    measurement_repository_commit: str | None = None,
    resolved_config_sha256: str | None = None,
    forward_model_manifest_sha256: str | None = None,
    measurement_log_sha256: str | None = None,
    config_sha256: str | None = None,
    resolved_estimator_config_sha256: str | None = None,
) -> dict[str, object]:
    """Return mandatory pure-MLE provenance with optional replay identities."""
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in {"count", "spectral"}:
        raise ValueError("variant must be 'count' or 'spectral'.")
    commit = repository_commit()
    return {
        "estimator_family": "surface_mle",
        "estimator_variant": normalized_variant,
        "candidate_domain": "complete_surface_dictionary",
        "uses_pf_state": False,
        "uses_pf_candidates": False,
        "estimator_repository": "moeuu/3D_estimation",
        "estimator_commit": commit,
        "repository_commit": commit,
        "measurement_log_schema_version": measurement_log_schema_version,
        "measurement_run_id": measurement_run_id,
        "measurement_repository_commit": measurement_repository_commit,
        "resolved_config_sha256": resolved_config_sha256,
        "forward_model_manifest_sha256": forward_model_manifest_sha256,
        "measurement_log_sha256": measurement_log_sha256,
        "config_sha256": config_sha256,
        "resolved_estimator_config_sha256": resolved_estimator_config_sha256,
    }


__all__ = ["estimator_provenance", "repository_commit", "resolved_mapping_sha256"]
