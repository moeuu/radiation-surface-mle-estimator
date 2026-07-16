"""Strict forward-model identity contract for MeasurementLog schema version 1."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from measurement.shielding import line_resolved_shield_mu_by_isotope
from runtime.records import canonical_json_sha256


FORWARD_MODEL_MANIFEST_SCHEMA_VERSION = 1
SOURCE_RATE_MODEL = "detector_cps_1m"
SOURCE_RATE_SEMANTICS = {
    "quantity": "expected_net_detector_count_rate",
    "unit": "cps",
    "normalization_distance_m": 1.0,
}
CANONICAL_UNITS = {
    "distance": "m",
    "time": "s",
    "energy": "keV",
    "source_strength": "detector_cps_1m",
    "linear_attenuation": "cm^-1",
}
RESPONSE_SEMANTICS = {
    "distance_attenuation": "inverse_square_with_modelled_near_field",
    "detector_geometry": "model_identifier_bound",
    "shield_attenuation": "fe_pb_orientation_pair_8x8",
    "obstacle_attenuation": "line_segment_material_attenuation",
    "live_time_scaling": "expected_counts_linear_in_live_time_s",
    "line_resolved_response": "energy_bin_integrated_isotope_line_response",
}
REQUIRED_MODEL_NAMES = (
    "detector",
    "shield",
    "environment",
    "obstacle",
    "transport",
    "spectrum",
)
CONFORMANCE_FORWARD_MODEL_ID = "rotating-shield-analytic-conformance-v1"
CONFORMANCE_MODEL_IDENTIFIERS = {
    "detector": {
        "id": "detector-v1",
        "sha256": "981049f0f4814240604524186d326e046cf23d9dfeb8b7d71ca3f1480bceaf6e",
    },
    "shield": {
        "id": "shield-fe-pb-8x8-v1",
        "sha256": "c5e24ded41d8f15b59cbcb08d37c41d281a3867aa39e5fde4bf1bfb6004160f3",
    },
    "environment": {
        "id": "room-6x6x3-v1",
        "sha256": "d89a96dac3846f84e72daac9559a95812e291824ac023d0f29420e37df798673",
    },
    "obstacle": {
        "id": "obstacle-box-v1",
        "sha256": "b3fb1cbad6e3fd9c44feb6d3a1a12a733b0ddd93ab87a083e4f9fde631d0c7bc",
    },
    "transport": {
        "id": "analytic-transport-v1",
        "sha256": "232443b41c8862d6247f4e8c2bd22d96e416107b50475f171be464540c7fa117",
    },
    "spectrum": {
        "id": "spectrum-lines-v1",
        "sha256": "49cc8ee41dea713ed6dcae459d676ffe78e6b70cacbfea2eba6df2eb732ace73",
    },
}
_CONFORMANCE_SHIELD_PROGRAM = {
    "fe_orientation_count": 8,
    "pb_orientation_count": 8,
    "pair_count": 64,
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def registered_conformance_line_mu_by_isotope() -> dict[str, list[dict[str, float]]]:
    """Return the exact production spectral table bound by the fixture registry."""
    isotopes = ("Cs-137", "Co-60", "Eu-154")
    raw = line_resolved_shield_mu_by_isotope(
        isotopes=isotopes,
        normalize_line_intensities=True,
    )
    return {
        isotope: [
            {name: float(entry[name]) for name in ("energy_keV", "weight", "fe", "pb")}
            for entry in raw[isotope]
        ]
        for isotope in isotopes
    }


def _line_energy_weight_table(
    line_table: Mapping[str, object],
) -> dict[str, list[dict[str, float]]]:
    """Return the spectrum identity subset of a full line attenuation table."""
    return {
        str(isotope): [
            {
                "energy_keV": float(entry["energy_keV"]),
                "weight": float(entry["weight"]),
            }
            for entry in entries
        ]
        for isotope, entries in line_table.items()
    }


def _selected(
    payload: Mapping[str, object],
    *tokens: str,
) -> dict[str, object]:
    """Return a deterministic copy of keys related to any supplied token."""
    lowered = tuple(token.lower() for token in tokens)
    return {
        str(key): deepcopy(value)
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
        if any(token in str(key).lower() for token in lowered)
    }


def _identifier(
    payloads: tuple[Mapping[str, object], ...],
    keys: tuple[str, ...],
    default: str,
) -> str:
    """Return the first explicit non-empty model identifier or a local default."""
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return default


def _safe_relative_asset_path(path_value: object, *, field_name: str) -> Path:
    """Return one canonical relative asset path without traversal ambiguity."""
    raw = str(path_value)
    if not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path.")
    if "\\" in raw:
        raise ValueError(f"{field_name} must use portable forward-slash separators.")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"{field_name} must be relative; absolute paths are forbidden.")
    if not relative.parts or any(part in {"", ".."} for part in relative.parts):
        raise ValueError(f"{field_name} must not contain parent-directory traversal.")
    return relative


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_file_backed_model_asset(
    path_value: object,
    *,
    field_name: str,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> Path:
    """Resolve an asset only from a run root or this standalone repository.

    Run-local files take precedence over repository files. Absolute paths,
    traversal, ambiguous separators, and symlink escapes fail closed.
    """
    relative = _safe_relative_asset_path(path_value, field_name=field_name)
    roots: list[Path] = []
    if run_root is not None:
        roots.append(Path(run_root))
    roots.append(Path(repository_root))
    for root in roots:
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        if not _is_within(candidate, resolved_root):
            raise ValueError(f"{field_name} escapes an allowed local asset root.")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{field_name} was not found in the measurement run or local repository: "
        f"{relative.as_posix()!r}."
    )


def _file_asset_identity(
    path_value: object,
    *,
    field_name: str,
    run_root: str | Path | None,
    repository_root: str | Path,
) -> dict[str, str]:
    """Return the portable path and raw-byte digest bound into a component hash."""
    relative = _safe_relative_asset_path(path_value, field_name=field_name)
    resolved = resolve_file_backed_model_asset(
        relative,
        field_name=field_name,
        run_root=run_root,
        repository_root=repository_root,
    )
    return {
        "path": relative.as_posix(),
        "sha256": sha256(resolved.read_bytes()).hexdigest(),
    }


def _runtime_file_asset_references(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[str, str, object]]:
    """Discover model asset path fields and their owning hash components."""
    references: list[tuple[str, str, object]] = []
    if isinstance(value, Mapping):
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            child_path = (*path, key)
            normalized_key = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            normalized_path = "".join(
                character
                for part in child_path
                for character in part.casefold()
                if character.isalnum()
            )
            is_path_field = normalized_key.endswith(("path", "file"))
            component = next(
                (
                    name
                    for name in ("transport", "detector", "spectrum")
                    if name in normalized_path
                ),
                None,
            )
            if child is not None and is_path_field and component is not None:
                if not isinstance(child, (str, Path)):
                    raise TypeError(
                        f"runtime_config.{'.'.join(child_path)} must be a path string."
                    )
                references.append((".".join(child_path), component, child))
            else:
                references.extend(
                    _runtime_file_asset_references(child, path=child_path)
                )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            references.extend(
                _runtime_file_asset_references(child, path=(*path, f"[{index}]"))
            )
    return references


def _runtime_file_asset_identities(
    runtime_config: Mapping[str, object],
    *,
    run_root: str | Path | None,
    repository_root: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Resolve and hash runtime file assets grouped by model component."""
    grouped: dict[str, dict[str, dict[str, str]]] = {
        "transport": {},
        "detector": {},
        "spectrum": {},
    }
    for field_path, component, path_value in _runtime_file_asset_references(
        runtime_config
    ):
        grouped[component][field_path] = _file_asset_identity(
            path_value,
            field_name=f"runtime_config.{field_path}",
            run_root=run_root,
            repository_root=repository_root,
        )
    return grouped


def forward_model_component_payloads(
    *,
    runtime_config: Mapping[str, object],
    environment: Mapping[str, object],
    obstacle_layout_path: str | None,
    isotopes: tuple[str, ...],
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, dict[str, object]]:
    """Build the canonical configuration payload hashed for each model component.

    The hashes bind exactly the persisted, resolved configuration plus raw
    bytes for every file-backed obstacle/transport/detector/spectrum asset.
    """
    runtime = deepcopy(dict(runtime_config))
    environment_payload = deepcopy(dict(environment))
    detector = _selected(runtime, "detector", "aperture", "crystal", "housing")
    raw_line_table = line_resolved_shield_mu_by_isotope(
        isotopes=isotopes,
        normalize_line_intensities=True,
    )
    line_table = {
        isotope: [
            {name: float(entry[name]) for name in ("energy_keV", "weight", "fe", "pb")}
            for entry in raw_line_table.get(isotope, ())
        ]
        for isotope in isotopes
    }
    shield = line_table
    obstacle = {
        "environment": _selected(
            environment_payload,
            "obstacle",
            "blocked_cells",
            "grid_shape",
            "cell_size",
            "origin",
        ),
        "runtime_config": _selected(
            runtime,
            "obstacle",
            "material",
            "buildup",
            "source_extent",
        ),
        "layout_path": obstacle_layout_path,
    }
    if obstacle_layout_path is not None:
        obstacle["layout_asset"] = _file_asset_identity(
            obstacle_layout_path,
            field_name="obstacle_layout_path",
            run_root=run_root,
            repository_root=repository_root,
        )
    transport = {
        "runtime_config": _selected(
            runtime,
            "transport",
            "attenuation",
            "inverse_square",
            "buildup",
        )
    }
    spectrum: dict[str, object] = _line_energy_weight_table(line_table)
    payloads: dict[str, dict[str, object]] = {
        "detector": detector,
        "shield": shield,
        "environment": environment_payload,
        "obstacle": obstacle,
        "transport": transport,
        "spectrum": spectrum,
    }
    file_assets = _runtime_file_asset_identities(
        runtime,
        run_root=run_root,
        repository_root=repository_root,
    )
    for component, identities in file_assets.items():
        if identities:
            payloads[component]["file_assets"] = identities
    return payloads


def build_forward_model_manifest(
    *,
    runtime_config: Mapping[str, object],
    environment: Mapping[str, object],
    obstacle_layout_path: str | None,
    isotopes: tuple[str, ...],
    repository_commit: str,
    resolved_config_sha256: str,
    source_rate_model: str,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, object]:
    """Return a complete schema-v1 forward-model identity manifest."""
    normalized_rate_model = str(source_rate_model).strip().lower()
    if normalized_rate_model != SOURCE_RATE_MODEL:
        raise ValueError(
            f"source_rate_model must be {SOURCE_RATE_MODEL!r}; "
            f"got {source_rate_model!r}."
        )
    component_payloads = forward_model_component_payloads(
        runtime_config=runtime_config,
        environment=environment,
        obstacle_layout_path=obstacle_layout_path,
        isotopes=isotopes,
        run_root=run_root,
        repository_root=repository_root,
    )
    line_mu_by_isotope = {
        isotope: [
            {name: float(entry[name]) for name in ("energy_keV", "weight", "fe", "pb")}
            for entry in line_resolved_shield_mu_by_isotope(
                isotopes=isotopes,
                normalize_line_intensities=True,
            ).get(isotope, ())
        ]
        for isotope in isotopes
    }
    runtime = dict(runtime_config)
    environment_mapping = dict(environment)
    identifiers = {
        "detector": _identifier(
            (runtime,),
            ("detector_model_id", "detector_model_identifier"),
            "local_detector_observation_geometry.v1",
        ),
        "shield": _identifier(
            (runtime,),
            ("shield_model_id", "shield_model_identifier"),
            "rotating_nested_octant_shield.v1",
        ),
        "environment": _identifier(
            (environment_mapping, runtime),
            ("environment_model_id", "environment_id", "environment_mode"),
            "rectangular_room_surface_environment.v1",
        ),
        "obstacle": _identifier(
            (environment_mapping, runtime),
            ("obstacle_model_id", "obstacle_layout_id"),
            (
                str(obstacle_layout_path)
                if obstacle_layout_path is not None
                else "embedded_or_empty_obstacle_grid.v1"
            ),
        ),
        "transport": _identifier(
            (runtime,),
            ("transport_model_id", "transport_response_model_id"),
            "continuous_inverse_square_shield_obstacle_transport.v1",
        ),
        "spectrum": _identifier(
            (runtime,),
            ("spectrum_model_id", "spectrum_response_model_id"),
            "line_resolved_detector_spectrum_response.v1",
        ),
    }
    model_identifiers = {
        name: {
            "id": identifiers[name],
            "sha256": canonical_json_sha256(component_payloads[name]),
        }
        for name in REQUIRED_MODEL_NAMES
    }
    return {
        "schema_version": FORWARD_MODEL_MANIFEST_SCHEMA_VERSION,
        "repository_commit": str(repository_commit).strip(),
        "resolved_config_sha256": str(resolved_config_sha256).lower(),
        "source_rate_model": SOURCE_RATE_MODEL,
        "source_rate_semantics": deepcopy(SOURCE_RATE_SEMANTICS),
        "units": deepcopy(CANONICAL_UNITS),
        "response_semantics": deepcopy(RESPONSE_SEMANTICS),
        "line_mu_by_isotope": line_mu_by_isotope,
        "model_identifiers": model_identifiers,
    }


def _sha256(value: object, *, name: str) -> str:
    """Return a validated lowercase SHA-256 string."""
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return normalized


def _validate_registered_conformance_manifest(
    payload: dict[str, object],
    *,
    isotopes: tuple[str, ...],
    repository_commit: str,
    resolved_config_sha256: str,
) -> dict[str, object]:
    """Fail closed while binding the versioned fixture model to local physics."""
    expected_fields = {
        "schema_version",
        "forward_model_id",
        "repository_commit",
        "resolved_config_sha256",
        "source_rate_model",
        "source_rate_semantics",
        "model_identifiers",
        "units",
        "response_semantics",
        "shield_program",
        "line_mu_by_isotope",
    }
    if set(payload) != expected_fields:
        raise ValueError(
            "Registered forward-model manifest fields must be exactly "
            f"{sorted(expected_fields)}."
        )
    if payload.get("forward_model_id") != CONFORMANCE_FORWARD_MODEL_ID:
        raise ValueError("Unknown registered forward_model_id.")
    expected_line_table = registered_conformance_line_mu_by_isotope()
    if tuple(isotopes) != tuple(expected_line_table):
        raise ValueError(
            "Registered forward-model isotope order is incompatible with the fixture."
        )
    if payload.get("repository_commit") != str(repository_commit).strip():
        raise ValueError("forward-model repository_commit does not match the log.")
    if payload.get("resolved_config_sha256") != str(resolved_config_sha256).lower():
        raise ValueError("forward-model resolved_config_sha256 does not match the log.")
    if payload.get("shield_program") != _CONFORMANCE_SHIELD_PROGRAM:
        raise ValueError("Registered forward-model shield_program is incompatible.")
    if payload.get("line_mu_by_isotope") != expected_line_table:
        raise ValueError("Registered forward-model line_mu_by_isotope is incompatible.")
    expected_spectrum_identity = _line_energy_weight_table(expected_line_table)
    if (
        canonical_json_sha256(expected_line_table)
        != (CONFORMANCE_MODEL_IDENTIFIERS["shield"]["sha256"])
        or canonical_json_sha256(expected_spectrum_identity)
        != (CONFORMANCE_MODEL_IDENTIFIERS["spectrum"]["sha256"])
    ):
        raise RuntimeError(
            "Local registered line tables no longer match their bound model hashes."
        )
    raw_identifiers = payload.get("model_identifiers")
    if not isinstance(raw_identifiers, Mapping):
        raise ValueError("forward_model_manifest.model_identifiers must be an object.")
    if set(raw_identifiers) != set(REQUIRED_MODEL_NAMES):
        raise ValueError(
            "forward_model_manifest.model_identifiers must contain exactly "
            f"{list(REQUIRED_MODEL_NAMES)}."
        )
    normalized_identifiers: dict[str, dict[str, str]] = {}
    for name in REQUIRED_MODEL_NAMES:
        entry = raw_identifiers[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"model_identifiers.{name} must be an object.")
        identifier = str(entry.get("id", "")).strip()
        digest = _sha256(
            entry.get("sha256"),
            name=f"model_identifiers.{name}.sha256",
        )
        expected_entry = CONFORMANCE_MODEL_IDENTIFIERS[name]
        if identifier != expected_entry["id"] or digest != expected_entry["sha256"]:
            raise ValueError(
                f"Forward-model compatibility error for {name}: identifier or "
                "SHA-256 differs from the registered conformance model."
            )
        normalized_identifiers[name] = {"id": identifier, "sha256": digest}
    payload["model_identifiers"] = normalized_identifiers
    return payload


def validate_forward_model_manifest(
    manifest: Mapping[str, object],
    *,
    runtime_config: Mapping[str, object],
    environment: Mapping[str, object],
    obstacle_layout_path: str | None,
    isotopes: tuple[str, ...],
    repository_commit: str,
    resolved_config_sha256: str,
    source_rate_model: str,
    run_root: str | Path | None = None,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> dict[str, object]:
    """Validate a manifest and prove it matches the replay configuration."""
    if not isinstance(manifest, Mapping):
        raise TypeError("forward_model_manifest must be a mapping.")
    if str(source_rate_model).strip().lower() != SOURCE_RATE_MODEL:
        raise ValueError("run-manifest source_rate_model is incompatible.")
    payload = deepcopy(dict(manifest))
    if payload.get("schema_version") != FORWARD_MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported forward-model manifest schema_version; expected 1."
        )
    if payload.get("source_rate_model") != SOURCE_RATE_MODEL:
        raise ValueError("forward-model source_rate_model is incompatible.")
    if payload.get("source_rate_semantics") != SOURCE_RATE_SEMANTICS:
        raise ValueError("forward-model source_rate_semantics is incompatible.")
    if payload.get("repository_commit") != str(repository_commit).strip():
        raise ValueError("forward-model repository_commit does not match the log.")
    if payload.get("units") != CANONICAL_UNITS:
        raise ValueError("forward-model units are incompatible.")
    response_semantics = payload.get("response_semantics")
    if response_semantics != RESPONSE_SEMANTICS:
        raise ValueError("forward-model response_semantics are incompatible.")
    _sha256(
        payload.get("resolved_config_sha256"),
        name="forward_model_manifest.resolved_config_sha256",
    )
    registered_id = payload.get("forward_model_id")
    if registered_id is not None:
        if registered_id != CONFORMANCE_FORWARD_MODEL_ID:
            raise ValueError(
                f"Unknown registered forward_model_id {registered_id!r}; refusing fallback."
            )
        if obstacle_layout_path is not None or _runtime_file_asset_references(
            runtime_config
        ):
            raise ValueError(
                "Registered conformance manifests cannot reference file-backed assets."
            )
        return _validate_registered_conformance_manifest(
            payload,
            isotopes=isotopes,
            repository_commit=repository_commit,
            resolved_config_sha256=resolved_config_sha256,
        )
    expected = build_forward_model_manifest(
        runtime_config=runtime_config,
        environment=environment,
        obstacle_layout_path=obstacle_layout_path,
        isotopes=isotopes,
        repository_commit=repository_commit,
        resolved_config_sha256=resolved_config_sha256,
        source_rate_model=source_rate_model,
        run_root=run_root,
        repository_root=repository_root,
    )
    for field in (
        "resolved_config_sha256",
        "source_rate_model",
        "source_rate_semantics",
        "units",
        "response_semantics",
        "line_mu_by_isotope",
    ):
        if payload.get(field) != expected[field]:
            raise ValueError(
                f"forward_model_manifest {field} does not match the resolved replay model."
            )
    raw_identifiers = payload.get("model_identifiers")
    if not isinstance(raw_identifiers, Mapping):
        raise ValueError("forward_model_manifest.model_identifiers must be an object.")
    if set(raw_identifiers) != set(REQUIRED_MODEL_NAMES):
        raise ValueError(
            "forward_model_manifest.model_identifiers must contain exactly "
            f"{list(REQUIRED_MODEL_NAMES)}."
        )
    expected_identifiers = expected["model_identifiers"]
    assert isinstance(expected_identifiers, dict)
    normalized_identifiers: dict[str, dict[str, str]] = {}
    for name in REQUIRED_MODEL_NAMES:
        entry = raw_identifiers[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"model_identifiers.{name} must be an object.")
        identifier = str(entry.get("id", "")).strip()
        if not identifier:
            raise ValueError(f"model_identifiers.{name}.id must be non-empty.")
        digest = _sha256(
            entry.get("sha256"),
            name=f"model_identifiers.{name}.sha256",
        )
        expected_entry = expected_identifiers[name]
        if identifier != expected_entry["id"] or digest != expected_entry["sha256"]:
            raise ValueError(
                f"Forward-model compatibility error for {name}: identifier or "
                "SHA-256 differs from the resolved replay model."
            )
        normalized_identifiers[name] = {"id": identifier, "sha256": digest}
    payload["model_identifiers"] = normalized_identifiers
    payload["resolved_config_sha256"] = str(payload["resolved_config_sha256"])
    return payload


def load_forward_model_manifest(path: str | Path) -> dict[str, object]:
    """Read one forward-model manifest as a JSON object."""
    import json

    target = Path(path)
    try:
        payload: Any = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read valid forward-model JSON from {target}."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("forward_model_manifest.json must contain a JSON object.")
    return payload


__all__ = [
    "FORWARD_MODEL_MANIFEST_SCHEMA_VERSION",
    "CANONICAL_UNITS",
    "CONFORMANCE_FORWARD_MODEL_ID",
    "CONFORMANCE_MODEL_IDENTIFIERS",
    "REQUIRED_MODEL_NAMES",
    "SOURCE_RATE_MODEL",
    "SOURCE_RATE_SEMANTICS",
    "RESPONSE_SEMANTICS",
    "build_forward_model_manifest",
    "forward_model_component_payloads",
    "load_forward_model_manifest",
    "registered_conformance_line_mu_by_isotope",
    "resolve_file_backed_model_asset",
    "validate_forward_model_manifest",
]
