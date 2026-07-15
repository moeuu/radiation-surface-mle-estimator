"""Shield-program ablation policies for RA-L comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineShieldProgram:
    """Represent a baseline shield program selected outside DSS-PP."""

    name: str
    pair_ids: tuple[int, ...]


def _read_policy_name(policy_config: Mapping[str, Any] | str | None) -> str:
    """Return a normalized baseline shield-policy name."""
    if policy_config is None:
        return ""
    if isinstance(policy_config, str):
        return policy_config.strip().lower()
    return str(policy_config.get("name", "")).strip().lower()


def _read_int(
    policy_config: Mapping[str, Any] | str | None,
    key: str,
    default: int,
) -> int:
    """Read an integer setting from a shield-policy payload."""
    if not isinstance(policy_config, Mapping):
        return int(default)
    return int(policy_config.get(key, default))


def select_baseline_shield_program(
    policy_config: Mapping[str, Any] | str | None,
    *,
    total_pairs: int,
    program_length: int,
    pose_index: int,
    current_pair_id: int | None = None,
) -> BaselineShieldProgram | None:
    """Return a baseline shield program, or None when no baseline policy is active."""
    policy = _read_policy_name(policy_config)
    if policy in {"", "none", "proposed", "dss_pp"}:
        return None
    total = max(1, int(total_pairs))
    length = max(1, int(program_length))
    if policy in {"fixed", "fixed_shield"}:
        fixed_pair = _read_int(policy_config, "fixed_pair_id", 0) % total
        return BaselineShieldProgram(
            name=f"fixed_shield_{fixed_pair}",
            pair_ids=tuple(fixed_pair for _ in range(length)),
        )
    if policy in {"round_robin", "round-robin", "round_robin_shield"}:
        start = _read_int(
            policy_config,
            "start_pair_id",
            0 if current_pair_id is None else int(current_pair_id) + 1,
        )
        if isinstance(policy_config, Mapping) and bool(
            policy_config.get("advance_by_pose", True)
        ):
            start += int(pose_index) * length
        return BaselineShieldProgram(
            name="round_robin_shield",
            pair_ids=tuple((start + idx) % total for idx in range(length)),
        )
    raise ValueError(f"Unknown baseline_shield_policy: {policy}")
