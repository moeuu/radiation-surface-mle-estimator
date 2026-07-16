"""Regenerate the deterministic observations member of the shared log fixture."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Sequence
import zipfile

import numpy as np
from numpy.typing import NDArray

from runtime.forward_model_manifest import registered_conformance_line_mu_by_isotope


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "fixtures"
    / "shared_measurement_log_v1"
    / "measurement_log"
    / "observations.npz"
)
ISOTOPE_ORDER = ("Cs-137", "Co-60", "Eu-154")


def _spectrum_from_isotope_counts(
    isotope_counts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Allocate net counts to 100-keV bins using production line weights."""
    energy_edges = np.linspace(0.0, 1600.0, 17, dtype=np.float64)
    background = np.asarray(
        [8, 7, 6, 5, 5, 5, 6, 5, 5, 5, 6, 7, 7, 6, 6, 5],
        dtype=np.float64,
    )
    spectra = np.broadcast_to(background, (isotope_counts.shape[0], 16)).copy()
    line_table = registered_conformance_line_mu_by_isotope()
    for isotope_index, isotope in enumerate(ISOTOPE_ORDER):
        for line in line_table[isotope]:
            bin_index = int(
                np.searchsorted(
                    energy_edges,
                    float(line["energy_keV"]),
                    side="right",
                )
                - 1
            )
            if not 0 <= bin_index < spectra.shape[1]:
                raise ValueError("A production line lies outside the fixture bins.")
            spectra[:, bin_index] += (
                float(line["weight"]) * isotope_counts[:, isotope_index]
            )
    return spectra


def fixture_arrays() -> dict[str, NDArray[np.generic]]:
    """Return the exact typed arrays of the provider-neutral shared run."""
    isotope_counts = np.asarray(
        [
            [260.0, 95.0, 150.0],
            [215.0, 120.0, 135.0],
            [175.0, 145.0, 118.0],
            [190.0, 170.0, 125.0],
            [155.0, 205.0, 112.0],
            [130.0, 230.0, 100.0],
            [120.0, 180.0, 245.0],
            [100.0, 160.0, 275.0],
            [85.0, 145.0, 305.0],
            [140.0, 260.0, 105.0],
            [115.0, 295.0, 92.0],
            [95.0, 330.0, 80.0],
        ],
        dtype=np.float64,
    )
    spectra = _spectrum_from_isotope_counts(isotope_counts)
    covariance = np.zeros((12, 3, 3), dtype=np.float64)
    for row_index in range(12):
        diagonal = np.maximum(isotope_counts[row_index], 1.0) * 1.2
        covariance[row_index] = np.diag(diagonal)
        for left in range(3):
            for right in range(left + 1, 3):
                value = 0.03 * float(np.sqrt(diagonal[left] * diagonal[right]))
                covariance[row_index, left, right] = value
                covariance[row_index, right, left] = value
    poses = np.repeat(
        np.asarray(
            [
                [0.8, 0.8, 0.45],
                [4.8, 0.9, 1.10],
                [2.0, 4.8, 1.80],
                [5.1, 4.7, 2.55],
            ],
            dtype=np.float64,
        ),
        3,
        axis=0,
    )
    return {
        "step_id": np.arange(12, dtype=np.int64),
        "action_id": np.arange(12, dtype=np.int64),
        "station_id": np.repeat(np.arange(4, dtype=np.int64), 3),
        "detector_pose_xyz": poses,
        "detector_quat_wxyz": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (12, 1),
        ),
        "fe_orientation_index": np.asarray(
            [0, 2, 5, 1, 3, 7, 0, 4, 6, 1, 3, 7],
            dtype=np.int64,
        ),
        "pb_orientation_index": np.asarray(
            [0, 4, 7, 6, 1, 3, 7, 2, 5, 1, 6, 0],
            dtype=np.int64,
        ),
        "live_time_s": np.full(12, 10.0, dtype=np.float64),
        "travel_time_s": np.asarray(
            [0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 5.2, 0.0, 0.0, 4.8, 0.0, 0.0],
            dtype=np.float64,
        ),
        "shield_actuation_time_s": np.asarray(
            [0.0, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6],
            dtype=np.float64,
        ),
        "energy_bin_edges_keV": np.linspace(0.0, 1600.0, 17, dtype=np.float64),
        "spectrum_counts": spectra,
        "spectrum_variance": spectra + 1.0,
        "spectrum_variance_present": np.ones(12, dtype=np.bool_),
        "isotope_counts": isotope_counts,
        "isotope_counts_present": np.ones((12, 3), dtype=np.bool_),
        "isotope_counts_record_present": np.ones(12, dtype=np.bool_),
        "isotope_count_covariance": covariance,
        "isotope_count_covariance_present": np.ones((12, 3, 3), dtype=np.bool_),
        "isotope_count_covariance_record_present": np.ones(12, dtype=np.bool_),
    }


def _npy_bytes(array: NDArray[np.generic]) -> bytes:
    """Return deterministic version-2 NPY bytes without pickle."""
    buffer = BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def create_fixture(output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write the exact observations NPZ without replacing an existing file."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing fixture: {target}")
    with target.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name, array in fixture_arrays().items():
                entry = zipfile.ZipInfo(
                    f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_STORED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, _npy_bytes(array))
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the output path and regenerate the shared binary observation member."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(None if argv is None else list(argv))
    print(create_fixture(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
