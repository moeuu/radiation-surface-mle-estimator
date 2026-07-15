"""Define and expose a small nuclide library used by the spectrum pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Candidate isotopes used for peak-based analysis.
ANALYSIS_ISOTOPES: List[str] = ["Cs-137", "Co-60", "Eu-154"]


@dataclass(frozen=True)
class NuclideLine:
    """Represent a single gamma line energy and relative intensity."""

    energy_keV: float
    intensity: float


@dataclass(frozen=True)
class Nuclide:
    """Hold nuclide metadata and its representative gamma lines."""

    name: str
    lines: List[NuclideLine]
    representative_energy_keV: float


# Detection key lines (keV) used for robust isotope identification.
KEY_LINES_KEV: Dict[str, List[float]] = {
    "Cs-137": [661.657],
    "Co-60": [1173.228, 1332.492],
    "Eu-154": [723.30, 873.19, 996.26, 1274.43],
}

# Analysis lines (keV) used in Chapter 4 (Table 4.1).
ANALYSIS_LINES_KEV: Dict[str, List[float]] = {
    "Cs-137": [662.0],
    "Co-60": [1173.0, 1332.0],
    "Eu-154": [723.3, 873.2, 996.3, 1274.5, 1494.0, 1596.5],
}

# Table 4.1 branching ratios (fallback when library intensities are missing).
ANALYSIS_LINE_INTENSITIES: Dict[str, Dict[float, float]] = {
    "Cs-137": {662.0: 0.85},
    "Co-60": {1173.0: 0.5, 1332.0: 0.5},
    "Eu-154": {
        723.3: 0.25,
        873.2: 0.14,
        996.3: 0.14,
        1274.5: 0.45,
        1494.0: 0.01,
        1596.5: 0.02,
    },
}

_MISSING_ANALYSIS_INTENSITY: set[tuple[str, float]] = set()


def get_detection_lines_keV(isotope: str) -> List[float]:
    """Return detection key lines (keV) for the requested isotope."""
    return list(KEY_LINES_KEV.get(isotope, []))


def get_analysis_lines_keV(isotope: str, *, max_energy_keV: float = 1500.0) -> List[float]:
    """
    Return analysis lines (keV) for Table 4.1, filtered by max energy.

    Args:
        isotope: Isotope name.
        max_energy_keV: Maximum energy to include.

    Returns:
        List of analysis line energies in keV.
    """
    lines = ANALYSIS_LINES_KEV.get(isotope, [])
    return [float(line) for line in lines if float(line) <= float(max_energy_keV)]


def get_analysis_lines_with_intensity(
    isotope: str,
    library: Dict[str, Nuclide],
    *,
    max_energy_keV: float = 1500.0,
    tol_keV: float = 2.0,
) -> List[Tuple[float, float]]:
    """
    Return analysis lines with intensities matched from the nuclide library.

    Args:
        isotope: Isotope name.
        library: Nuclide library mapping.
        max_energy_keV: Maximum energy to include.
        tol_keV: Matching tolerance in keV.

    Returns:
        List of (energy_keV, intensity) tuples.
    """
    energies = get_analysis_lines_keV(isotope, max_energy_keV=max_energy_keV)
    fallback_intensities = ANALYSIS_LINE_INTENSITIES.get(isotope, {})
    nuclide = library.get(isotope)
    if nuclide is None:
        for energy in energies:
            key = (isotope, float(energy))
            if key not in _MISSING_ANALYSIS_INTENSITY:
                logger.warning("Missing nuclide %s for analysis line %.1f keV; using intensity=1.0", isotope, energy)
                _MISSING_ANALYSIS_INTENSITY.add(key)
        return [(float(energy), float(fallback_intensities.get(float(energy), 1.0))) for energy in energies]
    matched: list[tuple[float, float]] = []
    for energy in energies:
        closest: NuclideLine | None = None
        best_diff = float("inf")
        for line in nuclide.lines:
            diff = abs(float(line.energy_keV) - float(energy))
            if diff < best_diff:
                best_diff = diff
                closest = line
        if closest is None or best_diff > float(tol_keV):
            key = (isotope, float(energy))
            if key not in _MISSING_ANALYSIS_INTENSITY:
                logger.warning(
                    "Missing intensity for %s analysis line %.1f keV; using intensity=1.0",
                    isotope,
                    energy,
                )
                _MISSING_ANALYSIS_INTENSITY.add(key)
            matched.append((float(energy), float(fallback_intensities.get(float(energy), 1.0))))
        else:
            matched.append((float(energy), float(closest.intensity)))
    return matched


def default_library() -> Dict[str, Nuclide]:
    """Return a default library with Cs-137, Co-60, and Eu-154 lines."""
    return {
        "Cs-137": Nuclide(
            name="Cs-137",
            lines=[NuclideLine(energy_keV=662.0, intensity=0.85)],
            representative_energy_keV=662.0,
        ),
        "Co-60": Nuclide(
            name="Co-60",
            lines=[
                NuclideLine(energy_keV=1173.0, intensity=0.5),
                NuclideLine(energy_keV=1332.0, intensity=0.5),
            ],
            representative_energy_keV=1250.0,
        ),
        "Eu-154": Nuclide(
            name="Eu-154",
            lines=[
                NuclideLine(energy_keV=723.3, intensity=0.25),
                NuclideLine(energy_keV=873.2, intensity=0.14),
                NuclideLine(energy_keV=996.3, intensity=0.14),
                NuclideLine(energy_keV=1274.5, intensity=0.45),
                NuclideLine(energy_keV=1494.0, intensity=0.01),
                NuclideLine(energy_keV=1596.5, intensity=0.02),
            ],
            representative_energy_keV=1274.5,
        ),
    }
