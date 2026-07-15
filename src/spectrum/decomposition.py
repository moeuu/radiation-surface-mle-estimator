"""Peak stripping and intensity estimation with overlap handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple


from spectrum.library import Nuclide


@dataclass(frozen=True)
class Peak:
    """Hold a detected peak's energy and area."""

    energy_keV: float
    area: float


def reference_line(nuclide: Nuclide, efficiency_fn: Callable[[float], float] | None = None) -> float:
    """Return the most intense line energy as the reference line."""
    if not nuclide.lines:
        raise ValueError("Nuclide has no line information.")
    if efficiency_fn is None:
        return max(nuclide.lines, key=lambda line: line.intensity).energy_keV
    return max(
        nuclide.lines,
        key=lambda line: line.intensity * float(efficiency_fn(line.energy_keV)),
    ).energy_keV


def intensity_ratios(
    nuclide: Nuclide, efficiency_fn: Callable[[float], float] | None = None
) -> Dict[float, float]:
    """
    Compute line intensity ratios relative to the reference line.

    Returns:
        {line_energy_keV: ratio}
    """
    if efficiency_fn is None:
        ref_intensity = max(line.intensity for line in nuclide.lines)
    else:
        ref_intensity = max(
            line.intensity * float(efficiency_fn(line.energy_keV))
            for line in nuclide.lines
        )
    ratios: Dict[float, float] = {}
    for line in nuclide.lines:
        weight = line.intensity
        if efficiency_fn is not None:
            weight *= float(efficiency_fn(line.energy_keV))
        ratios[line.energy_keV] = weight / ref_intensity if ref_intensity > 0 else 0.0
    return ratios


def match_peaks_to_library(
    peaks: Iterable[Peak],
    library: Dict[str, Nuclide],
    tolerance_keV: float = 5.0,
    efficiency_fn: Callable[[float], float] | None = None,
) -> Dict[str, Peak]:
    """
    Match detected peaks to library reference lines.

    Args:
        peaks: Detected peaks.
        library: Nuclide library.
        tolerance_keV: Allowed energy mismatch.

    Returns:
        {isotope: matched_peak}
    """
    matches: Dict[str, Peak] = {}
    for iso, nuclide in library.items():
        ref_energy = reference_line(nuclide, efficiency_fn=efficiency_fn)
        closest: Peak | None = None
        min_diff = tolerance_keV
        for pk in peaks:
            diff = abs(pk.energy_keV - ref_energy)
            if diff <= min_diff:
                min_diff = diff
                closest = pk
        if closest is not None:
            matches[iso] = closest
    return matches


def strip_overlaps(
    peaks: Iterable[Peak],
    library: Dict[str, Nuclide],
    tolerance_keV: float = 5.0,
    efficiency_fn: Callable[[float], float] | None = None,
) -> Tuple[Dict[str, float], List[Peak]]:
    """
    Strip overlapping peaks using library ratios and estimate reference areas.

    Args:
        peaks: Detected peaks.
        library: Nuclide library.
        tolerance_keV: Matching tolerance.

    Returns:
        (reference area per isotope, stripped peaks)
    """
    peak_list = list(peaks)
    matches = match_peaks_to_library(
        peak_list, library, tolerance_keV=tolerance_keV, efficiency_fn=efficiency_fn
    )
    stripped_peaks: List[Peak] = peak_list.copy()
    ref_areas: Dict[str, float] = {}

    for iso, ref_peak in matches.items():
        ref_areas[iso] = ref_peak.area
        ratios = intensity_ratios(library[iso], efficiency_fn=efficiency_fn)
        for energy, ratio in ratios.items():
            if energy == reference_line(library[iso], efficiency_fn=efficiency_fn):
                continue
            # Find the target peak and subtract expected contribution.
            target = _find_peak(stripped_peaks, energy, tolerance_keV)
            if target is None:
                continue
            expected = ref_peak.area * ratio
            updated_area = max(target.area - expected, 0.0)
            stripped_peaks = _replace_peak(stripped_peaks, target, updated_area)
    return ref_areas, stripped_peaks


def _find_peak(peaks: List[Peak], energy: float, tolerance_keV: float) -> Peak | None:
    """Find the closest peak within the given energy tolerance."""
    closest: Peak | None = None
    min_diff = tolerance_keV
    for pk in peaks:
        diff = abs(pk.energy_keV - energy)
        if diff <= min_diff:
            min_diff = diff
            closest = pk
    return closest


def _replace_peak(peaks: List[Peak], target: Peak, new_area: float) -> List[Peak]:
    """Replace a peak's area in the list and return the updated list."""
    updated: List[Peak] = []
    for pk in peaks:
        if pk is target:
            updated.append(Peak(energy_keV=pk.energy_keV, area=new_area))
        else:
            updated.append(pk)
    return updated
