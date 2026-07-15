"""Material presets and attenuation helpers for the Isaac Sim bridge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MaterialPreset:
    """Describe a reusable material preset for attenuation fallback."""

    name: str
    density_g_cm3: float | None = None
    mass_att_by_isotope_cm2_g: dict[str, float] = field(default_factory=dict)
    composition_by_mass: dict[str, float] = field(default_factory=dict)


# XCOM total mass attenuation coefficients in cm^2/g for the gamma-energy
# range used by the runtime isotopes.  These values are tabulated by NIST and
# match the material physics used by the Geant4 sidecar more closely than the
# older three-point engineering approximations.
ELEMENTAL_MASS_ATT_CURVES_CM2_G: dict[str, dict[float, float]] = {
    "H": {
        500.0: 0.17290,
        600.0: 0.15990,
        800.0: 0.14050,
        1000.0: 0.12630,
        1250.0: 0.11290,
        1500.0: 0.10270,
        2000.0: 0.08769,
    },
    "C": {
        500.0: 0.08715,
        600.0: 0.08058,
        800.0: 0.07076,
        1000.0: 0.06361,
        1250.0: 0.05690,
        1500.0: 0.05179,
        2000.0: 0.04442,
    },
    "N": {
        500.0: 0.08719,
        600.0: 0.08063,
        800.0: 0.07081,
        1000.0: 0.06364,
        1250.0: 0.05693,
        1500.0: 0.05180,
        2000.0: 0.04450,
    },
    "O": {
        500.0: 0.08729,
        600.0: 0.08070,
        800.0: 0.07087,
        1000.0: 0.06372,
        1250.0: 0.05697,
        1500.0: 0.05185,
        2000.0: 0.04459,
    },
    "Al": {
        500.0: 0.08445,
        600.0: 0.07802,
        800.0: 0.06841,
        1000.0: 0.06146,
        1250.0: 0.05496,
        1500.0: 0.05006,
        2000.0: 0.04324,
    },
    "Si": {
        500.0: 0.08748,
        600.0: 0.08077,
        800.0: 0.07082,
        1000.0: 0.06361,
        1250.0: 0.05688,
        1500.0: 0.05183,
        2000.0: 0.04480,
    },
    "Ca": {
        500.0: 0.08851,
        600.0: 0.08148,
        800.0: 0.07122,
        1000.0: 0.06388,
        1250.0: 0.05709,
        1500.0: 0.05207,
        2000.0: 0.04524,
    },
    "Cr": {
        500.0: 0.08281,
        600.0: 0.07598,
        800.0: 0.06620,
        1000.0: 0.05930,
        1250.0: 0.05295,
        1500.0: 0.04832,
        2000.0: 0.04213,
    },
    "Fe": {
        500.0: 0.08414,
        600.0: 0.07704,
        800.0: 0.06699,
        1000.0: 0.05995,
        1250.0: 0.05350,
        1500.0: 0.04883,
        2000.0: 0.04265,
    },
    "Ni": {
        500.0: 0.08698,
        600.0: 0.07944,
        800.0: 0.06891,
        1000.0: 0.06160,
        1250.0: 0.05494,
        1500.0: 0.05015,
        2000.0: 0.04387,
    },
    "Ar": {
        500.0: 0.07958,
        600.0: 0.07335,
        800.0: 0.06419,
        1000.0: 0.05762,
        1250.0: 0.05150,
        1500.0: 0.04695,
        2000.0: 0.04074,
    },
    "Pb": {
        500.0: 0.16140,
        600.0: 0.12480,
        800.0: 0.08870,
        1000.0: 0.07102,
        1250.0: 0.05876,
        1500.0: 0.05222,
        2000.0: 0.04606,
    },
}

ELEMENTAL_MASS_ATT_CM2_G: dict[str, dict[str, float]] = {
    "H": {"Cs-137": 0.153886, "Co-60": 0.113291, "Eu-154": 0.125769},
    "C": {"Cs-137": 0.077536, "Co-60": 0.057095, "Eu-154": 0.063368},
    "N": {"Cs-137": 0.077586, "Co-60": 0.057122, "Eu-154": 0.063403},
    "O": {"Cs-137": 0.077653, "Co-60": 0.057170, "Eu-154": 0.063460},
    "Al": {"Cs-137": 0.075041, "Co-60": 0.055157, "Eu-154": 0.061248},
    "Si": {"Cs-137": 0.077685, "Co-60": 0.057088, "Eu-154": 0.063397},
    "Ca": {"Cs-137": 0.078299, "Co-60": 0.057312, "Eu-154": 0.063715},
    "Cr": {"Cs-137": 0.072948, "Co-60": 0.053169, "Eu-154": 0.059180},
    "Fe": {"Cs-137": 0.073924, "Co-60": 0.053727, "Eu-154": 0.059853},
    "Ni": {"Cs-137": 0.076176, "Co-60": 0.055180, "Eu-154": 0.061531},
    "Ar": {"Cs-137": 0.070510, "Co-60": 0.051696, "Eu-154": 0.057445},
    "Pb": {"Cs-137": 0.113609, "Co-60": 0.059575, "Eu-154": 0.074094},
}

MATERIAL_PRESETS: dict[str, MaterialPreset] = {
    "air": MaterialPreset(
        name="air",
        density_g_cm3=0.001225,
        composition_by_mass={"N": 0.755, "O": 0.232, "Ar": 0.013},
    ),
    "water": MaterialPreset(
        name="water",
        density_g_cm3=1.0,
        composition_by_mass={"H": 0.1119, "O": 0.8881},
    ),
    "concrete": MaterialPreset(
        name="concrete",
        density_g_cm3=2.3,
        composition_by_mass={"O": 0.525, "Si": 0.325, "Ca": 0.090, "Al": 0.060},
    ),
    "aluminum": MaterialPreset(
        name="aluminum",
        density_g_cm3=2.7,
        composition_by_mass={"Al": 1.0},
    ),
    "iron": MaterialPreset(
        name="iron",
        density_g_cm3=7.87,
        composition_by_mass={"Fe": 1.0},
    ),
    "steel": MaterialPreset(
        name="steel",
        density_g_cm3=7.85,
        composition_by_mass={"Fe": 0.98, "C": 0.02},
    ),
    "stainless_steel": MaterialPreset(
        name="stainless_steel",
        density_g_cm3=8.0,
        composition_by_mass={"Fe": 0.70, "Cr": 0.19, "Ni": 0.10, "C": 0.01},
    ),
    "lead": MaterialPreset(
        name="lead",
        density_g_cm3=11.34,
        composition_by_mass={"Pb": 1.0},
    ),
}

MATERIAL_ALIASES: dict[str, str] = {
    "al": "aluminum",
    "alu": "aluminum",
    "aluminium": "aluminum",
    "fe": "iron",
    "iron": "iron",
    "lead": "lead",
    "pb": "lead",
    "ss": "stainless_steel",
    "ss304": "stainless_steel",
    "stainless": "stainless_steel",
}

ELEMENT_ALIASES: dict[str, str] = {
    "al": "Al",
    "aluminum": "Al",
    "aluminium": "Al",
    "ar": "Ar",
    "argon": "Ar",
    "c": "C",
    "carbon": "C",
    "ca": "Ca",
    "calcium": "Ca",
    "cr": "Cr",
    "chromium": "Cr",
    "fe": "Fe",
    "iron": "Fe",
    "h": "H",
    "hydrogen": "H",
    "n": "N",
    "nitrogen": "N",
    "ni": "Ni",
    "nickel": "Ni",
    "o": "O",
    "oxygen": "O",
    "pb": "Pb",
    "lead": "Pb",
    "si": "Si",
    "silicon": "Si",
}


def normalize_material_name(name: str) -> str:
    """Normalize a material identifier into a preset lookup key."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return MATERIAL_ALIASES.get(normalized, normalized)


def normalize_element_name(name: str) -> str | None:
    """Normalize an element token into its symbol."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return ELEMENT_ALIASES.get(normalized)


def parse_composition_string(raw_value: str) -> dict[str, float]:
    """Parse a simple mass-fraction composition string."""
    composition: dict[str, float] = {}
    text = str(raw_value).strip()
    if not text:
        return composition
    normalized = text.replace(";", ",")
    for part in normalized.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(f"Invalid composition entry: {item!r}")
        element = normalize_element_name(key)
        if element is None:
            continue
        composition[element] = float(value)
    return composition


def normalize_composition_by_mass(raw_value: dict[str, float] | str | None) -> dict[str, float]:
    """Normalize a composition dictionary or composition string."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, str):
        return parse_composition_string(raw_value)
    composition: dict[str, float] = {}
    for key, value in raw_value.items():
        element = normalize_element_name(str(key))
        if element is None:
            continue
        composition[element] = float(value)
    return composition


def resolve_material_preset(name: str | None) -> MaterialPreset | None:
    """Resolve a normalized preset entry by material name."""
    if name in (None, ""):
        return None
    return MATERIAL_PRESETS.get(normalize_material_name(str(name)))


def composition_mass_attenuation(
    composition_by_mass: dict[str, float] | str | None,
    isotope: str,
) -> float | None:
    """Return a mixture mass attenuation coefficient from elemental fractions."""
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for element, weight in composition.items():
        table = ELEMENTAL_MASS_ATT_CM2_G.get(element)
        if table is None:
            continue
        mass_att = table.get(isotope)
        if mass_att is None:
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0.0:
            continue
        total_weight += numeric_weight
        weighted_mu += numeric_weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return weighted_mu / total_weight


def interpolate_mass_attenuation_curve(curve_by_energy_keV: dict[float, float], energy_keV: float) -> float | None:
    """Interpolate a mass attenuation coefficient from a discrete energy curve."""
    if not curve_by_energy_keV:
        return None
    energies = np.asarray(sorted(float(energy) for energy in curve_by_energy_keV.keys()), dtype=float)
    values = np.asarray([float(curve_by_energy_keV[float(energy)]) for energy in energies], dtype=float)
    if energies.size == 0:
        return None
    if energies.size == 1:
        return float(values[0])
    clamped_energy_keV = float(np.clip(float(energy_keV), float(energies[0]), float(energies[-1])))
    return float(np.interp(clamped_energy_keV, energies, values))


def composition_mass_attenuation_at_energy(
    composition_by_mass: dict[str, float] | str | None,
    energy_keV: float,
) -> float | None:
    """Return a mixture mass attenuation coefficient at a specific photon energy."""
    composition = normalize_composition_by_mass(composition_by_mass)
    if not composition:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for element, weight in composition.items():
        curve = ELEMENTAL_MASS_ATT_CURVES_CM2_G.get(element)
        if curve is None:
            continue
        mass_att = interpolate_mass_attenuation_curve(curve, energy_keV)
        if mass_att is None:
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0.0:
            continue
        total_weight += numeric_weight
        weighted_mu += numeric_weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return weighted_mu / total_weight
