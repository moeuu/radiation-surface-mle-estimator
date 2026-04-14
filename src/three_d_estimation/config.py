from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class RadiationSource:
    x: float
    y: float
    z: float
    intensity: float

    @property
    def position(self):
        return (self.x, self.y, self.z)

    def as_tuple(self):
        return (self.x, self.y, self.z, self.intensity)


@dataclass(frozen=True)
class GeometryConfig:
    x: float
    y: float
    z: float
    g: float = 1.0

    def __post_init__(self):
        if self.x <= 0 or self.y <= 0 or self.z <= 0 or self.g <= 0:
            raise ValueError("Geometry dimensions and grid size must be positive.")
        for axis_name, length in (("x", self.x), ("y", self.y), ("z", self.z)):
            cells = length / self.g
            if not math.isclose(cells, round(cells)):
                raise ValueError(
                    f"{axis_name}={length} must be divisible by g={self.g} for a regular grid."
                )

    @property
    def x_cells(self):
        return int(round(self.x / self.g))

    @property
    def y_cells(self):
        return int(round(self.y / self.g))

    @property
    def z_cells(self):
        return int(round(self.z / self.g))


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iter: int = 2000
    min_q_value: float = 1e-7


@dataclass(frozen=True)
class EstimationConfig:
    geometry: GeometryConfig
    q_max: float
    measurement_ratio: float
    sources: list[RadiationSource] = field(default_factory=list)
    shield_orientations: list[dict] = field(default_factory=list)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    random_seed: int | None = 42
    plot_results: bool = False

    def __post_init__(self):
        if self.q_max <= 0:
            raise ValueError("q_max must be positive.")
        if not 0 <= self.measurement_ratio <= 1:
            raise ValueError("measurement_ratio must be between 0 and 1.")
        if not self.shield_orientations:
            raise ValueError("At least one shield orientation is required.")
        if not self.sources:
            raise ValueError("At least one radiation source is required.")


def default_shield_orientations():
    return [
        {"theta": (0, np.pi / 2), "phi": (0, np.pi / 2)},
        {"theta": (0, np.pi / 2), "phi": (np.pi / 2, np.pi)},
        {"theta": (0, np.pi / 2), "phi": (np.pi, 3 * np.pi / 2)},
        {"theta": (0, np.pi / 2), "phi": (3 * np.pi / 2, 2 * np.pi)},
        {"theta": (np.pi / 2, np.pi), "phi": (0, np.pi / 2)},
        {"theta": (np.pi / 2, np.pi), "phi": (np.pi / 2, np.pi)},
        {"theta": (np.pi / 2, np.pi), "phi": (np.pi, 3 * np.pi / 2)},
        {"theta": (np.pi / 2, np.pi), "phi": (3 * np.pi / 2, 2 * np.pi)},
    ]


def default_sources():
    return [
        RadiationSource(3.5, 3.5, 0.0, 100.0),
        RadiationSource(7.0, 3.0, 10.0, 200.0),
        RadiationSource(7.0, 10.0, 8.0, 150.0),
    ]


def build_default_config():
    return EstimationConfig(
        geometry=GeometryConfig(x=10.0, y=10.0, z=10.0, g=1.0),
        q_max=200.0,
        measurement_ratio=0.7,
        sources=default_sources(),
        shield_orientations=default_shield_orientations(),
        optimizer=OptimizerConfig(),
        random_seed=42,
        plot_results=False,
    )
