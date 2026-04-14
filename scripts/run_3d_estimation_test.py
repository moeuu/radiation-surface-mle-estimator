import json

from three_d_estimation.config import (
    EstimationConfig,
    GeometryConfig,
    OptimizerConfig,
    RadiationSource,
    default_shield_orientations,
)
from three_d_estimation.geometry import create_face_grids
from three_d_estimation.measurement import add_shield, decide_measurement_points
from three_d_estimation.pipeline import RadiationEstimation


def main():
    geometry = GeometryConfig(x=10, y=20, z=10, g=1)
    measurement_points = decide_measurement_points(geometry, 0.7, seed=42)
    measurement_points_with_shield = add_shield(measurement_points)
    config = EstimationConfig(
        geometry=geometry,
        q_max=200,
        measurement_ratio=0.7,
        sources=[
            RadiationSource(5.4, 7.3, 0.0, 100),
            RadiationSource(10.0, 8.2, 4.7, 200),
        ],
        shield_orientations=default_shield_orientations(),
        optimizer=OptimizerConfig(max_iter=200, learning_rate=0.1),
        random_seed=42,
        plot_results=False,
    )
    estimation = RadiationEstimation(config)
    summary = estimation.run()
    summary["measurement_points_with_shield"] = len(measurement_points_with_shield)
    summary["face_grid_sizes"] = [len(grid) for grid in create_face_grids(geometry)]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
