import json

import numpy as np

from three_d_estimation.config import (
    EstimationConfig,
    GeometryConfig,
    OptimizerConfig,
    RadiationSource,
    default_shield_orientations,
)
from three_d_estimation.geometry import get_face_vector_shapes, restore_q
from three_d_estimation.pipeline import RadiationEstimation
from three_d_estimation.priors import adjust_q_based_on_measurements, compute_prior_distribution_sparse


def main():
    geometry = GeometryConfig(x=10, y=10, z=10, g=1)
    config = EstimationConfig(
        geometry=geometry,
        q_max=200,
        measurement_ratio=0.5,
        sources=[
            RadiationSource(8.5, 3.5, 0.0, 100.0),
            RadiationSource(7.0, 3.0, 10.0, 200.0),
            RadiationSource(7.0, 10.0, 5.0, 150.0),
        ],
        shield_orientations=default_shield_orientations(),
        optimizer=OptimizerConfig(max_iter=50, learning_rate=0.1),
        random_seed=42,
        plot_results=False,
    )
    estimation = RadiationEstimation(config)
    estimation.setup_measurement_points()
    estimation.setup_grids()
    estimation.setup_A_matrix()
    estimation.setup_initial_q()
    estimation.generate_measurement_data()
    q_surfaces = restore_q(estimation.q, get_face_vector_shapes(geometry))
    updated_q = adjust_q_based_on_measurements(
        estimation.measurement_points,
        estimation.b_m.flatten(),
        config.shield_orientations,
        q_surfaces,
        geometry.x,
        geometry.y,
        geometry.z,
    )
    mu, sigma_inv = compute_prior_distribution_sparse(updated_q)
    print(json.dumps({"mu_shape": list(mu.shape), "sigma_inv_shape": list(sigma_inv.shape)}, indent=2))


if __name__ == "__main__":
    main()
