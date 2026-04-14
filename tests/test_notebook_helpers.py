import unittest

import numpy as np

from three_d_estimation.animation import build_shell_mesh, build_sphere_mesh
from three_d_estimation.config import GeometryConfig, RadiationSource
from three_d_estimation.geometry import create_face_grids
from three_d_estimation.measurement import assign_source_intensity_for_check
from three_d_estimation.priors import compute_prior_distribution_sparse


class NotebookHelperTests(unittest.TestCase):
    def test_assign_source_intensity_for_check_updates_single_grid(self):
        geometry = GeometryConfig(x=2, y=2, z=2, g=1)
        grids = create_face_grids(geometry)
        q_vectors = [np.zeros((len(grid), 1)) for grid in grids]

        assign_source_intensity_for_check(RadiationSource(0.5, 0.5, 0.0, 10.0), grids, q_vectors)

        nonzero_counts = [int(np.count_nonzero(vector)) for vector in q_vectors]
        self.assertEqual(sum(nonzero_counts), 1)
        self.assertIn(1, nonzero_counts)

    def test_compute_prior_distribution_sparse_returns_square_precision_matrix(self):
        updated_q = [np.ones((4, 1)), np.ones((4, 1)) * 2]

        mu, sigma_inv = compute_prior_distribution_sparse(updated_q)

        self.assertEqual(mu.shape, (8,))
        self.assertEqual(sigma_inv.shape, (8, 8))

    def test_animation_mesh_helpers_build_expected_shapes(self):
        sphere = build_sphere_mesh(3, resolution=10)
        shell = build_shell_mesh((0, np.pi / 2), (0, np.pi / 2), 4, resolution=8)

        self.assertEqual(sphere[0].shape, (10, 10))
        self.assertEqual(shell[0].shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
