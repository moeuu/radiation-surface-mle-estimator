import unittest

import numpy as np

from three_d_estimation.config import GeometryConfig
from three_d_estimation.geometry import create_grid, get_face_vector_shapes, restore_q
from three_d_estimation.measurement import create_A, decide_measurement_points


class CalculateTests(unittest.TestCase):
    def test_measurement_points_are_seeded(self):
        geometry = GeometryConfig(x=4, y=3, z=2, g=1)
        points_a = decide_measurement_points(geometry, 0.5, seed=7)
        points_b = decide_measurement_points(geometry, 0.5, seed=7)

        self.assertEqual(points_a, points_b)
        self.assertEqual(len(points_a), 6)

    def test_create_grid_uses_configured_geometry(self):
        grid = create_grid("x", 0.0, g=1, x=4, y=3, z=2)

        self.assertEqual(len(grid), 6)
        self.assertEqual(grid[0], [0.0, 0.5, 0.5])
        self.assertEqual(grid[-1], [0.0, 2.5, 1.5])

    def test_create_a_shape_matches_measurements_and_grid(self):
        measurement_points = [[0.5, 0.5, 0.5], [1.5, 1.5, 0.5]]
        shield_orientations = [{"theta": (0.0, np.pi), "phi": (0.0, 2 * np.pi)}]
        grid = create_grid("z", 0.0, g=1, x=2, y=2, z=2)

        matrix = create_A(2, 2, measurement_points, grid, shield_orientations, g=1)

        self.assertEqual(matrix.shape, (2, 4))

    def test_restore_q_round_trip(self):
        geometry = GeometryConfig(x=2, y=2, z=2, g=1)
        q_shapes = get_face_vector_shapes(geometry)
        original = np.arange(sum(shape[0] for shape in q_shapes), dtype=float).reshape(-1, 1)

        restored = restore_q(original, q_shapes)
        reconstructed = np.vstack(restored)

        np.testing.assert_array_equal(original, reconstructed)


if __name__ == "__main__":
    unittest.main()
