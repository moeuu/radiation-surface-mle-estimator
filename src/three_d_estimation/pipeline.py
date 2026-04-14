import numpy as np

from .config import EstimationConfig
from .geometry import build_initial_q, create_face_grids, get_face_vector_shapes, restore_q
from .measurement import create_A, decide_measurement_points, measurement_shield
from .optimization import Adam, score_func
from .plotting import plot_3d_heatmap_cube, plot_measurement_points


FACE_DIMENSIONS = (
    lambda geometry: (geometry.x, geometry.y),
    lambda geometry: (geometry.x, geometry.y),
    lambda geometry: (geometry.y, geometry.z),
    lambda geometry: (geometry.x, geometry.z),
    lambda geometry: (geometry.y, geometry.z),
    lambda geometry: (geometry.x, geometry.z),
)


class RadiationEstimation:
    def __init__(self, config: EstimationConfig):
        self.config = config
        self.geometry = config.geometry
        self.measurement_points = []
        self.grids = []
        self.A = None
        self.q = None
        self.q_optimized = None
        self.b_m = None
        self.restored_qs = None

    def setup_measurement_points(self):
        self.measurement_points = decide_measurement_points(
            self.geometry,
            self.config.measurement_ratio,
            seed=self.config.random_seed,
        )

    def setup_grids(self):
        self.grids = create_face_grids(self.geometry)

    def setup_A_matrix(self):
        matrices = []
        for index, grid in enumerate(self.grids):
            axis_length_1, axis_length_2 = FACE_DIMENSIONS[index](self.geometry)
            matrices.append(
                create_A(
                    axis_length_1,
                    axis_length_2,
                    self.measurement_points,
                    grid,
                    self.config.shield_orientations,
                    g=self.geometry.g,
                )
            )
        self.A = np.hstack(matrices)

    def setup_initial_q(self):
        self.q = build_initial_q(self.geometry, self.config.q_max)

    def generate_measurement_data(self):
        self.b_m = np.asarray(
            measurement_shield(
                self.measurement_points,
                self.config.sources,
                self.config.shield_orientations,
            ),
            dtype=float,
        ).reshape(-1, 1)

    def optimize_q(self):
        if self.A is None or self.b_m is None or self.q is None:
            raise ValueError("A, b_m, and q must be initialized before optimization.")
        self.q_optimized = Adam(self.A, self.b_m, self.q.copy(), optimizer_config=self.config.optimizer)

    def restore_q(self):
        if self.q_optimized is None:
            raise ValueError("Optimization has not been run yet.")
        self.restored_qs = restore_q(self.q_optimized, get_face_vector_shapes(self.geometry))

    def summarize(self):
        final_q = self.q_optimized if self.q_optimized is not None else self.q
        final_score = None
        if self.A is not None and self.b_m is not None and final_q is not None:
            final_score = score_func(self.A, self.b_m, final_q)

        return {
            "seed": self.config.random_seed,
            "measurement_points": len(self.measurement_points),
            "orientations": len(self.config.shield_orientations),
            "A_shape": list(self.A.shape) if self.A is not None else None,
            "q_shape": list(final_q.shape) if final_q is not None else None,
            "max_iter": self.config.optimizer.max_iter,
            "learning_rate": self.config.optimizer.learning_rate,
            "final_score": final_score,
        }

    def run(self):
        self.setup_measurement_points()
        self.setup_grids()
        self.setup_A_matrix()
        self.setup_initial_q()
        self.generate_measurement_data()
        self.optimize_q()
        self.restore_q()
        return self.summarize()

    def plot_results(self):
        if self.restored_qs is None:
            raise ValueError("Restored q values are not available.")

        all_q_values = np.concatenate([surface.flatten() for surface in self.restored_qs])
        vmin = float(all_q_values.min())
        vmax = float(all_q_values.max())
        plot_measurement_points(
            self.measurement_points,
            self.config.sources,
            "Measurement Points",
            self.geometry.x,
            self.geometry.y,
        )
        plot_3d_heatmap_cube(
            self.restored_qs,
            self.geometry.x,
            self.geometry.y,
            self.geometry.z,
            sources=self.config.sources,
            vmin=vmin,
            vmax=vmax,
        )
