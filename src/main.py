import sys
import numpy as np

# パスの設定（モジュールのインポート先を指定）
sys.path.append('/home/morita/src/3D_Estimation/src')

# モジュールのインポート
from plot import plot_measurement_points, plot_3d_heatmap_cube
from calculate import decide_measurement_points, measurement_shield, create_grid, create_A, restore_q
from optimization import score_func, grad_func, Adam 


class RadiationEstimation:
    def __init__(self, x, y, z, g, q_max, r, sources, shield_orientations):
        self.x = x
        self.y = y
        self.z = z
        self.g = g
        self.q_max = q_max
        self.q_init = 1 / q_max
        self.r = r
        self.sources = sources
        self.shield_orientations = shield_orientations

        # 初期化
        self.measurement_points = []
        self.grids = []
        self.A = None
        self.q = None
        self.q_optimized = None
        self.b_m = None
        self.restored_qs = None

    def setup_measurement_points(self):
        """測定ポイントを設定"""
        self.measurement_points = decide_measurement_points(self.x, self.y, self.r)

    def setup_grids(self):
        """部屋の各面のグリッドを作成"""
        self.grids = [
            create_grid('z', 0, g=self.g),
            create_grid('z', self.z, g=self.g),
            create_grid('x', 0, g=self.g),
            create_grid('y', 0, g=self.g),
            create_grid('x', self.x, g=self.g),
            create_grid('y', self.y, g=self.g),
        ]

    def setup_A_matrix(self):
        """A行列を作成"""
        A_matrices = [
            create_A(self.x, self.y, self.measurement_points, grid, self.shield_orientations)
            for grid in self.grids
        ]
        self.A = np.hstack(A_matrices)

    def setup_initial_q(self):
        """初期qを設定"""
        q1 = np.array([self.q_init] * (self.x * self.y)).reshape(self.x * self.y, 1)
        q2 = np.array([self.q_init] * (self.x * self.y)).reshape(self.x * self.y, 1)
        q3 = np.array([self.q_init] * (self.y * self.z)).reshape(self.y * self.z, 1)
        q4 = np.array([self.q_init] * (self.x * self.z)).reshape(self.x * self.z, 1)
        q5 = np.array([self.q_init] * (self.y * self.z)).reshape(self.y * self.z, 1)
        q6 = np.array([self.q_init] * (self.x * self.z)).reshape(self.x * self.z, 1)
        self.q = np.vstack((q1, q2, q3, q4, q5, q6))

    def generate_measurement_data(self):
        """測定データb_mを生成"""
        self.b_m = measurement_shield(self.measurement_points, self.sources, self.shield_orientations)

    def optimize_q(self, learning_rate=0.1):
        """qを最適化"""
        if self.A is None or self.b_m is None or self.q is None:
            raise ValueError("A, b_m, または q が未初期化です。")
        self.q_optimized = Adam(self.A, np.array(self.b_m).reshape(-1, 1), self.q, learning_rate=learning_rate)

    def restore_q(self):
        """qを元の形状に復元"""
        q_shapes = [
            (self.x * self.y, 1),
            (self.x * self.y, 1),
            (self.y * self.z, 1),
            (self.x * self.z, 1),
            (self.y * self.z, 1),
            (self.x * self.z, 1),
        ]
        self.restored_qs = restore_q(self.q_optimized, q_shapes)

    def plot_results(self, vmin=None, vmax=None):
        """結果をプロット"""
        plot_measurement_points(self.measurement_points, self.sources, "Measurement Points", self.x, self.y)
        plot_3d_heatmap_cube(self.restored_qs, self.x, self.y, self.z, sources=self.sources, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    # パラメータ設定
    x, y, z, g = 10, 10, 10, 1
    q_max = 200
    r = 0.7
    sources = [
        [3.5, 3.5, 0.0, 100],
        [7.0, 3.0, 10.0, 200],
        [7.0, 10, 8.0, 150],
    ]
    shield_orientations = [
        {'theta': (0, np.pi / 2), 'phi': (0, np.pi / 2)},
        {'theta': (0, np.pi / 2), 'phi': (np.pi / 2, np.pi)},
        {'theta': (0, np.pi / 2), 'phi': (np.pi, 3 * np.pi / 2)},
        {'theta': (0, np.pi / 2), 'phi': (3 * np.pi / 2, 2 * np.pi)},
        {'theta': (np.pi / 2, np.pi), 'phi': (0, np.pi / 2)},
        {'theta': (np.pi / 2, np.pi), 'phi': (np.pi / 2, np.pi)},
        {'theta': (np.pi / 2, np.pi), 'phi': (np.pi, 3 * np.pi / 2)},
        {'theta': (np.pi / 2, np.pi), 'phi': (3 * np.pi / 2, 2 * np.pi)},
    ]

    # クラスのインスタンス化と処理の実行
    estimation = RadiationEstimation(x, y, z, g, q_max, r, sources, shield_orientations)
    estimation.setup_measurement_points()
    estimation.setup_grids()
    estimation.setup_A_matrix()
    estimation.setup_initial_q()
    estimation.generate_measurement_data()

    # デバッグ用: A行列とb_mの形状を確認
    print(f"A shape: {estimation.A.shape}")
    print(f"b_m shape: {len(estimation.b_m)}")
    print(f"q shape: {estimation.q.shape}")

    # qを最適化
    estimation.optimize_q()

    # 結果を復元してプロット
    estimation.restore_q()
    vmin, vmax = estimation.q.min(), estimation.q.max()
    estimation.plot_results(vmin, vmax)
