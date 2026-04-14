import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

from three_d_estimation.cli import main
from three_d_estimation.config import EstimationConfig, GeometryConfig, OptimizerConfig, RadiationSource, default_shield_orientations
from three_d_estimation.pipeline import RadiationEstimation


class PipelineTests(unittest.TestCase):
    def test_small_pipeline_runs_end_to_end(self):
        config = EstimationConfig(
            geometry=GeometryConfig(x=2, y=2, z=2, g=1),
            q_max=20,
            measurement_ratio=0.5,
            sources=[RadiationSource(0.5, 0.5, 0.0, 10.0)],
            shield_orientations=default_shield_orientations()[:2],
            optimizer=OptimizerConfig(max_iter=5, learning_rate=0.05),
            random_seed=3,
            plot_results=False,
        )
        estimation = RadiationEstimation(config)

        summary = estimation.run()

        self.assertEqual(summary["A_shape"], [4, 24])
        self.assertEqual(summary["q_shape"], [24, 1])
        self.assertEqual(len(estimation.restored_qs), 6)
        self.assertIsInstance(summary["final_score"], float)

    def test_cli_json_output(self):
        stdout = StringIO()
        argv = [
            "--x",
            "2",
            "--y",
            "2",
            "--z",
            "2",
            "--max-iter",
            "2",
            "--measurement-ratio",
            "0.5",
            "--json",
        ]

        with redirect_stdout(stdout):
            exit_code = main(argv)

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["measurement_points"], 2)
        self.assertEqual(payload["orientations"], 8)


if __name__ == "__main__":
    unittest.main()
