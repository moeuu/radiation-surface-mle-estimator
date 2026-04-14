import argparse
import json
from dataclasses import replace

from .config import GeometryConfig, build_default_config
from .pipeline import RadiationEstimation


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Estimate radiation intensity on room surfaces.")
    parser.add_argument("--x", type=float, default=10.0, help="Room size in the x direction.")
    parser.add_argument("--y", type=float, default=10.0, help="Room size in the y direction.")
    parser.add_argument("--z", type=float, default=10.0, help="Room size in the z direction.")
    parser.add_argument("--g", type=float, default=1.0, help="Grid size.")
    parser.add_argument("--measurement-ratio", type=float, default=0.7, help="Fraction of floor cells to sample.")
    parser.add_argument("--q-max", type=float, default=200.0, help="Maximum source intensity used for q initialization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for measurement point sampling.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument("--max-iter", type=int, default=2000, help="Maximum Adam iterations.")
    parser.add_argument("--plot", action="store_true", help="Render measurement and heatmap plots.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    return parser


def config_from_args(args):
    default_config = build_default_config()
    geometry = GeometryConfig(x=args.x, y=args.y, z=args.z, g=args.g)
    optimizer = replace(default_config.optimizer, learning_rate=args.learning_rate, max_iter=args.max_iter)
    return replace(
        default_config,
        geometry=geometry,
        measurement_ratio=args.measurement_ratio,
        q_max=args.q_max,
        random_seed=args.seed,
        plot_results=args.plot,
        optimizer=optimizer,
    )


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    estimation = RadiationEstimation(config)
    summary = estimation.run()

    if config.plot_results:
        estimation.plot_results()

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"seed: {summary['seed']}")
        print(f"measurement_points: {summary['measurement_points']}")
        print(f"orientations: {summary['orientations']}")
        print(f"A_shape: {tuple(summary['A_shape'])}")
        print(f"q_shape: {tuple(summary['q_shape'])}")
        print(f"max_iter: {summary['max_iter']}")
        print(f"learning_rate: {summary['learning_rate']}")
        print(f"final_score: {summary['final_score']:.6f}")

    return 0
