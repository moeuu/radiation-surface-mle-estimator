import json

from three_d_estimation.config import build_default_config
from three_d_estimation.pipeline import RadiationEstimation


def main():
    estimation = RadiationEstimation(build_default_config())
    print(json.dumps(estimation.run(), indent=2))


if __name__ == "__main__":
    main()
