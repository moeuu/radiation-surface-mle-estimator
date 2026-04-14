import json
from dataclasses import replace

from three_d_estimation.config import RadiationSource, build_default_config
from three_d_estimation.pipeline import RadiationEstimation


def main():
    config = replace(
        build_default_config(),
        sources=[
            RadiationSource(8.5, 3.5, 0.0, 100.0),
            RadiationSource(7.0, 3.0, 10.0, 200.0),
            RadiationSource(7.0, 10.0, 5.0, 150.0),
        ],
    )
    estimation = RadiationEstimation(config)
    print(json.dumps(estimation.run(), indent=2))


if __name__ == "__main__":
    main()
