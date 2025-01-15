from os.path import dirname
from sys import path as sys_path

sys_path.append(dirname(__file__))

import argparse
from time import time

from od import OD
from utils import get_weights, validate_source, filter_by_area_ratio
from classify_poi import POIClassification


DEAULT_CONFIG_PATH = "./weights/default.yaml"
DEFAULT_WEIGHTS = get_weights(
    "RELEASE_0.1.2/yolov9_od_2507.pt"  # this file is moved from another server
)
DEFAULT_WEIGHT_POI_CLS = get_weights(
    "/mnt/data/POI_tiny/RELEASE_0.1.3/1510_64.pt"
)  # this file also


def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Parse two string arguments.")

    # Add string arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./test_in",
        help="Input images source (file / folder)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./test_out_od",
        help="Output folder to save images",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # (Optional) Preprocess: Validate source
    images = validate_source(args.input)

    # Detect labels in images, return generator because of memory limit
    od = OD(args.output, DEAULT_CONFIG_PATH, DEFAULT_WEIGHTS)
    results = []

    for image in images:
        results.extend(od.detect(image))

    # (Optional) Postprocess: Filter by area ratio
    poi_classifier = POIClassification(DEFAULT_WEIGHT_POI_CLS)
    filtered_results = filter_by_area_ratio(results, poi_classifier)
