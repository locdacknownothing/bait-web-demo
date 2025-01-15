from os.path import dirname, join
from sys import path as sys_path

sys_path.append(join(dirname(__file__), "../"))

from pathlib import Path

from od import OD
from helpers import measure_runtime
from utils import (
    load_json,
    load_object,
    get_weights,
    validate_source,
    filter_by_area_ratio,
)
from classify_poi import POIClassification


TEST_ASSETS = Path(__file__) / "../test_assets"
TEST_ASSETS = TEST_ASSETS.resolve()


@measure_runtime
def test_integration():
    DEFAULT_CONFIG_PATH = Path.cwd() / "weights/default.yaml"
    DEFAULT_WEIGHTS = get_weights(
        "RELEASE_0.1.2/yolov9_od_2507.pt"  # this file is moved from another server
    )
    DEFAULT_WEIGHT_POI_CLS = get_weights("/mnt/data/POI_tiny/RELEASE_0.1.3/1510_64.pt")

    od = OD(None, DEFAULT_CONFIG_PATH, DEFAULT_WEIGHTS)
    images = validate_source(TEST_ASSETS / "source")
    results = []

    for image in images:
        results.extend(od.detect(image))

    poi_classifier = POIClassification(DEFAULT_WEIGHT_POI_CLS)
    filtered_results = filter_by_area_ratio(results, poi_classifier)

    expected = load_object(TEST_ASSETS / "filtered_results.pkl")
    assert len(filtered_results) == len(expected), "Test integration failed"


if __name__ == "__main__":
    test_integration()
