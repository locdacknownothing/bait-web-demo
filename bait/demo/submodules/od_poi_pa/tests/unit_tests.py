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
def test_validate_source():
    source = TEST_ASSETS / "source"
    images = validate_source(source)
    expected = load_json(TEST_ASSETS / "validated_images.json")
    assert images == expected, "Test validate_source failed"


@measure_runtime
def test_detect(batch=1):
    DEFAULT_CONFIG_PATH = Path.cwd() / "weights/default.yaml"
    DEFAULT_WEIGHTS = get_weights(
        "RELEASE_0.1.2/yolov9_od_2507.pt"  # this file is moved from another server
    )

    od = OD(None, DEFAULT_CONFIG_PATH, DEFAULT_WEIGHTS)
    images = load_json(TEST_ASSETS / "validated_images.json")
    batch_results = od._detect_batch_(images, batch_size=batch)

    results = []

    for batch in batch_results:
        results.extend(batch)

    expected = load_object(TEST_ASSETS / "results.pkl")
    assert len(results) == len(expected), "Test detect failed"


@measure_runtime
def test_filter_by_area_ratio():
    results = load_object(TEST_ASSETS / "results.pkl")
    DEFAULT_WEIGHT_POI_CLS = get_weights("/mnt/data/POI_tiny/RELEASE_0.1.3/1510_64.pt")
    poi_classifier = POIClassification(DEFAULT_WEIGHT_POI_CLS)
    filtered_results = filter_by_area_ratio(results, poi_classifier)

    expected = load_object(TEST_ASSETS / "filtered_results.pkl")
    assert len(filtered_results) == len(expected), "Test filter_by_area_ratio failed"


if __name__ == "__main__":
    test_validate_source()
    test_detect(batch=8)
    test_filter_by_area_ratio()
