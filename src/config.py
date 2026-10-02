"""Paths and study settings for the Nyandungu phase-1 prototype."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AOI_PATH = ROOT / "data" / "aoi" / "nyandungu.geojson"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CUBE_PATH = PROCESSED_DIR / "nyandungu_sentinel2.npz"
CATALOG_PATH = PROCESSED_DIR / "scene_catalog.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MAP_DIR = OUTPUT_DIR / "maps"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "seasonal_mlp.joblib"

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Window around the park so the control area sits outside the restored wetland.
# Order: west, south, east, north.
BOUNDS_WGS84 = (30.123, -1.976, 30.174, -1.940)
DST_CRS = "EPSG:32736"
RESOLUTION_M = 10

# Documented dates. Works began in September 2018. The eco-park opened on 8 July 2022.
PRE_END = "2018-09-01"
POST_START = "2022-07-08"
SEARCH_START = "2017-01-01"
SEARCH_END = "2026-06-30"

CLOUD_COVER_MAX = 35
MIN_CLEAR_FRACTION = 0.55
ATTEMPTS_PER_QUARTER = 5

# A post-opening date counts toward the reference when NDVI is at least this
# far above the pixel's pre-restoration median. Nyandungu's documented change
# in the Sentinel-2 years is a moderate, lasting greening, so the reference is
# an increase rather than a change in either direction.
REFERENCE_NDVI_SHIFT = 0.05
REFERENCE_PERSIST_FRACTION = 0.70

MIN_PRE_OBS = 3
MIN_POST_OBS = 4

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "max_iter": 250,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 12,
    "random_state": 42,
}

# SCL classes kept: dark, vegetation, bare soil, water, unclassified.
CLEAR_SCL = {2, 4, 5, 6, 7}
