"""Configuration for the Kigali wetland persistent-change monitoring system.

Multi-site design (approved): the system is wetland-agnostic. Each site is a
registry entry with its own AOI, study window, and normal-period definition.

Sites
-----
nyandungu : restoration story. Works began Sept 2018, eco-park opened 8 Jul 2022.
            Normal period = before the works.
gikondo   : degradation story (former industrial park, industries relocated
            from 2018) and one of REMA's five wetlands under rehabilitation
            since Feb 2024. Normal period = 2017-2019 (earliest reliable years).
rwampara  : degradation story (urban growth, agriculture, waste dumping;
            wetland area reported to have fallen sharply since 1987) and a
            REMA rehabilitation site where works started later (2024).
            Normal period = 2017-2019.

For every site the surrounding land outside the AOI inside the study window
is mapped as a control ring for context. In fast-urbanising Kigali the ring
itself genuinely changes, so formal accuracy is measured on labelled stable
pixels (spatial blocks), not on ring area.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AOI_DIR = ROOT / "data" / "aoi"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_ROOT = OUTPUT_DIR / "figures"
MAP_ROOT = OUTPUT_DIR / "maps"

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
DST_CRS = "EPSG:32736"  # UTM 36S
RESOLUTION_M = 10

SEARCH_START = "2017-01-01"
SEARCH_END = "2026-06-30"
CLOUD_COVER_MAX = 35
MIN_CLEAR_FRACTION = 0.55
ATTEMPTS_PER_QUARTER = 5

# SCL classes kept: dark area, vegetation, bare soil, water, unclassified.
CLEAR_SCL = {2, 4, 5, 6, 7}

# Indices computed per scene.
INDICES = ["ndvi", "ndwi", "mndwi", "ndmi", "savi", "bsi"]

WETLANDS = {
    "nyandungu": {
        "label": "Nyandungu Wetland",
        "subtitle": "Nyandungu Urban Wetland Eco-Tourism Park, Gasabo",
        "story": "restoration",
        "aoi_file": "nyandungu.geojson",
        "aoi_source": "OpenStreetMap (Nyandungu Urban Wetland Eco-Tourism Park) via Nominatim",
        "bounds_wgs84": (30.123, -1.976, 30.174, -1.940),
        "train_end": "2018-09-01",      # works began September 2018
        "monitor_start": "2018-09-01",
        "intervention_label": "Restoration works began (Sep 2018); park opened 8 Jul 2022",
        "expected_change": "Persistent vegetation gain (greening) after restoration",
    },
    "gikondo": {
        "label": "Gikondo Wetland",
        "subtitle": "Gikondo valley, Kicukiro / Gasabo / Nyarugenge",
        "story": "degradation_and_rehabilitation",
        "aoi_file": "gikondo.geojson",
        "aoi_source": "OpenStreetMap way 1209698095 (natural=wetland) via Overpass",
        "bounds_wgs84": (30.062, -1.982, 30.103, -1.936),
        "train_end": "2020-01-01",
        "monitor_start": "2020-01-01",
        "intervention_label": "Industrial relocation from 2018; REMA rehabilitation from Feb 2024",
        "expected_change": "Long industrial-era degradation, disturbed ground and recovery signals during rehabilitation",
    },
    "rwampara": {
        "label": "Rwampara Wetland",
        "subtitle": "Rwampara valley, Nyarugenge / Kicukiro",
        "story": "degradation_and_rehabilitation",
        "aoi_file": "rwampara.geojson",
        "aoi_source": "Study delineation: 180 m buffer of the OSM Rwampara stream (way 1117393177) via Overpass -- not an official boundary",
        "bounds_wgs84": (30.035, -1.995, 30.082, -1.955),
        "train_end": "2020-01-01",
        "monitor_start": "2020-01-01",
        "intervention_label": "Urban/agricultural pressure for decades; REMA rehabilitation started later in 2024",
        "expected_change": "Persistent vegetation/water loss from encroachment, with disturbance during rehabilitation",
    },
}

# --- Model settings -------------------------------------------------------
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
MIN_TRAIN_OBS = 3       # clear training observations per pixel
MIN_MONITOR_OBS = 4     # clear monitoring observations per pixel
PERSISTENCE_GRID_CONSECUTIVE = [2, 3, 4]
RESIDUAL_THRESHOLDS = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
MIN_COMPONENT_PIXELS = 9   # minimum mapping unit ~0.09 ha

# Reference ("silver standard") labelling: a pixel is a reference positive
# when its monitoring-period index level shifts from its normal-period median
# by at least this amount on at least this fraction of clear observations.
REFERENCE_SHIFT = 0.08
REFERENCE_FRACTION = 0.60


def site_paths(site: str) -> dict:
    cfg = WETLANDS[site]
    return {
        "aoi": AOI_DIR / cfg["aoi_file"],
        "cube": PROCESSED_DIR / f"{site}_sentinel2.npz",
        "catalog": PROCESSED_DIR / f"{site}_catalog.csv",
        "figures": FIGURE_ROOT / site,
        "maps": MAP_ROOT / site,
        "metrics": OUTPUT_DIR / f"metrics_{site}.json",
        "changes_geojson": OUTPUT_DIR / f"changes_{site}.geojson",
        "model": OUTPUT_DIR / f"model_{site}.joblib",
    }
