import numpy as np
import pytest

from src.analysis import (anomaly_details, classify_types, detect,
                          fit_harmonic, max_true_run, predict_harmonic)
from src.config import WETLANDS, site_paths
from src.download import compute_indices


def test_indices_ranges():
    rng = np.random.default_rng(0)
    b = {k: rng.random((4, 4), dtype=np.float32) * 0.5 + 0.05
         for k in ["red", "green", "blue", "nir", "swir1", "swir2"]}
    clear = np.ones((4, 4), dtype=bool)
    idx = compute_indices(b, clear)
    for v in idx.values():
        assert np.nanmin(v) >= -1.01 and np.nanmax(v) <= 1.51


def test_max_true_run():
    m = np.array([[True, True, False, True], [False, True, True, True]])
    assert list(max_true_run(m)) == [2, 3]


def test_persistence_requires_consecutive():
    resid = np.array([[0.2, 0.0, 0.2, 0.2, 0.2]], dtype=float)
    pred, anomaly = detect(resid, 0.1, 3)
    assert bool(pred[0])
    pred2, _ = detect(resid, 0.1, 4)
    assert not bool(pred2[0])


def test_anomaly_details_onset():
    anomaly = np.array([[False, True, True, True, True, False, True]])
    resid = np.where(anomaly, 0.2, 0.0)
    dates = np.array([f"2024-0{i+1}-01" for i in range(7)])
    d = anomaly_details(anomaly, 0.1, 3, resid, dates)
    assert bool(d["detected"][0])
    assert d["onset_date"][0] == "2024-02-01"
    assert int(d["longest_run"][0]) == 4
    assert 0 < float(d["confidence"][0]) <= 1


def test_classify_types():
    t = classify_types(np.array([0.2, -0.2, 0.0]), np.array([0.1, -0.1, -0.2]),
                       np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert t[0] == "Vegetation & water gain (recovery)"
    assert t[1] == "Vegetation & water loss"
    assert t[2] == "Water loss (drying)"


def test_harmonic_recovers_seasonality():
    doy = np.tile(np.array([15, 105, 196, 288]), 4).astype(float)
    from src.analysis import seasonal_design
    X = seasonal_design(doy)
    coef_true = np.array([0.4, 0.15, 0.05, 0.02, 0.01])
    y = (X @ coef_true).astype(np.float32)
    cube = np.tile(y, (3, 1))
    coef = fit_harmonic(cube, doy, np.array([0.4, 0.4, 0.4], dtype=np.float32))
    pred = predict_harmonic(coef, doy)
    assert np.allclose(pred[0], y, atol=0.02)


def test_sites_registered_with_aois():
    for site, cfg in WETLANDS.items():
        assert cfg["train_end"] <= cfg["monitor_start"]
        aoi = site_paths(site)["aoi"]
        assert aoi.exists(), f"missing AOI for {site}"


@pytest.mark.skipif(not site_paths("nyandungu")["cube"].exists(),
                    reason="cube not downloaded yet")
def test_app_pages():
    from app.app import app
    client = app.test_client()
    for url in ["/", "/wetlands", "/alerts", "/evaluation", "/admin",
                "/map/nyandungu", "/wetland/gikondo", "/data", "/model"]:
        assert client.get(url).status_code == 200, url
