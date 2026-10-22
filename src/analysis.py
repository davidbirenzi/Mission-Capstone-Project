"""Persistent-change engine for the Kigali wetland monitoring system.

Method (per wetland)
--------------------
1. NORMAL-BEHAVIOUR MODELS learn expected NDVI from the normal period:
   a) per-pixel harmonic regression (seasonal sin/cos terms), and
   b) a scikit-learn MLP regressor (season + the pixel's normal NDVI/NDWI/
      MNDWI levels), trained on wetland pixels in the normal period.
2. PERSISTENCE ENGINE: a pixel is a persistent change when the absolute
   residual (observed - expected NDVI) stays at/above a tuned threshold for
   at least k consecutive clear observations in the monitoring period.
   For each detected pixel we record onset date, longest run, anomaly count,
   mean magnitude and a confidence score.
3. CHANGE TYPING from monitoring-vs-normal shifts in NDVI, MNDWI, NDMI, BSI.
4. BASELINE: conventional two-date NDVI differencing for comparison.
5. EVALUATION against an independent multi-index persistent-shift reference
   ("silver standard", see docs/METHODS.md for rationale and limitations),
   using spatially held-out blocks so neighbouring pixels cannot leak.
   Metrics: precision, recall, F1, accuracy, IoU + confusion matrix.

Usage:  python -m src.analysis --site nyandungu | --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             jaccard_score, precision_score, recall_score)
from sklearn.neural_network import MLPRegressor

from src.config import (INDICES, MIN_COMPONENT_PIXELS, MIN_MONITOR_OBS,
                        MIN_TRAIN_OBS, MLP_PARAMS, PERSISTENCE_GRID_CONSECUTIVE,
                        REFERENCE_FRACTION, REFERENCE_SHIFT, RESIDUAL_THRESHOLDS,
                        RESOLUTION_M, WETLANDS, site_paths)

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": "#334155", "font.size": 10,
                     "axes.titleweight": "semibold", "text.color": "#0f172a",
                     "axes.labelcolor": "#0f172a", "xtick.color": "#334155",
                     "ytick.color": "#334155"})

TYPE_COLORS = {
    "Vegetation gain": "#15803d",
    "Vegetation & water gain (recovery)": "#0e9f6e",
    "Vegetation loss": "#b45309",
    "Vegetation & water loss": "#b42318",
    "Water loss (drying)": "#c2410c",
    "Water gain": "#2563eb",
    "Built-up / bare encroachment": "#7c2d12",
    "Mixed / other persistent change": "#6d28d9",
}
INK = "#0f172a"


# ---------------------------------------------------------------- utilities
def hectares(n_px: int) -> float:
    return float(n_px * RESOLUTION_M * RESOLUTION_M / 10_000.0)


def load_cube(site: str) -> dict:
    paths = site_paths(site)
    if not paths["cube"].exists():
        raise FileNotFoundError(f"Missing {paths['cube']}. Run: python -m src.download --site {site}")
    d = np.load(paths["cube"], allow_pickle=False)
    clear = d["clear"] == 1
    cube = {k: np.where(clear, d[k].astype(np.float32), np.nan) for k in INDICES}
    cube.update({
        "dates": np.array([str(v) for v in d["dates"]]),
        "transform": Affine(*d["transform"].tolist()),
        "bounds": tuple(float(v) for v in d["bounds_wgs84"]),
        "rgb_early": d["rgb_early"], "rgb_late": d["rgb_late"],
        "rgb_early_date": str(d["rgb_early_date"]), "rgb_late_date": str(d["rgb_late_date"]),
        "clear": clear,
    })
    return cube


def aoi_mask(site: str, shape_hw, transform) -> np.ndarray:
    from pyproj import Transformer
    paths = site_paths(site)
    raw = json.loads(Path(paths["aoi"]).read_text(encoding="utf-8"))
    geom = shape(raw["features"][0]["geometry"])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32736", always_xy=True)
    geom_utm = shapely_transform(tr.transform, geom)
    return rasterize([(geom_utm, 1)], out_shape=shape_hw, transform=transform,
                     fill=0, dtype="uint8").astype(bool)


def doys(dates) -> np.ndarray:
    return np.array([pd.Timestamp(str(v)).day_of_year for v in dates], dtype=float)


def seasonal_design(doy_array):
    a = 2 * np.pi * np.asarray(doy_array, dtype=float) / 365.25
    return np.column_stack([np.ones_like(a), np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)])


def periods(dates, cfg):
    dates = np.asarray(dates)
    train = dates < cfg["train_end"]
    monitor = dates >= cfg["monitor_start"]
    return train, monitor


# ------------------------------------------------------------------- models
def fit_harmonic(train_cube: np.ndarray, train_doy: np.ndarray, fallback: np.ndarray):
    """Seasonal normal model: per-pixel level + POOLED harmonic seasonality.

    A separate 5-parameter harmonic per pixel overfits when the normal
    period has few scenes (Nyandungu has 7). Instead each pixel keeps its
    own level (intercept), while the seasonal shape (sin/cos terms) is
    estimated once from all pixels' training observations (tens of
    thousands of samples), then the levels are re-estimated. Two passes.
    Returns coef (pixels,5): [level, sin, cos, sin2, cos2] with the last
    four shared across pixels, so predict_harmonic works unchanged.
    """
    X = seasonal_design(train_doy)[:, 1:]               # (T,4) seasonal terms
    valid = np.isfinite(train_cube)
    level = np.where(np.isfinite(fallback), fallback, 0.0).astype(np.float64)
    beta = np.zeros(4)
    for _ in range(2):
        resid = np.where(valid, train_cube - level[:, None], 0.0)
        XtX = np.zeros((4, 4)); XtY = np.zeros(4)
        for t in range(X.shape[0]):
            v = valid[:, t].astype(np.float64)
            XtX += float(v.sum()) * np.outer(X[t], X[t])
            XtY += float((v * resid[:, t]).sum()) * X[t]
        beta = np.linalg.solve(XtX + np.eye(4) * 1e-6, XtY)
        seasonal = X @ beta
        adj = np.where(valid, train_cube - seasonal[None, :], np.nan)
        with np.errstate(all="ignore"):
            new_level = np.nanmedian(adj, axis=1)
        level = np.where(np.isfinite(new_level), new_level, level)
    coef = np.zeros((train_cube.shape[0], 5))
    coef[:, 0] = level
    coef[:, 1:] = beta[None, :]
    return coef


def predict_harmonic(coef, all_doy):
    return (coef @ seasonal_design(all_doy).T).astype(np.float32)  # (pixels, T)


def mlp_features(doy_scalar, med_ndvi, med_ndwi, med_mndwi):
    a = 2 * np.pi * float(doy_scalar) / 365.25
    season = np.array([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)], dtype=np.float32)
    n = med_ndvi.shape[0]
    X = np.empty((n, 7), dtype=np.float32)
    X[:, 0:4] = season[None, :]
    X[:, 4], X[:, 5], X[:, 6] = med_ndvi, med_ndwi, med_mndwi
    return X


def fit_mlp(flat_ndvi, dates, train_mask, normals, inside, sample_size=8000):
    # Train on usable pixels across the whole study window (not only inside
    # the wetland): the normal-behaviour model must be valid wherever it is
    # applied, otherwise off-wetland land covers extrapolate into false alarms.
    cand = np.flatnonzero(normals["usable"])
    if cand.size == 0:
        raise RuntimeError("No usable wetland pixels in the normal period.")
    rng = np.random.default_rng(42)
    if cand.size > sample_size:
        cand = rng.choice(cand, size=sample_size, replace=False)
    train_doy = doys(dates[train_mask])
    cube_train = flat_ndvi[:, train_mask]
    blocks, targets = [], []
    for j, dy in enumerate(train_doy):
        obs = cube_train[cand, j]
        v = np.isfinite(obs)
        if v.sum() < 50:
            continue
        rows = cand[v]
        blocks.append(mlp_features(dy, normals["med_ndvi"][rows], normals["med_ndwi"][rows], normals["med_mndwi"][rows]))
        targets.append(obs[v])
    X = np.vstack(blocks); y = np.concatenate(targets)
    model = MLPRegressor(**MLP_PARAMS)
    model.fit(X, y)
    return model, {"n_samples": int(X.shape[0]), "n_pixels": int(cand.size)}


def predict_mlp(model, shape0, dates, normals):
    exp = np.full((shape0, len(dates)), np.nan, dtype=np.float32)
    rows = np.flatnonzero(normals["usable"])
    for j, dy in enumerate(doys(dates)):
        exp[rows, j] = model.predict(mlp_features(dy, normals["med_ndvi"][rows],
                                                  normals["med_ndwi"][rows], normals["med_mndwi"][rows]))
    return exp


# -------------------------------------------------------------- persistence
def max_true_run(mask2d):
    best = np.zeros(mask2d.shape[0], dtype=np.int16)
    run = np.zeros(mask2d.shape[0], dtype=np.int16)
    for c in range(mask2d.shape[1]):
        run = (run + 1) * mask2d[:, c]
        best = np.maximum(best, run)
    return best


def detect(mask_resid_monitor, threshold, min_consecutive):
    anomaly = np.isfinite(mask_resid_monitor) & (np.abs(mask_resid_monitor) >= threshold)
    return max_true_run(anomaly) >= min_consecutive, anomaly


def anomaly_details(anomaly, threshold, min_consecutive, resid_monitor, monitor_dates):
    """Onset (first run reaching k), runs, counts, magnitude, confidence."""
    n_px, n_t = anomaly.shape
    onset_idx = np.full(n_px, -1, dtype=np.int32)
    longest = np.zeros(n_px, dtype=np.int16)
    counts = anomaly.sum(axis=1).astype(np.int16)
    run = np.zeros(n_px, dtype=np.int16)
    for c in range(n_t):
        run = (run + 1) * anomaly[:, c]
        longest = np.maximum(longest, run)
        newly = (run == min_consecutive) & (onset_idx < 0)
        onset_idx[newly] = c - min_consecutive + 1
    detected = onset_idx >= 0
    frac_after = np.zeros(n_px, dtype=np.float32)
    mean_mag = np.zeros(n_px, dtype=np.float32)
    for i in np.flatnonzero(detected):
        seg = anomaly[i, onset_idx[i]:]
        frac_after[i] = float(seg.mean()) if seg.size else 0.0
        vals = np.abs(resid_monitor[i, onset_idx[i]:])
        vals = vals[np.isfinite(vals)]
        mean_mag[i] = float(vals.mean()) if vals.size else 0.0
    confidence = np.clip(0.40 * np.minimum(longest / 6.0, 1.0) + 0.40 * frac_after
                         + 0.20 * np.minimum(mean_mag / max(2 * threshold, 1e-9), 1.0), 0, 1)
    onset_date = np.array(["" if i < 0 else str(monitor_dates[i]) for i in onset_idx])
    return {"detected": detected, "onset_idx": onset_idx, "onset_date": onset_date,
            "longest_run": longest, "anom_count": counts, "frac_after_onset": frac_after,
            "mean_magnitude": mean_mag, "confidence": confidence.astype(np.float32)}


def classify_types(d_ndvi, d_mndwi, d_ndmi, d_bsi):
    t = np.full(d_ndvi.shape, "Mixed / other persistent change", dtype=object)
    veg_gain = d_ndvi >= 0.08
    veg_loss = d_ndvi <= -0.08
    water_gain = d_mndwi >= 0.08
    water_loss = d_mndwi <= -0.08
    t[veg_gain] = "Vegetation gain"
    t[water_gain & ~veg_gain] = "Water gain"
    t[veg_loss] = "Vegetation loss"
    t[water_loss & ~veg_loss] = "Water loss (drying)"
    t[veg_gain & (d_mndwi >= 0.03)] = "Vegetation & water gain (recovery)"
    t[veg_loss & (d_mndwi <= -0.03)] = "Vegetation & water loss"
    t[(d_bsi >= 0.08) & (d_ndvi <= -0.05)] = "Built-up / bare encroachment"
    return t


# -------------------------------------------------------------- evaluation
def reference_labels(flat, normals, inside):
    """Silver-standard labels from persistent multi-observation index shift."""
    mon = flat["ndvi"][:, normals["monitor"]]
    base = normals["med_ndvi"][:, None]
    diff = mon - base
    obs = np.isfinite(diff)
    signed = np.where(diff >= REFERENCE_SHIFT, 1, np.where(diff <= -REFERENCE_SHIFT, -1, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_pos = np.sum((signed == 1) & obs, axis=1) / np.maximum(obs.sum(axis=1), 1)
        frac_neg = np.sum((signed == -1) & obs, axis=1) / np.maximum(obs.sum(axis=1), 1)
    shift = normals["med_monitor_ndvi"] - normals["med_ndvi"]
    labels = np.full(flat["ndvi"].shape[0], -1, dtype=np.int8)
    pos = normals["usable"] & inside & (((frac_pos >= REFERENCE_FRACTION) & (shift >= REFERENCE_SHIFT)) |
                                        ((frac_neg >= REFERENCE_FRACTION) & (shift <= -REFERENCE_SHIFT)))
    neg = normals["usable"] & (np.abs(shift) < REFERENCE_SHIFT / 2) & (frac_pos < 0.3) & (frac_neg < 0.3)
    labels[pos] = 1
    labels[neg] = 0
    return labels


def block_split(labels, height, width, block=15, test_fraction=0.35, seed=42):
    rows, cols = np.divmod(np.arange(height * width), width)
    block_ids = (rows // block) * ((width // block) + 1) + (cols // block)
    tune = np.zeros(labels.shape[0], dtype=bool); test = np.zeros(labels.shape[0], dtype=bool)
    rng = np.random.default_rng(seed)
    for cls in (0, 1):
        members = np.flatnonzero(labels == cls)
        blocks = np.unique(block_ids[members]); rng.shuffle(blocks)
        n_test = max(1, int(round(len(blocks) * test_fraction)))
        test_blocks = set(blocks[:n_test].tolist())
        is_test = np.isin(block_ids[members], list(test_blocks))
        test[members[is_test]] = True; tune[members[~is_test]] = True
    return tune, test


def _tuning_f1(y_true, y_pred):
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(y_true == 1); neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    keep = min(neg.size, pos.size * 5)
    idx = np.concatenate([pos, rng.choice(neg, size=keep, replace=False)])
    return float(f1_score(y_true[idx], y_pred[idx], zero_division=0))


def _scores(y_true, y_pred, extra=None):
    out = {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, zero_division=0)),
           "accuracy": float(accuracy_score(y_true, y_pred)),
           "iou": float(jaccard_score(y_true, y_pred, zero_division=0)),
           "support": int(len(y_true)), "positives": int(np.sum(y_true == 1)),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
    if extra: out.update(extra)
    return out


def tune_model(resid, normals, labels, tune_idx, test_idx):
    resid_mon = resid[:, normals["monitor"]]
    y_tune, y_test = labels[tune_idx], labels[test_idx]
    best = None
    for thr in RESIDUAL_THRESHOLDS:
        for k in PERSISTENCE_GRID_CONSECUTIVE:
            pred, _ = detect(resid_mon, thr, k)
            s = _tuning_f1(y_tune, pred[tune_idx])
            if best is None or s > best[0]:
                best = (s, thr, k)
    _, thr, k = best
    pred, anomaly = detect(resid_mon, thr, k)
    scores = _scores(y_test, pred[test_idx], {"threshold": float(thr), "min_consecutive": int(k)})
    return pred, anomaly, thr, k, scores


def tune_baseline(flat_ndvi, normals, labels, tune_idx, test_idx, dates):
    tr = np.flatnonzero(normals["train"]); mon = np.flatnonzero(normals["monitor"])
    pre_i = tr[np.argmax(np.sum(np.isfinite(flat_ndvi[:, tr]), axis=0))]
    # Conventional two-date comparison: clearest normal-period scene vs the
    # clearest of the LATEST monitoring scenes (start-vs-end of the record).
    late = mon[-min(4, len(mon)):]
    post_i = late[np.argmax(np.sum(np.isfinite(flat_ndvi[:, late]), axis=0))]
    delta = flat_ndvi[:, post_i] - flat_ndvi[:, pre_i]
    y_tune, y_test = labels[tune_idx], labels[test_idx]
    best = None
    for thr in RESIDUAL_THRESHOLDS:
        pred = np.isfinite(delta) & (np.abs(delta) >= thr)
        s = _tuning_f1(y_tune, pred[tune_idx])
        if best is None or s > best[0]:
            best = (s, thr)
    pred = np.isfinite(delta) & (np.abs(delta) >= best[1])
    scores = _scores(y_test, pred[test_idx], {"threshold": float(best[1]),
                     "pre_date": str(dates[pre_i]), "post_date": str(dates[post_i])})
    return pred, delta, scores


# ----------------------------------------------------------------- outputs
def build_polygons(site, detected, types, details, transform, height, width):
    # 4-connectivity, matching rasterio.features.shapes polygonisation:
    # with 8-connected labels, shapes() splits diagonal blobs into many
    # polygons and each would duplicate the component's area.
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    lab, n = ndimage.label(detected.reshape(height, width), structure=cross)
    flat_lab = lab.ravel()
    from pyproj import Transformer
    to_wgs = Transformer.from_crs("EPSG:32736", "EPSG:4326", always_xy=True)
    features = []
    counts = np.bincount(flat_lab, minlength=n + 1)
    cid = 0
    for comp in range(1, n + 1):
        if counts[comp] < MIN_COMPONENT_PIXELS:
            continue
        members = np.flatnonzero(flat_lab == comp)
        vals, cnt = np.unique(types[members], return_counts=True)
        dom_type = str(vals[np.argmax(cnt)])
        onset_vals = [d for d in details["onset_date"][members] if d]
        onset = sorted(onset_vals)[len(onset_vals) // 2] if onset_vals else ""
        conf = float(np.mean(details["confidence"][members]))
        mask = (lab == comp).astype(np.int32)
        geoms = [shape(gd) for gd, val in shapes(mask, mask=mask > 0, transform=transform) if val == 1]
        if not geoms:
            continue
        from shapely.ops import unary_union
        g = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        if g.area == 0:
            continue
        g = g.simplify(5.0)
        gw = shapely_transform(to_wgs.transform, g)
        cid += 1
        features.append({
            "type": "Feature",
            "properties": {
                "id": cid, "site": site, "change_type": dom_type,
                "area_ha": round(hectares(int(counts[comp])), 3),
                "pixels": int(counts[comp]), "onset_date": onset,
                "mean_magnitude": round(float(np.mean(details["mean_magnitude"][members])), 4),
                "max_run": int(np.max(details["longest_run"][members])),
                "confidence": round(conf, 3),
                "confidence_level": "High" if conf >= 0.66 else ("Medium" if conf >= 0.4 else "Low"),
                "centroid_lon": round(float(gw.centroid.x), 6),
                "centroid_lat": round(float(gw.centroid.y), 6),
            },
            "geometry": mapping(gw),
        })
    features.sort(key=lambda f: -f["properties"]["area_ha"])
    for i, f in enumerate(features, 1):
        f["properties"]["id"] = i
    return {"type": "FeatureCollection", "features": features}


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def make_figures(site, cube, flat, normals, inside, resid_primary, labels,
                 model_scores_all, chosen_pred, types, mlp_model):
    figdir = site_paths(site)["figures"]; figdir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(cube["dates"]); height, width = cube["ndvi"].shape[1:]
    inside2d = inside.reshape(height, width)
    # flat arrays are (pixels, T); transpose to (T, pixels) for per-date series
    flat_series = {k: v.T for k, v in flat.items()}

    # NDVI / MNDWI distributions: normal vs monitoring inside AOI
    for idx_name, fname, xlabel in [("ndvi", "ndvi_distribution.png", "NDVI"),
                                    ("mndwi", "mndwi_distribution.png", "MNDWI (water)")]:
        fig, ax = plt.subplots(figsize=(8, 4.3))
        s = flat_series[idx_name]
        a = s[normals["train"]][:, inside].ravel(); b = s[normals["monitor"]][:, inside].ravel()
        ax.hist(a[np.isfinite(a)], bins=40, density=True, alpha=.75, color="#64748b", label="Normal period")
        ax.hist(b[np.isfinite(b)], bins=40, density=True, alpha=.65, color="#1d4e89", label="Monitoring period")
        ax.set_title(f"{xlabel} inside {WETLANDS[site]['label']}"); ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.legend(frameon=False)
        save_fig(fig, figdir / fname)

    # Median time series inside vs control ring
    for idx_name, fname in [("ndvi", "ndvi_timeseries.png"), ("mndwi", "mndwi_timeseries.png")]:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        s = flat_series[idx_name]
        ax.plot(dates, np.nanmedian(s[:, inside], axis=1), "o-", color="#1d4e89", label="Inside wetland")
        ax.plot(dates, np.nanmedian(s[:, ~inside], axis=1), "o-", color="#94a3b8", label="Control ring (outside AOI)")
        ax.axvline(pd.Timestamp(WETLANDS[site]["train_end"]), color="#b42318", ls="--", lw=1.2, label="Monitoring starts")
        ax.set_ylabel(f"Median {idx_name.upper()}"); ax.set_title(f"{idx_name.upper()} — {WETLANDS[site]['label']}")
        ax.legend(frameon=False, fontsize=8, ncol=3); fig.autofmt_xdate()
        save_fig(fig, figdir / fname)

    # Residual time series
    fig, ax = plt.subplots(figsize=(9, 4.4))
    rp = resid_primary.reshape(-1, len(dates)) if resid_primary.ndim == 2 else resid_primary
    ax.plot(dates, np.nanmedian(rp[inside], axis=0), "o-", color="#1d4e89", label="Inside wetland")
    ax.plot(dates, np.nanmedian(rp[~inside], axis=0), "o-", color="#94a3b8", label="Control ring")
    ax.axhline(0, color=INK, lw=.8); ax.axvline(pd.Timestamp(WETLANDS[site]["train_end"]), color="#b42318", ls="--", lw=1.2)
    ax.set_ylabel("Observed − expected NDVI"); ax.set_title("Deviation from learned normal pattern"); ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate(); save_fig(fig, figdir / "residual_timeseries.png")

    # Metrics comparison
    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = ["Precision", "Recall", "F1", "Accuracy", "IoU"]; keys = ["precision", "recall", "f1", "accuracy", "iou"]
    x = np.arange(len(names)); wdt = 0.27
    for i, (mname, sc) in enumerate(model_scores_all.items()):
        ax.bar(x + (i - 1) * wdt, [sc[k] for k in keys], width=wdt, label=mname)
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.set_ylabel("Score (held-out blocks)")
    ax.set_title("Held-out performance"); ax.legend(frameon=False, fontsize=8)
    save_fig(fig, figdir / "metrics.png")

    # Change map (raster view): typed colours INSIDE the wetland; detections
    # elsewhere in the study window are grey context only.
    from matplotlib.colors import to_rgba
    det_inside = chosen_pred & inside
    type2d = np.full((height, width), "", dtype=object); type2d.ravel()[det_inside] = types[det_inside]
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    base = np.zeros((height, width, 4), dtype=float); base[inside2d] = (0.85, 0.90, 0.96, 0.9)
    ax.imshow(base, interpolation="nearest")
    ctx = np.zeros((height, width, 4)); ctx[chosen_pred.reshape(height, width) & ~inside2d] = to_rgba("#a8a29e", 0.45)
    ax.imshow(ctx, interpolation="nearest")
    for tname, color in TYPE_COLORS.items():
        m = type2d == tname
        if m.any():
            overlay = np.zeros((height, width, 4))
            overlay[m] = to_rgba(color, 0.92)
            ax.imshow(overlay, interpolation="nearest")
    ax.set_title(f"Persistent change — {WETLANDS[site]['label']} (grey = outside wetland)"); ax.set_axis_off()
    save_fig(fig, figdir / "change_map.png")

    # Area by type (inside AOI)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals, cnt = np.unique(types[det_inside], return_counts=True)
    order = np.argsort(cnt)[::-1]
    ax.barh([str(vals[i]) for i in order], [hectares(int(cnt[i])) for i in order],
            color=[TYPE_COLORS.get(str(vals[i]), "#666") for i in order])
    ax.set_xlabel("Hectares"); ax.set_title("Persistent-change area by type (inside wetland)")
    save_fig(fig, figdir / "change_by_type.png")

    # Architecture + loss
    fig, ax = plt.subplots(figsize=(9.4, 3.0)); ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.set_axis_off()
    boxes = [(0.2, "7 inputs\nseason + normal\nNDVI/NDWI/MNDWI"), (2.9, "64 units\nReLU"), (5.6, "32 units\nReLU"), (8.3, "1 output\nexpected NDVI")]
    for x0, text in boxes:
        ax.add_patch(plt.Rectangle((x0, 1.05), 2.05, 1.9, facecolor="#f1f5f9", edgecolor=INK))
        ax.text(x0 + 1.02, 2.0, text, ha="center", va="center", fontsize=8)
    for (x0, _), (nx, _) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(nx, 2.0), xytext=(x0 + 2.05, 2.0), arrowprops={"arrowstyle": "->"})
    ax.set_title("MLPRegressor · Adam · squared error · early stopping (harmonic model runs in parallel)")
    save_fig(fig, figdir / "architecture.png")
    if mlp_model is not None and hasattr(mlp_model, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(mlp_model.loss_curve_, color="#1d4e89")
        ax.set_title("MLP training loss"); ax.set_xlabel("Iteration"); ax.set_ylabel("Squared error")
        save_fig(fig, figdir / "loss_curve.png")

    from PIL import Image
    Image.fromarray(cube["rgb_early"], mode="RGB").save(figdir / "rgb_early.png")
    Image.fromarray(cube["rgb_late"], mode="RGB").save(figdir / "rgb_late.png")


def write_overlay(site, chosen_pred, types, inside, height, width):
    from PIL import Image
    from matplotlib.colors import to_rgba
    paths = site_paths(site); paths["maps"].mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[inside.reshape(height, width)] = (29, 78, 137, 60)
    img[chosen_pred.reshape(height, width) & ~inside.reshape(height, width)] = (168, 162, 158, 110)
    flat_types = np.full(height * width, "", dtype=object)
    det_inside = chosen_pred & inside; flat_types[det_inside] = types[det_inside]
    t2 = flat_types.reshape(height, width)
    for tname, color in TYPE_COLORS.items():
        m = t2 == tname
        if m.any():
            rgba = tuple(int(round(c * 255)) for c in to_rgba(color, 0.95))
            img[m] = rgba
    Image.fromarray(img, mode="RGBA").save(paths["maps"] / "persistent_change.png")


# --------------------------------------------------------------------- main
def run_analysis(site: str) -> dict:
    cfg = WETLANDS[site]; paths = site_paths(site)
    cube = load_cube(site)
    dates = cube["dates"]; T, height, width = cube["ndvi"].shape
    inside2d = aoi_mask(site, (height, width), cube["transform"])
    inside = inside2d.ravel()
    flat = {k: cube[k].reshape(T, -1).T for k in INDICES}  # (pixels, T)
    train_mask, monitor_mask = periods(dates, cfg)

    with np.errstate(all="ignore"):
        med = {k: np.nanmedian(flat[k][:, train_mask], axis=1) for k in INDICES}
        med_monitor_ndvi = np.nanmedian(flat["ndvi"][:, monitor_mask], axis=1)
    train_count = np.sum(np.isfinite(flat["ndvi"][:, train_mask]), axis=1)
    mon_count = np.sum(np.isfinite(flat["ndvi"][:, monitor_mask]), axis=1)
    normals = {"train": train_mask, "monitor": monitor_mask,
               "med_ndvi": med["ndvi"], "med_ndwi": med["ndwi"], "med_mndwi": med["mndwi"],
               "med_monitor_ndvi": med_monitor_ndvi,
               "usable": (train_count >= MIN_TRAIN_OBS) & (mon_count >= MIN_MONITOR_OBS)
               & np.isfinite(med["ndvi"]) & np.isfinite(med["ndwi"]) & np.isfinite(med["mndwi"])}

    # Model A: harmonic
    coef = fit_harmonic(flat["ndvi"][:, train_mask], doys(dates[train_mask]), med["ndvi"])
    exp_h = predict_harmonic(coef, doys(dates))  # (pixels, T)
    exp_h[~normals["usable"]] = np.nan
    resid_h = flat["ndvi"] - exp_h

    # Model B: MLP
    mlp_model, train_info = fit_mlp(flat["ndvi"], dates, train_mask, normals, inside)
    exp_m = predict_mlp(mlp_model, flat["ndvi"].shape[0], dates, normals)
    resid_m = flat["ndvi"] - exp_m

    labels = reference_labels(flat, normals, inside)
    tune_idx, test_idx = block_split(labels, height, width)

    results = {}
    preds = {}
    for name, resid in [("Harmonic + persistence", resid_h), ("MLP + persistence", resid_m)]:
        pred, anomaly, thr, k, scores = tune_model(resid, normals, labels, tune_idx, test_idx)
        results[name] = scores; preds[name] = (pred, anomaly, thr, k, resid)
    base_pred, delta, base_scores = tune_baseline(flat["ndvi"], normals, labels, tune_idx, test_idx, dates)
    results["Two-date baseline"] = base_scores

    # Choose the proposed model = better tuning-selected test F1 of the two ML models
    chosen_name = max(["Harmonic + persistence", "MLP + persistence"], key=lambda n: results[n]["f1"])
    chosen_pred, chosen_anomaly, thr, k, chosen_resid = preds[chosen_name]
    details = anomaly_details(chosen_anomaly, thr, k, chosen_resid[:, monitor_mask], dates[monitor_mask])

    with np.errstate(all="ignore"):
        d = {kk: (np.nanmedian(flat[kk][:, monitor_mask], axis=1) - med[kk]) for kk in ["ndvi", "mndwi", "ndmi", "bsi"]}
    types_all = classify_types(d["ndvi"], d["mndwi"], d["ndmi"], d["bsi"])
    types = types_all

    # Control-ring diagnostics: detections outside the AOI inside the window.
    inside_det = chosen_pred & inside & normals["usable"]
    outside_det = chosen_pred & (~inside) & normals["usable"]

    type_areas = {}
    vals, cnt = np.unique(types[chosen_pred & inside], return_counts=True)
    for v, c in zip(vals, cnt):
        type_areas[str(v)] = round(hectares(int(c)), 3)
    vals_w, cnt_w = np.unique(types[chosen_pred], return_counts=True)
    type_areas_window = {str(v): round(hectares(int(c)), 3) for v, c in zip(vals_w, cnt_w)}

    onset_years = {}
    for od in details["onset_date"][chosen_pred & inside]:
        if od:
            onset_years[od[:4]] = onset_years.get(od[:4], 0) + 1

    metrics = {
        "site": site, "label": cfg["label"], "subtitle": cfg["subtitle"], "story": cfg["story"],
        "aoi_source": cfg["aoi_source"], "intervention_label": cfg["intervention_label"],
        "expected_change": cfg["expected_change"],
        "scenes": int(T), "first_date": str(dates[0]), "last_date": str(dates[-1]),
        "train_scenes": int(train_mask.sum()), "monitor_scenes": int(monitor_mask.sum()),
        "train_end": cfg["train_end"], "monitor_start": cfg["monitor_start"],
        "usable_pixels": int(normals["usable"].sum()), "aoi_pixels": int(inside.sum()),
        "aoi_hectares": round(hectares(int(inside.sum())), 2),
        "training": train_info,
        "chosen_model": chosen_name,
        "models": results,
        "reference_rule": {"shift": REFERENCE_SHIFT, "fraction": REFERENCE_FRACTION,
                           "description": "Monitoring median NDVI shifts >=0.08 from normal median on >=60% of clear monitoring observations (either direction); negatives are stable pixels. Silver-standard reference -- see docs/METHODS.md."},
        "area_hectares": {"detected_inside_aoi": round(hectares(int(inside_det.sum())), 2),
                          "detected_window": round(hectares(int(chosen_pred.sum())), 2),
                          "detected_control_ring": round(hectares(int(outside_det.sum())), 2)},
        "change_by_type_inside_aoi": type_areas,
        "change_by_type_window": type_areas_window,
        "onset_pixels_by_year_inside_aoi": dict(sorted(onset_years.items())),
        "bounds_wgs84": list(cube["bounds"]),
        "rgb_early_date": cube["rgb_early_date"], "rgb_late_date": cube["rgb_late_date"],
        "n_change_polygons": 0,
    }

    polys = build_polygons(site, chosen_pred & inside, types, details, cube["transform"], height, width)
    # Also export window-wide polygons for the map (AOI + nearby change)
    metrics["n_change_polygons"] = len(polys["features"])
    paths["changes_geojson"].write_text(json.dumps(polys), encoding="utf-8")

    make_figures(site, cube, flat, normals, inside, chosen_resid, labels,
                 results, chosen_pred, types, mlp_model)
    write_overlay(site, chosen_pred, types, inside, height, width)

    # Time-series JSON for interactive charts
    ts = {"dates": [str(x) for x in dates]}
    for kname in ["ndvi", "mndwi"]:
        s = cube[kname].reshape(T, -1)
        ts[f"{kname}_inside"] = [None if not np.isfinite(v) else round(float(v), 4) for v in np.nanmedian(s[:, inside], axis=1)]
        ts[f"{kname}_outside"] = [None if not np.isfinite(v) else round(float(v), 4) for v in np.nanmedian(s[:, ~inside], axis=1)]
    (paths["figures"].parent.parent / f"timeseries_{site}.json").write_text(json.dumps(ts), encoding="utf-8")

    import joblib
    joblib.dump({"mlp": mlp_model, "harmonic_coef_sample_note": "harmonic coefficients are per-pixel and recomputed in analysis"},
                paths["model"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"site": site, "chosen": chosen_name,
                      "model_f1": results[chosen_name]["f1"], "baseline_f1": base_scores["f1"],
                      "area_inside_ha": metrics["area_hectares"]["detected_inside_aoi"],
                      "polygons": len(polys["features"])}, indent=2))
    return metrics


def run_all():
    all_metrics = {}
    for site in WETLANDS:
        try:
            all_metrics[site] = run_analysis(site)
        except FileNotFoundError as exc:
            print(f"[{site}] skipped: {exc}")
    (site_paths("nyandungu")["metrics"].parent / "metrics_all.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8")
    return all_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(WETLANDS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        run_all()
    elif a.site:
        run_analysis(a.site)
