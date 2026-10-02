"""Learn Nyandungu's normal seasonal pattern and flag persistent deviations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPRegressor

from src.config import (
    AOI_PATH,
    CATALOG_PATH,
    CUBE_PATH,
    DST_CRS,
    FIGURE_DIR,
    MAP_DIR,
    METRICS_PATH,
    MIN_POST_OBS,
    MIN_PRE_OBS,
    MLP_PARAMS,
    MODEL_PATH,
    POST_START,
    PRE_END,
    REFERENCE_NDVI_SHIFT,
    REFERENCE_PERSIST_FRACTION,
    RESOLUTION_M,
)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#1f2933",
        "axes.labelcolor": "#1f2933",
        "xtick.color": "#1f2933",
        "ytick.color": "#1f2933",
        "text.color": "#1f2933",
        "axes.titleweight": "medium",
        "font.size": 11,
    }
)

PARK = "#1d4e89"
CONTROL = "#6b7280"
CHANGE = "#b42318"
INK = "#1f2933"


def load_cube():
    if not CUBE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CUBE_PATH}. Run python -m src.download first."
        )
    data = np.load(CUBE_PATH, allow_pickle=False)
    ndvi = data["ndvi"].astype(np.float32)
    ndwi = data["ndwi"].astype(np.float32)
    ndvi[data["clear"] == 0] = np.nan
    ndwi[data["clear"] == 0] = np.nan
    dates = np.array([str(value) for value in data["dates"]])
    transform = Affine(*data["transform"].tolist())
    return {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "dates": dates,
        "transform": transform,
        "bounds": tuple(float(v) for v in data["bounds_wgs84"]),
        "rgb_early": data["rgb_early"],
        "rgb_late": data["rgb_late"],
        "rgb_early_date": str(data["rgb_early_date"]),
        "rgb_late_date": str(data["rgb_late_date"]),
        "crs": str(data["crs"]),
    }


def park_mask(shape_hw, transform):
    """Raster of the OSM Nyandungu park polygon on the study grid."""
    from pyproj import Transformer

    raw = json.loads(Path(AOI_PATH).read_text(encoding="utf-8"))
    geom = shape(raw["features"][0]["geometry"])
    transformer = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    geom_utm = shapely_transform(transformer.transform, geom)
    mask = rasterize(
        [(geom_utm, 1)],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _period_index(dates, start=None, end=None):
    keep = np.ones(len(dates), dtype=bool)
    if start:
        keep &= dates >= start
    if end:
        keep &= dates < end
    return keep


def pixel_normals(ndvi, ndwi, dates):
    pre = _period_index(dates, end=PRE_END)
    post = _period_index(dates, start=POST_START)
    pre_ndvi = ndvi[:, pre]
    pre_ndwi = ndwi[:, pre]
    post_ndvi = ndvi[:, post]
    pre_count = np.sum(~np.isnan(pre_ndvi), axis=1)
    post_count = np.sum(~np.isnan(post_ndvi), axis=1)
    with np.errstate(all="ignore"):
        pre_median_ndvi = np.nanmedian(pre_ndvi, axis=1)
        pre_median_ndwi = np.nanmedian(pre_ndwi, axis=1)
    usable = (pre_count >= MIN_PRE_OBS) & (post_count >= MIN_POST_OBS)
    usable &= np.isfinite(pre_median_ndvi) & np.isfinite(pre_median_ndwi)
    return {
        "pre": pre,
        "post": post,
        "pre_count": pre_count,
        "post_count": post_count,
        "pre_median_ndvi": pre_median_ndvi,
        "pre_median_ndwi": pre_median_ndwi,
        "usable": usable,
    }


def seasonal_matrix(doys, pre_median_ndvi, pre_median_ndwi):
    """Features for one date across many pixels: season plus the pixel's normal level."""
    angle = 2 * np.pi * doys / 365.25
    season = np.array(
        [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)],
        dtype=np.float32,
    )
    # doys is a scalar here when called per date. season shape (4,)
    n = pre_median_ndvi.shape[0]
    features = np.empty((n, 6), dtype=np.float32)
    features[:, 0:4] = season.reshape(1, 4)
    features[:, 4] = pre_median_ndvi
    features[:, 5] = pre_median_ndwi
    return features


def day_of_year(dates):
    return np.array([pd.Timestamp(str(value)).day_of_year for value in dates], dtype=np.int16)


def fit_seasonal_mlp(ndvi, dates, normals, park, sample_size=6000):
    """Train on pre-restoration wetland pixels only."""
    pre_dates = dates[normals["pre"]]
    doys = day_of_year(pre_dates)
    candidates = np.flatnonzero(normals["usable"] & park)
    rng = np.random.default_rng(42)
    if candidates.size == 0:
        raise RuntimeError("No usable wetland pixels in the pre-restoration period.")
    if candidates.size > sample_size:
        candidates = rng.choice(candidates, size=sample_size, replace=False)

    blocks = []
    targets = []
    pre_cube = ndvi[:, normals["pre"]]
    for date_i, doy in enumerate(doys):
        observed = pre_cube[candidates, date_i]
        valid = np.isfinite(observed)
        if valid.sum() < 50:
            continue
        rows = candidates[valid]
        features = seasonal_matrix(
            doy,
            normals["pre_median_ndvi"][rows],
            normals["pre_median_ndwi"][rows],
        )
        blocks.append(features)
        targets.append(observed[valid])

    x_train = np.vstack(blocks)
    y_train = np.concatenate(targets)
    model = MLPRegressor(**MLP_PARAMS)
    model.fit(x_train, y_train)
    return model, {"n_samples": int(x_train.shape[0]), "n_pixels": int(candidates.size)}


def predict_ndvi(model, ndvi_shape, dates, normals):
    """Expected NDVI on every date for every usable pixel. Other pixels stay NaN."""
    n_pixels = ndvi_shape[0]
    expected = np.full((n_pixels, len(dates)), np.nan, dtype=np.float32)
    usable = normals["usable"]
    rows = np.flatnonzero(usable)
    doys = day_of_year(dates)
    level_v = normals["pre_median_ndvi"][rows]
    level_w = normals["pre_median_ndwi"][rows]
    for date_i, doy in enumerate(doys):
        features = seasonal_matrix(doy, level_v, level_w)
        expected[rows, date_i] = model.predict(features).astype(np.float32)
    return expected


def max_true_run(mask):
    """Longest run of True values along axis 1."""
    best = np.zeros(mask.shape[0], dtype=np.int16)
    run = np.zeros(mask.shape[0], dtype=np.int16)
    for column in range(mask.shape[1]):
        run = (run + 1) * mask[:, column]
        best = np.maximum(best, run)
    return best


def reference_labels(ndvi, dates, normals, park):
    """Independent labels from a fixed NDVI-shift rule, not from the MLP."""
    post = ndvi[:, normals["post"]]
    pre_level = normals["pre_median_ndvi"][:, None]
    shifted = (post - pre_level) >= REFERENCE_NDVI_SHIFT
    observed = np.isfinite(post)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.sum(shifted & observed, axis=1) / np.maximum(np.sum(observed, axis=1), 1)
    persistent = fraction >= REFERENCE_PERSIST_FRACTION
    labels = np.full(ndvi.shape[0], -1, dtype=np.int8)
    positive = normals["usable"] & park & persistent
    negative = normals["usable"] & (~park) & (~persistent)
    labels[positive] = 1
    labels[negative] = 0
    return labels


def block_split(labels, height, width, block=10, test_fraction=0.35):
    """Hold out whole blocks of pixels so neighboring pixels do not leak into tuning."""
    rows, cols = np.divmod(np.arange(height * width), width)
    block_ids = (rows // block) * ((width // block) + 1) + (cols // block)
    tune = np.zeros(labels.shape[0], dtype=bool)
    test = np.zeros(labels.shape[0], dtype=bool)
    rng = np.random.default_rng(42)
    for cls in (0, 1):
        members = np.flatnonzero(labels == cls)
        blocks = np.unique(block_ids[members])
        rng.shuffle(blocks)
        n_test = max(1, int(round(len(blocks) * test_fraction)))
        test_blocks = set(blocks[:n_test])
        is_test = np.isin(block_ids[members], list(test_blocks))
        test[members[is_test]] = True
        tune[members[~is_test]] = True
    return tune, test


def _tuning_f1(y_true, y_pred):
    """F1 on every positive and at most five negatives per positive.

    The full map is mostly stable ground. Choosing the threshold on that raw
    mix pushes the model toward calling nothing a change. The cap is used only
    while the threshold is selected. The reported scores use every test pixel.
    """
    rng = np.random.default_rng(0)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if positive.size == 0:
        return 0.0
    keep_negative = min(negative.size, positive.size * 5)
    chosen = rng.choice(negative, size=keep_negative, replace=False)
    index = np.concatenate([positive, chosen])
    return float(f1_score(y_true[index], y_pred[index], zero_division=0))


def _scores(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "iou": float(jaccard_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
    }


def detect_persistent(residual, post_columns, threshold, min_consecutive):
    post_residual = residual[:, post_columns]
    anomaly = np.isfinite(post_residual) & (post_residual >= threshold)
    runs = max_true_run(anomaly)
    return runs >= min_consecutive


def tune_and_evaluate(residual, ndvi, dates, normals, labels, tune, test):
    pre_residual = residual[tune][:, normals["pre"]]
    pre_values = np.abs(pre_residual[np.isfinite(pre_residual)])
    percentile_grid = [70, 80, 90, 95] if pre_values.size else []
    thresholds = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
    thresholds += [float(np.percentile(pre_values, q)) for q in percentile_grid]
    consecutives = [2, 3, 4]

    y_tune = labels[tune]
    best = None
    for threshold in thresholds:
        for consecutive in consecutives:
            predicted = detect_persistent(residual, normals["post"], threshold, consecutive)
            score = _tuning_f1(y_tune, predicted[tune])
            if best is None or score > best["f1"]:
                best = {
                    "f1": float(score),
                    "threshold": float(threshold),
                    "min_consecutive": int(consecutive),
                }

    model_mask = detect_persistent(
        residual, normals["post"], best["threshold"], best["min_consecutive"]
    )
    model_test = _scores(labels[test], model_mask[test])
    model_test["threshold"] = best["threshold"]
    model_test["min_consecutive"] = best["min_consecutive"]
    model_test["confusion_matrix"] = confusion_matrix(
        labels[test], model_mask[test], labels=[0, 1]
    ).tolist()

    # Baseline: the conventional two-date comparison. It uses the clearest
    # pre-restoration scene and the clearest post-opening scene, with no
    # seasonal model and no requirement that the change continue afterwards.
    pre_idx = np.flatnonzero(normals["pre"])
    post_idx = np.flatnonzero(normals["post"])
    pre_date = pre_idx[np.argmax(np.sum(np.isfinite(ndvi[:, pre_idx]), axis=0))]
    post_date = post_idx[np.argmax(np.sum(np.isfinite(ndvi[:, post_idx]), axis=0))]
    delta = ndvi[:, post_date] - ndvi[:, pre_date]
    delta_thresholds = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
    best_delta = None
    for threshold in delta_thresholds:
        predicted = delta >= threshold
        score = _tuning_f1(y_tune, predicted[tune])
        if best_delta is None or score > best_delta["f1"]:
            best_delta = {"f1": float(score), "threshold": float(threshold)}
    baseline_mask = delta >= best_delta["threshold"]
    baseline_test = _scores(labels[test], baseline_mask[test])
    baseline_test["threshold"] = best_delta["threshold"]
    baseline_test["pre_date"] = str(dates[pre_date])
    baseline_test["post_date"] = str(dates[post_date])
    baseline_test["confusion_matrix"] = confusion_matrix(
        labels[test], baseline_mask[test], labels=[0, 1]
    ).tolist()
    return model_mask, baseline_mask, delta, model_test, baseline_test


def _save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_figures(cube, park, normals, residual, labels, model_mask, metrics, model):
    ndvi = cube["ndvi"]
    ndwi = cube["ndwi"]
    dates = pd.to_datetime(cube["dates"])
    height, width = ndvi.shape[1], ndvi.shape[2]
    flat_park = park.reshape(-1)

    # Distributions before the works and after opening, inside the park.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    flat_series = ndvi.reshape(len(dates), -1)
    pre_vals = flat_series[normals["pre"]][:, flat_park].ravel()
    post_vals = flat_series[normals["post"]][:, flat_park].ravel()
    pre_vals = pre_vals[np.isfinite(pre_vals)]
    post_vals = post_vals[np.isfinite(post_vals)]
    ax.hist(pre_vals, bins=40, density=True, alpha=0.75, color=CONTROL, label="Before Sept 2018")
    ax.hist(post_vals, bins=40, density=True, alpha=0.65, color=PARK, label="After 8 July 2022")
    ax.set_title("NDVI inside Nyandungu")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _save(fig, "ndvi_distribution.png")

    # Correlation of the model inputs and the NDVI target, on a pixel sample.
    rng = np.random.default_rng(7)
    pool = np.flatnonzero(normals["usable"] & flat_park)
    sample = rng.choice(pool, size=min(1500, pool.size), replace=False)
    mid = len(dates) // 2
    doy = day_of_year(cube["dates"])[mid]
    features = seasonal_matrix(
        doy,
        normals["pre_median_ndvi"][sample],
        normals["pre_median_ndwi"][sample],
    )
    target = ndvi.reshape(len(dates), -1).T[sample, mid]
    table = np.column_stack([features[:, 0], features[:, 4], features[:, 5], target])
    frame = pd.DataFrame(table, columns=["season_sin", "normal_ndvi", "normal_ndwi", "ndvi"])
    corr = frame.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    image = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(4), frame.columns, rotation=30, ha="right")
    ax.set_yticks(range(4), frame.columns)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color=INK)
    ax.set_title("Feature correlation on a mid-series date")
    fig.colorbar(image, ax=ax, fraction=0.046)
    _save(fig, "correlation.png")

    # Time series: median NDVI in the park and in the control ring.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    park_median = np.nanmedian(ndvi.reshape(len(dates), -1)[:, flat_park], axis=1)
    control_median = np.nanmedian(ndvi.reshape(len(dates), -1)[:, ~flat_park], axis=1)
    ax.plot(dates, park_median, color=PARK, marker="o", label="Inside Nyandungu")
    ax.plot(dates, control_median, color=CONTROL, marker="o", label="Outside the park")
    ax.axvspan(pd.Timestamp("2018-09-01"), pd.Timestamp("2022-07-08"), color="#d9e2ec", alpha=0.45, label="Restoration works")
    ax.set_title("Median NDVI through the Sentinel-2 record")
    ax.set_ylabel("Median NDVI")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.autofmt_xdate()
    _save(fig, "ndvi_timeseries.png")

    # Residual of the learned normal pattern. residual is (pixels, dates).
    fig, ax = plt.subplots(figsize=(9, 4.6))
    park_residual = np.nanmedian(residual[flat_park], axis=0)
    control_residual = np.nanmedian(residual[~flat_park], axis=0)
    ax.plot(dates, park_residual, color=PARK, marker="o", label="Inside Nyandungu")
    ax.plot(dates, control_residual, color=CONTROL, marker="o", label="Outside the park")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvspan(pd.Timestamp("2018-09-01"), pd.Timestamp("2022-07-08"), color="#d9e2ec", alpha=0.45, label="Restoration works")
    ax.set_title("Deviation from the learned normal NDVI")
    ax.set_ylabel("Observed NDVI minus expected NDVI")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.autofmt_xdate()
    _save(fig, "residual_timeseries.png")

    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.set_axis_off()
    boxes = [
        (0.2, "6 inputs\nseason + normal\nNDVI and NDWI"),
        (2.9, "64 units\nReLU"),
        (5.6, "32 units\nReLU"),
        (8.3, "1 output\nexpected NDVI\nlinear"),
    ]
    box_width = 2.05
    for x, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, 1.05), box_width, 1.9, facecolor="#f4f6f8", edgecolor=INK, linewidth=1.2)
        )
        ax.text(x + box_width / 2, 2.0, text, ha="center", va="center", fontsize=9)
    for (x, _), (nxt, _) in zip(boxes, boxes[1:]):
        ax.annotate(
            "",
            xy=(nxt, 2.0),
            xytext=(x + box_width, 2.0),
            arrowprops={"arrowstyle": "->", "color": INK},
        )
    ax.set_title("MLPRegressor  ·  Adam  ·  squared error  ·  early stopping")
    _save(fig, "architecture.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(model.loss_curve_, color=PARK)
    ax.set_title("MLP training loss (squared error)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    _save(fig, "loss_curve.png")

    # Map of detected persistent change.
    detected = model_mask.reshape(height, width)
    show = np.full((height, width, 4), 0, dtype=np.float32)
    # Park footprint as a light blue wash, detections in red.
    show[park, :3] = (0.86, 0.91, 0.96)
    show[park, 3] = 0.85
    show[detected, 0] = 0.706
    show[detected, 1] = 0.137
    show[detected, 2] = 0.094
    show[detected, 3] = 0.92
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.imshow(show, interpolation="nearest")
    ax.set_title("Persistent change detected inside the study window")
    ax.set_axis_off()
    _save(fig, "change_map.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    names = ["Precision", "Recall", "F1", "Accuracy", "IoU"]
    keys = ["precision", "recall", "f1", "accuracy", "iou"]
    x = np.arange(len(names))
    proposed = [metrics["model"][key] for key in keys]
    baseline = [metrics["baseline"][key] for key in keys]
    ax.bar(x - 0.18, proposed, width=0.36, color=PARK, label="Seasonal MLP + persistence")
    ax.bar(x + 0.18, baseline, width=0.36, color=CONTROL, label="Two-date NDVI difference")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on held-out blocks")
    ax.set_title("Initial performance on the held-out pixels")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "metrics.png")

    # NDWI distribution, so water is visible as well as vegetation.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    flat_water = ndwi.reshape(len(dates), -1)
    pre_w = flat_water[normals["pre"]][:, flat_park].ravel()
    post_w = flat_water[normals["post"]][:, flat_park].ravel()
    pre_w = pre_w[np.isfinite(pre_w)]
    post_w = post_w[np.isfinite(post_w)]
    ax.hist(pre_w, bins=40, density=True, alpha=0.75, color=CONTROL, label="Before Sept 2018")
    ax.hist(post_w, bins=40, density=True, alpha=0.65, color=PARK, label="After 8 July 2022")
    ax.set_title("NDWI inside Nyandungu")
    ax.set_xlabel("NDWI (McFeeters)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _save(fig, "ndwi_distribution.png")


def write_map_overlay(model_mask, park, height, width):
    """Transparent PNG for the Leaflet overlay. North is up."""
    from PIL import Image

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    park_pixels = park.reshape(height, width)
    image[park_pixels, 0] = 29
    image[park_pixels, 1] = 78
    image[park_pixels, 2] = 137
    image[park_pixels, 3] = 70
    detected = model_mask.reshape(height, width)
    image[detected] = (180, 35, 24, 210)
    path = MAP_DIR / "persistent_change.png"
    Image.fromarray(image, mode="RGBA").save(path)
    return path


def hectares(count):
    return float(count * (RESOLUTION_M * RESOLUTION_M) / 10000.0)


def run_analysis():
    cube = load_cube()
    ndvi = cube["ndvi"]
    ndwi = cube["ndwi"]
    dates = cube["dates"]
    time_count, height, width = ndvi.shape
    flat_ndvi = ndvi.reshape(time_count, -1).T
    flat_ndwi = ndwi.reshape(time_count, -1).T
    park = park_mask((height, width), cube["transform"]).reshape(-1)

    normals = pixel_normals(flat_ndvi, flat_ndwi, dates)
    if normals["pre"].sum() < MIN_PRE_OBS:
        raise RuntimeError(
            f"Only {int(normals['pre'].sum())} scenes fall before {PRE_END}. "
            "The normal-pattern period needs at least "
            f"{MIN_PRE_OBS} clear scenes."
        )

    model, train_info = fit_seasonal_mlp(flat_ndvi, dates, normals, park)
    expected = predict_ndvi(model, flat_ndvi.shape, dates, normals)
    residual = flat_ndvi - expected
    labels = reference_labels(flat_ndvi, dates, normals, park)
    tune, test = block_split(labels, height, width)
    model_mask, baseline_mask, delta, model_scores, baseline_scores = tune_and_evaluate(
        residual, flat_ndvi, dates, normals, labels, tune, test
    )

    inside = park & normals["usable"]
    detected_inside = model_mask & inside
    metrics = {
        "study_area": "Nyandungu Urban Wetland Eco-Tourism Park, Kigali",
        "scenes": int(time_count),
        "pre_scenes": int(normals["pre"].sum()),
        "post_scenes": int(normals["post"].sum()),
        "transition_scenes": int((~normals["pre"] & ~normals["post"]).sum()),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "usable_pixels": int(normals["usable"].sum()),
        "park_pixels": int(park.sum()),
        "training": train_info,
        "architecture": {
            "type": "MLPRegressor",
            "input_features": [
                "sin(day of year)",
                "cos(day of year)",
                "sin(2 * day of year)",
                "cos(2 * day of year)",
                "pre-restoration median NDVI",
                "pre-restoration median NDWI",
            ],
            "hidden_layers": list(MLP_PARAMS["hidden_layer_sizes"]),
            "activation": MLP_PARAMS["activation"],
            "output": "expected NDVI (linear output)",
            "optimizer": MLP_PARAMS["solver"],
            "learning_rate": MLP_PARAMS["learning_rate_init"],
            "loss": "squared error",
            "early_stopping": True,
            "n_iter": int(model.n_iter_),
        },
        "reference_rule": {
            "ndvi_shift": REFERENCE_NDVI_SHIFT,
            "fraction_of_post_dates": REFERENCE_PERSIST_FRACTION,
            "direction": "NDVI increase above the pre-restoration median",
            "positive": "inside the park and the greening persists on at least 70 percent of post-opening dates",
            "negative": "outside the park and that greening does not persist",
        },
        "model": model_scores,
        "baseline": baseline_scores,
        "area_hectares": {
            "detected_persistent_window": hectares(int(model_mask.sum())),
            "detected_persistent_inside_park": hectares(int(detected_inside.sum())),
            "park": hectares(int(park.sum())),
        },
        "rgb_early_date": cube["rgb_early_date"],
        "rgb_late_date": cube["rgb_late_date"],
        "bounds_wgs84": list(cube["bounds"]),
    }

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_READY = METRICS_PATH.parent
    OUTPUT_DIR_READY.mkdir(parents=True, exist_ok=True)
    make_figures(
        cube,
        park.reshape(height, width),
        normals,
        residual,
        labels,
        model_mask,
        metrics,
        model,
    )
    write_map_overlay(model_mask, park, height, width)

    # Early and late true-color snapshots for the study-area page.
    from PIL import Image

    Image.fromarray(cube["rgb_early"], mode="RGB").save(FIGURE_DIR / "rgb_early.png")
    Image.fromarray(cube["rgb_late"], mode="RGB").save(FIGURE_DIR / "rgb_late.png")

    import joblib

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    catalog = pd.read_csv(CATALOG_PATH) if CATALOG_PATH.exists() else pd.DataFrame()
    summary = {
        "metrics": metrics,
        "catalog": catalog,
        "dates": dates,
        "park_median_ndvi_pre": float(
            np.nanmedian(flat_ndvi[park][:, normals["pre"]])
        ),
        "park_median_ndvi_post": float(
            np.nanmedian(flat_ndvi[park][:, normals["post"]])
        ),
    }
    print(json.dumps({"model": model_scores, "baseline": baseline_scores, "area": metrics["area_hectares"]}, indent=2))
    return summary


if __name__ == "__main__":
    run_analysis()
