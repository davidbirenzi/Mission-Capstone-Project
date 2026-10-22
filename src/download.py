"""Download a local Sentinel-2 time series for one or all study wetlands.

Data source (hybrid approach): the public Element84 Earth Search STAC catalog
(Sentinel-2 L2A, tile MGRS-35MRT area) is used to build an analysis-ready,
offline cube per wetland. The web app never depends on a live satellite
service. Google Earth Engine can be used as an alternative acquisition route
for the same cube layout (see docs/DATA.md).

Per kept scene we store six spectral indices (NDVI, NDWI, MNDWI, NDMI, SAVI,
BSI), a clear-pixel mask and a true-colour preview. One best scene per quarter
2017-2026 is kept, resumable per quarter.

Usage:
    python -m src.download --site gikondo
    python -m src.download --all
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from src.config import (
    CLEAR_SCL, CLOUD_COVER_MAX, COLLECTION, DST_CRS, INDICES,
    MIN_CLEAR_FRACTION, PROCESSED_DIR, RAW_DIR, RESOLUTION_M,
    SEARCH_END, SEARCH_START, STAC_URL, WETLANDS, ATTEMPTS_PER_QUARTER,
    site_paths,
)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

BAND_ASSETS = {
    "blue": ("blue", "B02"), "green": ("green", "B03"), "red": ("red", "B04"),
    "nir": ("nir", "B08"), "swir1": ("swir16", "B11"), "swir2": ("swir22", "B12"),
    "scl": ("scl", "SCL"),
}


def study_grid(bounds):
    west, south, east, north = bounds
    left, bottom, right, top = transform_bounds("EPSG:4326", DST_CRS, west, south, east, north)
    minx = np.floor(left / RESOLUTION_M) * RESOLUTION_M
    maxx = np.ceil(right / RESOLUTION_M) * RESOLUTION_M
    miny = np.floor(bottom / RESOLUTION_M) * RESOLUTION_M
    maxy = np.ceil(top / RESOLUTION_M) * RESOLUTION_M
    width = int(round((maxx - minx) / RESOLUTION_M))
    height = int(round((maxy - miny) / RESOLUTION_M))
    return from_origin(minx, maxy, RESOLUTION_M, RESOLUTION_M), height, width


def _asset_href(item, names):
    for name in names:
        asset = item.assets.get(name)
        if asset is not None:
            return asset.href
    raise KeyError(f"None of {names} found in {list(item.assets)}")


def _read_onto_grid(href, bounds, dst_transform, height, width, resampling, dtype):
    west, south, east, north = bounds
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 2 or window.height < 2:
            return None
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        source = src.read(1, window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs
    destination = np.zeros((height, width), dtype=dtype)
    reproject(source=source, destination=destination, src_transform=src_transform,
              src_crs=src_crs, src_nodata=0, dst_transform=dst_transform,
              dst_crs=DST_CRS, dst_nodata=0, resampling=resampling)
    return destination


def _safe_div(num, den):
    out = np.full(num.shape, np.nan, dtype=np.float32)
    ok = den != 0
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def compute_indices(b, clear):
    """b: dict of reflectance arrays (0..~1). Returns dict of index arrays."""
    red, green, blue, nir, swir1 = b["red"], b["green"], b["blue"], b["nir"], b["swir1"]
    idx = {
        "ndvi": _safe_div(nir - red, nir + red),
        "ndwi": _safe_div(green - nir, green + nir),
        "mndwi": _safe_div(green - swir1, green + swir1),
        "ndmi": _safe_div(nir - swir1, nir + swir1),
        "savi": _safe_div(1.5 * (nir - red), nir + red + 0.5),
        "bsi": _safe_div((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue)),
    }
    for key in idx:
        idx[key][~clear] = np.nan
    return idx


def _rgb(red, green, blue, clear):
    rgb = np.zeros(red.shape + (3,), dtype=np.uint8)
    for i, band in enumerate((red, green, blue)):
        values = band[clear]
        if values.size == 0:
            continue
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1e-6
        channel = np.clip((band - low) / (high - low), 0, 1)
        channel[~clear] = 0
        rgb[:, :, i] = (channel * 255).astype(np.uint8)
    return rgb


def _asset_scale_offset(item, names):
    """Per-asset radiometric scale from STAC raster:bands (offset NOT applied).

    Verified empirically on this catalogue (Oct 2026, scripts/check_offset3.py):
    Earth Search COG digital numbers are already harmonised across processing
    baselines -- median DN over the same ground is ~850 (green) / ~2100 (NIR)
    for baseline 00.01, 04.00, 05.00 and 05.09 scenes alike, and index values
    are temporally stable only when reflectance = DN * scale. Applying the
    -0.1 metadata offset a second time drives green reflectance negative over
    vegetation (clipped to 0) and collapses MNDWI to -1. The STAC offset field
    describes the original Copernicus product, not these COG DNs. See
    docs/METHODS.md.
    """
    for name in names:
        asset = item.assets.get(name)
        if asset is not None:
            rb = (asset.extra_fields.get("raster:bands") or [{}])[0]
            return asset.href, float(rb.get("scale", 0.0001)), 0.0
    raise KeyError(f"None of {names} found in {list(item.assets)}")


def _scene_arrays(item, bounds, dst_transform, height, width):
    jobs = []
    for name, assets in BAND_ASSETS.items():
        href, scale, offset = _asset_scale_offset(item, assets) if name != "scl" else (_asset_href(item, assets), 1.0, 0.0)
        jobs.append((name, href, scale, offset,
                     Resampling.nearest if name == "scl" else Resampling.bilinear,
                     np.uint8 if name == "scl" else np.float32))

    def _load(job):
        name, href, scale, offset, resampling, dtype = job
        arr = _read_onto_grid(href, bounds, dst_transform, height, width, resampling, dtype)
        if arr is not None and name != "scl":
            arr = np.maximum(arr.astype(np.float32) * scale + offset, 0.0)
        return name, arr

    with ThreadPoolExecutor(max_workers=7) as pool:
        bands = dict(pool.map(_load, jobs))
    if any(v is None for v in bands.values()):
        return None
    scl = bands.pop("scl")
    clear = np.isin(scl, list(CLEAR_SCL)) & (bands["red"] > 0) & (bands["nir"] > 0)
    indices = compute_indices(bands, clear)
    rgb = _rgb(bands["red"], bands["green"], bands["blue"], clear)
    return indices, clear, rgb


def _clear_fraction(item, bounds, dst_transform, height, width):
    scl = _read_onto_grid(_asset_href(item, BAND_ASSETS["scl"]), bounds,
                          dst_transform, height, width, Resampling.nearest, np.uint8)
    if scl is None:
        return None
    return float(np.isin(scl, list(CLEAR_SCL)).mean())


def _quarter(dt):
    return dt.year, (dt.month - 1) // 3 + 1


def search_items(bounds):
    import pystac_client
    catalog = pystac_client.Client.open(STAC_URL)
    west, south, east, north = bounds
    search = catalog.search(collections=[COLLECTION], bbox=[west, south, east, north],
                            datetime=f"{SEARCH_START}/{SEARCH_END}",
                            query={"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}}, max_items=900)
    items = list(search.items())
    items.sort(key=lambda it: (it.datetime, it.properties.get("eo:cloud_cover", 100)))
    return items


def download(site: str, force: bool = False):
    cfg = WETLANDS[site]
    paths = site_paths(site)
    bounds = cfg["bounds_wgs84"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if paths["cube"].exists() and not force:
        print(f"[{site}] cube already exists: {paths['cube']}")
        return paths["cube"]

    transform, height, width = study_grid(bounds)
    print(f"[{site}] grid {width} x {height}")
    items = search_items(bounds)
    print(f"[{site}] STAC scenes cloud<{CLOUD_COVER_MAX}%: {len(items)}")
    grouped = {}
    for item in items:
        grouped.setdefault(_quarter(item.datetime), []).append(item)
    for v in grouped.values():
        v.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))

    scene_dir = RAW_DIR / site / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    done_path = RAW_DIR / site / "quarters_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    catalog_rows = []
    if paths["catalog"].exists():
        catalog_rows = pd.read_csv(paths["catalog"]).to_dict("records")

    for quarter in sorted(grouped):
        qkey = f"{quarter[0]}-Q{quarter[1]}"
        if qkey in done:
            continue
        chosen = None
        for item in grouped[quarter][:ATTEMPTS_PER_QUARTER]:
            when = item.datetime.strftime("%Y-%m-%d")
            cloud = item.properties.get("eo:cloud_cover")
            try:
                fraction = _clear_fraction(item, bounds, transform, height, width)
                if fraction is None or fraction < MIN_CLEAR_FRACTION:
                    continue
                arrays = _scene_arrays(item, bounds, transform, height, width)
            except Exception as exc:
                print(f"[{site}] {when} skipped ({exc})", flush=True)
                continue
            if arrays is None:
                continue
            indices, clear, rgb = arrays
            chosen = {"date": when, "item_id": item.id, "cloud_cover": cloud,
                      "mgrs": item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile"),
                      "clear_fraction": round(float(fraction), 3)}
            np.savez_compressed(scene_dir / f"{when}.npz",
                                **{k: v.astype(np.float16) for k, v in indices.items()},
                                clear=clear.astype(np.uint8), rgb=rgb)
            break
        if chosen:
            catalog_rows.append(chosen)
            pd.DataFrame(catalog_rows).drop_duplicates("date").sort_values("date").to_csv(paths["catalog"], index=False)
            with done_path.open("a") as fh:
                fh.write(qkey + "\n")
            print(f"[{site}] kept {chosen['date']} clear={chosen['clear_fraction']}", flush=True)
        else:
            print(f"[{site}] no usable scene {qkey}", flush=True)

    scene_files = sorted(scene_dir.glob("*.npz"))
    if len(scene_files) < 10:
        raise RuntimeError(f"[{site}] only {len(scene_files)} scenes kept; need a longer series.")
    loaded = [np.load(p) for p in scene_files]
    dates = [p.stem for p in scene_files]
    catalog = pd.read_csv(paths["catalog"]).drop_duplicates("date").sort_values("date")
    catalog.to_csv(paths["catalog"], index=False)
    cloud_by_date = {r["date"]: (r["cloud_cover"] if pd.notna(r["cloud_cover"]) else 100)
                     for r in catalog.to_dict("records")}
    early_date = min(dates[:max(3, len(dates)//4)], key=lambda d: cloud_by_date.get(d, 100))
    late_date = min(dates[-max(3, len(dates)//4):], key=lambda d: cloud_by_date.get(d, 100))
    early = loaded[dates.index(early_date)]; late = loaded[dates.index(late_date)]
    np.savez_compressed(
        paths["cube"],
        **{k: np.stack([s[k] for s in loaded]).astype(np.float16) for k in INDICES},
        clear=np.stack([s["clear"] for s in loaded]).astype(np.uint8),
        dates=np.array(dates), transform=np.array(transform)[:6], crs=np.array(DST_CRS),
        bounds_wgs84=np.array(bounds), height=np.array(height), width=np.array(width),
        rgb_early=early["rgb"], rgb_late=late["rgb"],
        rgb_early_date=np.array(early_date), rgb_late_date=np.array(late_date),
    )
    print(f"[{site}] saved {len(dates)} scenes -> {paths['cube']}")
    return paths["cube"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(WETLANDS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sites = list(WETLANDS) if args.all else [args.site]
    for s in sites:
        if s:
            download(s, force=args.force)
