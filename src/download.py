"""Download a local Sentinel-2 time series for Nyandungu.

Scenes come from the public Element84 Earth Search catalog. Each selected
scene is clipped to the study window and stored in one compressed cube so
later training and the web demo do not need a network connection.
"""

from __future__ import annotations

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
    BOUNDS_WGS84,
    CATALOG_PATH,
    CLEAR_SCL,
    CLOUD_COVER_MAX,
    COLLECTION,
    CUBE_PATH,
    DST_CRS,
    MIN_CLEAR_FRACTION,
    PROCESSED_DIR,
    RAW_DIR,
    RESOLUTION_M,
    SEARCH_END,
    SEARCH_START,
    STAC_URL,
    ATTEMPTS_PER_QUARTER,
)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")


def study_grid():
    """North-up 10 m grid covering the study window, in UTM zone 36S."""
    west, south, east, north = BOUNDS_WGS84
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", DST_CRS, west, south, east, north
    )
    minx = np.floor(left / RESOLUTION_M) * RESOLUTION_M
    maxx = np.ceil(right / RESOLUTION_M) * RESOLUTION_M
    miny = np.floor(bottom / RESOLUTION_M) * RESOLUTION_M
    maxy = np.ceil(top / RESOLUTION_M) * RESOLUTION_M
    width = int(round((maxx - minx) / RESOLUTION_M))
    height = int(round((maxy - miny) / RESOLUTION_M))
    transform = from_origin(minx, maxy, RESOLUTION_M, RESOLUTION_M)
    return transform, height, width


def _asset_href(item, *names):
    for name in names:
        asset = item.assets.get(name)
        if asset is not None:
            return asset.href
    raise KeyError(f"None of {names} found in {list(item.assets)}")


def _read_onto_grid(href, dst_transform, height, width, resampling, dtype):
    west, south, east, north = BOUNDS_WGS84
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, west, south, east, north
        )
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 2 or window.height < 2:
            return None
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        source = src.read(1, window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs

    destination = np.zeros((height, width), dtype=dtype)
    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=DST_CRS,
        dst_nodata=0,
        resampling=resampling,
    )
    return destination


def _rgb(red, green, blue, clear, scale):
    rgb = np.zeros(red.shape + (3,), dtype=np.uint8)
    for i, band in enumerate((red, green, blue)):
        values = band[clear] / scale
        if values.size == 0:
            continue
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            high = low + 1e-6
        channel = np.clip((band / scale - low) / (high - low), 0, 1)
        channel[~clear] = 0
        rgb[:, :, i] = (channel * 255).astype(np.uint8)
    return rgb


def _clear_fraction(item, dst_transform, height, width):
    scl_href = _asset_href(item, "scl", "SCL")
    scl = _read_onto_grid(
        scl_href, dst_transform, height, width, Resampling.nearest, np.uint8
    )
    if scl is None:
        return None
    return float(np.isin(scl, list(CLEAR_SCL)).mean())


def _scene_arrays(item, dst_transform, height, width):
    jobs = [
        ("red", _asset_href(item, "red", "B04"), Resampling.bilinear, np.float32),
        ("green", _asset_href(item, "green", "B03"), Resampling.bilinear, np.float32),
        ("blue", _asset_href(item, "blue", "B02"), Resampling.bilinear, np.float32),
        ("nir", _asset_href(item, "nir", "B08"), Resampling.bilinear, np.float32),
        ("scl", _asset_href(item, "scl", "SCL"), Resampling.nearest, np.uint8),
    ]

    def _load(job):
        name, href, resampling, dtype = job
        return name, _read_onto_grid(href, dst_transform, height, width, resampling, dtype)

    with ThreadPoolExecutor(max_workers=5) as pool:
        bands = dict(pool.map(_load, jobs))
    red, green, blue, nir, scl = (bands[name] for name in ("red", "green", "blue", "nir", "scl"))
    if any(band is None for band in (red, green, blue, nir, scl)):
        return None
    scale = 10000.0 if np.nanmedian(nir[nir > 0]) > 3 else 1.0
    clear = np.isin(scl, list(CLEAR_SCL)) & (red > 0) & (green > 0) & (nir > 0)
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndwi = np.full(red.shape, np.nan, dtype=np.float32)
    ok_v = clear & ((nir + red) != 0)
    ok_w = clear & ((green + nir) != 0)
    ndvi[ok_v] = (nir[ok_v] - red[ok_v]) / (nir[ok_v] + red[ok_v])
    ndwi[ok_w] = (green[ok_w] - nir[ok_w]) / (green[ok_w] + nir[ok_w])
    rgb = _rgb(red, green, blue, clear, scale)
    return ndvi, ndwi, clear, rgb


def _quarter(dt):
    return dt.year, (dt.month - 1) // 3 + 1


def search_items():
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL)
    west, south, east, north = BOUNDS_WGS84
    search = catalog.search(
        collections=[COLLECTION],
        bbox=[west, south, east, north],
        datetime=f"{SEARCH_START}/{SEARCH_END}",
        query={"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}},
        max_items=800,
    )
    items = list(search.items())
    items.sort(key=lambda item: (item.datetime, item.properties.get("eo:cloud_cover", 100)))
    return items


def download(force=False):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CUBE_PATH.exists() and not force:
        print(f"Cube already exists at {CUBE_PATH}. Pass force=True to rebuild it.")
        return CUBE_PATH

    transform, height, width = study_grid()
    print(f"Study grid: {width} x {height} pixels at {RESOLUTION_M} m")
    items = search_items()
    print(f"STAC scenes with cloud cover under {CLOUD_COVER_MAX}%: {len(items)}")
    if not items:
        raise RuntimeError("No Sentinel-2 scenes were returned for the Nyandungu window.")

    grouped = {}
    for item in items:
        grouped.setdefault(_quarter(item.datetime), []).append(item)
    for quarter_items in grouped.values():
        quarter_items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))

    # Park mask is applied after the cube is built. During download, require
    # the study window itself to be mostly clear. Each kept quarter is written
    # to disk so a restarted download continues from the next quarter.
    scene_dir = RAW_DIR / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    done_path = RAW_DIR / "quarters_done.txt"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    selected = []
    for quarter in sorted(grouped):
        quarter_key = f"{quarter[0]}-{quarter[1]}"
        if quarter_key in done:
            print(f"  quarter {quarter_key} already saved", flush=True)
            continue
        chosen = None
        for item in grouped[quarter][:ATTEMPTS_PER_QUARTER]:
            when = item.datetime.strftime("%Y-%m-%d")
            cloud = item.properties.get("eo:cloud_cover")
            print(f"  trying {when} cloud={cloud}", flush=True)
            try:
                fraction = _clear_fraction(item, transform, height, width)
            except Exception as exc:
                print(f"    skipped ({exc})", flush=True)
                continue
            if fraction is None:
                print("    skipped (scene does not cover the window)", flush=True)
                continue
            print(f"    clear fraction {fraction:.2f}", flush=True)
            if fraction < MIN_CLEAR_FRACTION:
                continue
            try:
                arrays = _scene_arrays(item, transform, height, width)
            except Exception as exc:
                print(f"    skipped while reading bands ({exc})", flush=True)
                continue
            if arrays is None:
                continue
            ndvi, ndwi, clear, rgb = arrays
            chosen = {
                "date": when,
                "item_id": item.id,
                "cloud_cover": cloud,
                "mgrs": item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile"),
                "clear_fraction": round(fraction, 3),
            }
            np.savez_compressed(
                scene_dir / f"{when}.npz",
                ndvi=ndvi.astype(np.float16),
                ndwi=ndwi.astype(np.float16),
                clear=clear.astype(np.uint8),
                rgb=rgb,
            )
            break
        if chosen:
            selected.append(chosen)
            pd.DataFrame([chosen]).to_csv(
                CATALOG_PATH, mode="a", header=not CATALOG_PATH.exists(), index=False
            )
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(quarter_key + "\n")
            print(f"  kept {chosen['date']}", flush=True)

    scene_files = sorted(scene_dir.glob("*.npz"))
    if len(scene_files) < 8:
        raise RuntimeError(
            f"Only {len(scene_files)} clear scenes were kept. Need a longer time series."
        )

    catalog = pd.read_csv(CATALOG_PATH).drop_duplicates("date").sort_values("date")
    catalog.to_csv(CATALOG_PATH, index=False)

    loaded = [np.load(path) for path in scene_files]
    dates = [path.stem for path in scene_files]
    pre_dates = [date for date in dates if date < "2018-09-01"] or dates
    post_dates = [date for date in dates if date >= "2022-07-08"] or dates
    # Prefer the scene whose catalog cloud cover is lowest.
    cloud_by_date = {
        row["date"]: row["cloud_cover"] if pd.notna(row["cloud_cover"]) else 100
        for row in catalog.to_dict(orient="records")
    }
    early_date = min(pre_dates, key=lambda date: cloud_by_date.get(date, 100))
    late_date = min(post_dates, key=lambda date: cloud_by_date.get(date, 100))
    early = next(scene for scene, date in zip(loaded, dates) if date == early_date)
    late = next(scene for scene, date in zip(loaded, dates) if date == late_date)

    np.savez_compressed(
        CUBE_PATH,
        ndvi=np.stack([scene["ndvi"] for scene in loaded]).astype(np.float16),
        ndwi=np.stack([scene["ndwi"] for scene in loaded]).astype(np.float16),
        clear=np.stack([scene["clear"] for scene in loaded]).astype(np.uint8),
        dates=np.array(dates),
        transform=np.array(transform)[:6],
        crs=np.array(DST_CRS),
        bounds_wgs84=np.array(BOUNDS_WGS84),
        height=np.array(height),
        width=np.array(width),
        rgb_early=early["rgb"],
        rgb_late=late["rgb"],
        rgb_early_date=np.array(early_date),
        rgb_late_date=np.array(late_date),
    )
    print(f"Saved {len(scene_files)} scenes to {CUBE_PATH}")
    return CUBE_PATH


if __name__ == "__main__":
    download()
