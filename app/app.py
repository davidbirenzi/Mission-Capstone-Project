"""Flask web application for the Kigali wetland persistent-change monitor.

Local-first: the app reads pre-computed cubes' metrics, figures, map overlays
and change polygons from outputs/. It never needs a live satellite service.
PostgreSQL/PostGIS is the deployment database (see db/schema.sql and
src/load_postgis.py); the file store keeps the demo fully offline-capable.
"""

import csv
import io
import json

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from src.config import OUTPUT_DIR, ROOT, WETLANDS, site_paths

app = Flask(__name__)

NAV = [
    ("dashboard", "Dashboard", "/"),
    ("wetlands", "Wetlands", "/wetlands"),
    ("change_map", "Change map", "/map/nyandungu"),
    ("alerts", "Change alerts", "/alerts"),
    ("data", "Data", "/data"),
    ("model", "Model & method", "/model"),
    ("evaluation", "Evaluation", "/evaluation"),
    ("admin", "System status", "/admin"),
]

_cube_cache = {}


def load_all_metrics():
    p = OUTPUT_DIR / "metrics_all.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_metrics(site):
    m = load_all_metrics()
    if site in m:
        return m[site]
    p = site_paths(site)["metrics"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_changes(site):
    p = site_paths(site)["changes_geojson"]
    if not p.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(p.read_text(encoding="utf-8"))


def load_aoi(site):
    p = site_paths(site)["aoi"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_catalog(site):
    p = site_paths(site)["catalog"]
    if not p.exists():
        # legacy prototype catalog location for nyandungu
        legacy = ROOT / "data" / "processed" / "scene_catalog.csv"
        if site == "nyandungu" and legacy.exists():
            return pd.read_csv(legacy).to_dict("records")
        return []
    return pd.read_csv(p).to_dict("records")


def all_alerts():
    rows = []
    for site in WETLANDS:
        for f in load_changes(site)["features"]:
            r = dict(f["properties"])
            r["site_label"] = WETLANDS[site]["label"]
            rows.append(r)
    rows.sort(key=lambda r: -r["area_ha"])
    return rows


@app.context_processor
def inject():
    return {"nav": NAV, "wetlands": WETLANDS, "all_metrics": load_all_metrics()}


@app.route("/")
def dashboard():
    metrics = load_all_metrics()
    alerts = all_alerts()
    totals = {
        "sites": len(metrics),
        "area_inside": round(sum(m.get("area_hectares", {}).get("detected_inside_aoi", 0) for m in metrics.values()), 2),
        "polygons": sum(m.get("n_change_polygons", 0) for m in metrics.values()),
        "scenes": sum(m.get("scenes", 0) for m in metrics.values()),
    }
    return render_template("dashboard.html", active="dashboard", metrics=metrics,
                           totals=totals, top_alerts=alerts[:8])


@app.route("/wetlands")
def wetlands():
    return render_template("wetlands.html", active="wetlands", metrics=load_all_metrics())


@app.route("/wetland/<site>")
def wetland_detail(site):
    if site not in WETLANDS:
        abort(404)
    ts_path = OUTPUT_DIR / f"timeseries_{site}.json"
    ts = json.loads(ts_path.read_text()) if ts_path.exists() else None
    return render_template("wetland_detail.html", active="wetlands", site=site,
                           cfg=WETLANDS[site], metrics=load_metrics(site),
                           catalog=load_catalog(site), timeseries=ts,
                           changes=load_changes(site))


@app.route("/map/<site>")
def change_map(site):
    if site not in WETLANDS:
        abort(404)
    m = load_metrics(site)
    bounds = None
    if m:
        w, s, e, n = m["bounds_wgs84"]
        bounds = [[s, w], [n, e]]
    return render_template("map.html", active="change_map", site=site, cfg=WETLANDS[site],
                           metrics=m, aoi=load_aoi(site), changes=load_changes(site), bounds=bounds)


@app.route("/alerts")
def alerts():
    rows = all_alerts()
    site_f = request.args.get("site", "")
    type_f = request.args.get("type", "")
    conf_f = request.args.get("confidence", "")
    if site_f:
        rows = [r for r in rows if r["site"] == site_f]
    if type_f:
        rows = [r for r in rows if r["change_type"] == type_f]
    if conf_f:
        rows = [r for r in rows if r["confidence_level"] == conf_f]
    types = sorted({r["change_type"] for r in all_alerts()})
    return render_template("alerts.html", active="alerts", rows=rows, types=types,
                           site_f=site_f, type_f=type_f, conf_f=conf_f)


@app.route("/alerts.csv")
def alerts_csv():
    rows = all_alerts()
    buf = io.StringIO()
    fields = ["id", "site", "site_label", "change_type", "area_ha", "onset_date",
              "mean_magnitude", "max_run", "confidence", "confidence_level", "centroid_lon", "centroid_lat"]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="persistent_change_alerts.csv")


@app.route("/data")
def data():
    site = request.args.get("site", "nyandungu")
    if site not in WETLANDS:
        abort(404)
    return render_template("data.html", active="data", site=site, cfg=WETLANDS[site],
                           metrics=load_metrics(site), catalog=load_catalog(site))


@app.route("/model")
def model():
    site = request.args.get("site", "nyandungu")
    return render_template("model.html", active="model", site=site, cfg=WETLANDS[site],
                           metrics=load_metrics(site))


@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html", active="evaluation", metrics=load_all_metrics())


@app.route("/admin")
def admin():
    status = []
    for site, cfg in WETLANDS.items():
        p = site_paths(site)
        status.append({"site": site, "label": cfg["label"],
                       "cube": p["cube"].exists(), "aoi": p["aoi"].exists(),
                       "metrics": p["metrics"].exists() or (OUTPUT_DIR / "metrics_all.json").exists(),
                       "scenes": len(load_catalog(site))})
    return render_template("admin.html", active="admin", status=status)


@app.route("/report/<site>.pdf")
def report(site):
    if site not in WETLANDS:
        abort(404)
    from src.report import build_report
    pdf = build_report(site, load_metrics(site), load_changes(site))
    return send_file(io.BytesIO(pdf), mimetype="application/pdf",
                     as_attachment=True, download_name=f"{site}_persistent_change_report.pdf")


@app.route("/api/pixel/<site>")
def api_pixel(site):
    """Observed NDVI/MNDWI series for the pixel nearest to ?lat=&lon=."""
    if site not in WETLANDS:
        abort(404)
    try:
        lat = float(request.args["lat"]); lon = float(request.args["lon"])
    except Exception:
        return jsonify({"error": "lat and lon query parameters are required"}), 400
    if site not in _cube_cache:
        from src.analysis import load_cube
        _cube_cache[site] = load_cube(site)
    cube = _cube_cache[site]
    inv = ~cube["transform"]
    # Convert lon/lat (WGS84) to UTM, then to pixel coordinates
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32736", always_xy=True).transform(lon, lat)
    col_f, row_f = inv * (x, y)
    col, row = int(col_f), int(row_f)
    h, w = cube["ndvi"].shape[1:]
    if not (0 <= row < h and 0 <= col < w):
        return jsonify({"error": "point outside study window"}), 400
    def series(name):
        vals = cube[name][:, row, col]
        return [None if not np.isfinite(v) else round(float(v), 4) for v in vals]
    return jsonify({"site": site, "row": row, "col": col, "dates": list(cube["dates"]),
                    "ndvi": series("ndvi"), "mndwi": series("mndwi")})


@app.route("/figures/<site>/<path:name>")
def figures(site, name):
    return send_from_directory(site_paths(site)["figures"], name)


@app.route("/maps/<site>/<path:name>")
def maps(site, name):
    return send_from_directory(site_paths(site)["maps"], name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
