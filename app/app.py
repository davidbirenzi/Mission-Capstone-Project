"""Local web MVP for the Nyandungu persistent-change monitor."""

import json

import pandas as pd
from flask import Flask, render_template, send_from_directory

from src.config import AOI_PATH, CATALOG_PATH, FIGURE_DIR, MAP_DIR, METRICS_PATH

app = Flask(__name__)

NAV = [
    ("overview", "Overview", "overview"),
    ("study_area", "Study area", "study_area"),
    ("data", "Data", "data"),
    ("model", "Model", "model"),
    ("change_map", "Change map", "change_map"),
    ("performance", "Performance", "performance"),
]


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_catalog():
    if not CATALOG_PATH.exists():
        return []
    frame = pd.read_csv(CATALOG_PATH)
    return frame.to_dict(orient="records")


def load_aoi():
    return json.loads(AOI_PATH.read_text(encoding="utf-8"))


@app.context_processor
def inject_nav():
    return {"nav": NAV, "metrics": load_metrics()}


@app.route("/")
def overview():
    return render_template("overview.html", active="overview")


@app.route("/study-area")
def study_area():
    return render_template("study_area.html", active="study_area")


@app.route("/data")
def data():
    return render_template("data.html", active="data", catalog=load_catalog())


@app.route("/model")
def model():
    return render_template("model.html", active="model")


@app.route("/map")
def change_map():
    metrics = load_metrics()
    bounds = None
    if metrics:
        west, south, east, north = metrics["bounds_wgs84"]
        bounds = [[south, west], [north, east]]
    return render_template(
        "map.html",
        active="change_map",
        aoi=load_aoi(),
        bounds=bounds,
    )


@app.route("/performance")
def performance():
    return render_template("performance.html", active="performance")


@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(FIGURE_DIR, name)


@app.route("/maps/<path:name>")
def maps(name):
    return send_from_directory(MAP_DIR, name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
