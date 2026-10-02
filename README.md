# Nyandungu persistent-change monitor

A phase-1 machine learning prototype that detects persistent change in the Nyandungu Urban Wetland Eco-Tourism Park, Kigali, from multi-temporal Sentinel-2 imagery.

The seasonal model learns how the wetland looked before restoration works began in September 2018. After the eco-park opened on 8 July 2022, a pixel is reported only when its departure from that normal pattern continues across successive clear observations. A two-date NDVI difference is the baseline.

GitHub repository: https://github.com/davidbirenzi/Mission-Capstone-Project

## What the prototype does

- Builds a local Sentinel-2 time series for Nyandungu (NDVI, NDWI, cloud mask).
- Trains a scikit-learn multilayer perceptron: inputs are seasonal harmonics plus each pixel’s pre-restoration NDVI and NDWI; hidden layers are 64 and 32 units with ReLU; the optimizer is Adam; the loss is squared error.
- Flags a persistent change when the residual stays large for successive post-opening dates.
- Maps the affected area and scores precision, recall, F1, accuracy, and IoU on held-out map blocks.

On the current cube the seasonal model, scored on held-out blocks, reaches precision 0.233, recall 0.899, F1 0.370, accuracy 0.937, and IoU 0.227. The two-date baseline reaches precision 0.127, recall 0.617, F1 0.211, accuracy 0.905, and IoU 0.118. The mapped persistent greening inside the park is about 62 hectares of the 128 hectare park footprint on this grid. Precision is the weaker number because some lasting greening also sits outside the park boundary, and those pixels are not treated as reference positives.

## Setup

Use Python 3.11 or newer. The project was prepared with Python 3.14.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The satellite cube is already in `data/processed/` after the first download, so the notebook and the web app run offline. To rebuild the cube (this step needs the internet):

```bash
python -m src.download
```

Train the model and write the figures, metrics, and map overlay:

```bash
python -m src.analysis
```

Open `notebooks/nyandungu_persistent_change.ipynb` and run all cells. That retrains the model and refreshes the figures.

Start the web MVP:

```bash
python -m app.app
```

Then open http://127.0.0.1:5000 . The pages are Overview, Study area, Data, Model, Change map, and Performance.

## Designs

Screenshots of the web interface are in [docs/screenshots](docs/screenshots).

| Screen | What it shows |
| --- | --- |
| Overview | Requirement-to-tool mapping and the headline result |
| Study area | Nyandungu before the works and after opening |
| Data | NDVI, NDWI, correlation, and the scene list |
| Model | Network diagram, loss, persistence rule, residual series |
| Change map | Leaflet map of persistent change and the area in hectares |
| Performance | Held-out precision, recall, F1, accuracy, and IoU against the baseline |

## Deployment plan

Phase 1 runs on a local machine. Flask serves the interface and reads the metrics and map overlay produced by the notebook. Nothing in the demo depends on a live Earth Engine session.

A later deployment can put the same app in a container on a small cloud virtual machine, with the processed cube stored beside it. PostgreSQL/PostGIS, user accounts, and a live satellite pull belong to the next phase, once this Nyandungu result is stable.

## Video demo

Record a single take of 6 to 8 minutes. Spend the time on the working prototype.

1. Start the app and walk the sidebar once, so the navigation is visible (about 30 seconds).
2. Study area: show the before and after images and say why Nyandungu is the test (about 1 minute).
3. Data: show the NDVI shift and the time series staying up after opening (about 1 minute).
4. Model: show the layer diagram, Adam and ReLU, and the residual series (about 1.5 minutes).
5. Change map: pan the terracotta pixels against the park boundary and read the hectare figure (about 1.5 minutes).
6. Performance: read the held-out scores next to the two-date baseline (about 1 minute).
7. Open the notebook and scroll the architecture cell, the metrics table, and the change map (about 1 minute).

## Project layout

```text
app/                  Flask MVP
data/aoi/             Nyandungu park boundary from OpenStreetMap
data/processed/       Offline Sentinel-2 cube and scene catalog
notebooks/            Model notebook
src/                  Download, seasonal MLP, persistence rule, metrics
outputs/              Figures, metrics, and the map overlay
```

## Data

Sentinel-2 L2A scenes are from the public Element84 Earth Search catalog, tile MGRS-35MRT. One clear scene is kept per quarter from 2017 through mid-2026 when cloud, shadow, and cirrus leave at least 55 percent of the study window usable. The scene list, including scene ids, is `data/processed/scene_catalog.csv`.
