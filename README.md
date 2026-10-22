# Kigali Wetland Persistent-Change Monitor

A machine-learning system that detects **persistent** changes in Kigali's wetlands from multi-temporal Sentinel-2 imagery — changes that continue across successive observations, as distinct from temporary seasonal or water-level variation.

**Capstone:** *Machine Learning-Based Monitoring and Detection of Persistent Changes in Wetlands: A Case Study of Kigali, Rwanda* — David Birenzi, BSc Software Engineering (Supervisor: Dirac Murairi).

## The idea in one paragraph

Most change detection compares two dates and fires on seasonality, a dry month, or one cloudy scene. This system learns **what each pixel normally does across the year** (two models: per-pixel harmonic regression, and an MLP regressor on seasonal harmonics + the pixel's normal NDVI/NDWI/MNDWI), tracks the deviation from that normal, and reports a change only when the deviation **persists for consecutive clear observations**. Every reported change has an **onset date, type, area in hectares, magnitude and confidence**, and becomes a polygon in an alerts register a wetland officer can verify.

## Study wetlands (contrasting validation cases)

| Wetland | Story | Why it validates the system |
|---|---|---|
| Nyandungu | Restoration: works from Sep 2018, eco-park opened 8 Jul 2022 | Should show persistent vegetation gain with onsets in the works/opening period |
| Gikondo | Former industrial park (industries relocated from 2018); REMA rehabilitation from Feb 2024 | Long degradation/disturbance signal, then rehabilitation-era change |
| Rwampara | Urban growth, agriculture and waste pressure for decades; REMA rehabilitation started later in 2024 | Persistent vegetation/water loss from encroachment, then works disturbance |

Land outside each boundary inside the study window is a **control ring**: persistent change should concentrate inside the wetland, not in the ring. Rwampara's AOI is a documented study delineation (180 m buffer of the OSM Rwampara stream); the others are OSM wetland/park polygons. See `docs/METHODS.md`.

## Headline results

Produced by `python -m src.analysis --all` (112 Sentinel-2 scenes) and shown live in the app (Dashboard / Evaluation). Scores are on spatially held-out blocks against the persistent-shift reference (`docs/METHODS.md` explains this "silver standard" and its limits).

| Wetland | Best proposed model F1 / IoU | Other ML model F1 | Two-date baseline F1 / IoU | Persistent change inside wetland |
|---|---|---|---|---|
| Nyandungu (127.5 ha) | **MLP 0.666 / 0.499** (P 0.56, R 0.82) | Harmonic 0.664 | 0.592 / 0.420 | 51.5 ha — vegetation gain (28.8 ha) + vegetation & water recovery (7.4 ha); onsets in the restoration-works years and a second greening wave from 2024 |
| Gikondo (104.9 ha) | **Harmonic 0.876 / 0.779** (P 0.80, R 0.97) | MLP 0.710 | 0.623 / 0.453 | 81.7 ha — vegetation gain (42.5 ha) after industrial relocation, with water loss (18.4 ha); onsets 2020–2023 |
| Rwampara (134.7 ha study reach) | **Harmonic 0.598 / 0.427** (P 0.45, R 0.88) | MLP 0.508 | 0.228 / 0.129 | 56.6 ha — built-up/bare encroachment (18.2 ha) and vegetation loss (10.4 ha), the degradation signature the literature reports |

112 change polygons (alerts, ≥0.09 ha) are mapped across the three wetlands, each with type, onset date, magnitude and confidence; their areas sum to the detected hectares above (minus sub-mapping-unit fragments). The proposed approach beats the two-date baseline at every site — most decisively at Rwampara, where the baseline collapses (F1 0.23) because a single scene pair cannot separate lasting encroachment from seasonal states. Each wetland's dominant change type matches its documented history, which is an independent plausibility check on the method. Two engineering findings worth noting: Sentinel-2 scenes in this catalogue mix processing baselines, and their STAC metadata advertises a −0.1 radiometric offset that the COG digital numbers have *already* had applied — we verified empirically (DN distributions over identical ground across baselines; MNDWI stability) that reflectance = DN × scale only, and that applying the offset again corrupts every index (see `docs/METHODS.md`). And a purely per-pixel harmonic model overfits short normal periods, so the seasonal shape is pooled.

## Quick start (local, demo-safe offline)

Use Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python -m src.analysis --all  # if outputs/ are already present you can skip straight to the app
python -m app.app             # http://127.0.0.1:5000
```

To rebuild the satellite cubes (needs internet; resumable per quarter):

```bash
python scripts/make_aois.py   # wetland boundaries from OpenStreetMap
python -m src.download --all  # Sentinel-2 cubes via STAC (Copernicus / Earth Search)
python -m src.analysis --all  # models, persistence, evaluation, maps, metrics, alerts
python -m pytest tests/ -q    # tests
```

Notebook: open `notebooks/kigali_wetland_persistent_change.ipynb` and run all cells — it retrains everything and presents the results with discussion prompts.

Docker (app + PostgreSQL/PostGIS): `docker compose up --build`, then optionally load results into PostGIS with `python -m src.load_postgis`. The demo itself never depends on the database or on a live satellite service — that is deliberate, so it runs anywhere.

## What the app shows

| Page | What it shows |
|---|---|
| Dashboard | Headline results across all wetlands, largest alerts, method summary |
| Wetlands | The three study wetlands, their stories, boundaries and key figures |
| Wetland explorer | Before/recent imagery, interactive NDVI/MNDWI time series (wetland vs control ring), change-by-type, onset years |
| Change map | Leaflet map: change polygons by type with popups (area, onset, magnitude, confidence); **click anywhere to inspect that pixel's own history** |
| Change alerts | Filterable register of all change polygons; CSV export for field verification |
| Data | Index definitions (NDVI, NDWI, MNDWI, NDMI, SAVI, BSI) and the full scene catalogue with scene IDs |
| Model & method | The two normal-behaviour models, persistence rule, typing rules, baseline |
| Evaluation | Per-wetland precision/recall/F1/accuracy/IoU for both models vs baseline, confusion matrices, control-ring check |
| System status | Pipeline status, how to add a new wetland, field-verification workflow |
| PDF report | One-click per-wetland report (Dashboard/Wetlands pages) |

## Project layout

```text
src/config.py       Wetland registry, periods, model/persistence settings
src/download.py     Sentinel-2 STAC pipeline -> offline index cubes
src/analysis.py     Harmonic + MLP normal models, persistence engine, typing,
                    spatial-block evaluation, polygons, figures, metrics
src/report.py       Per-wetland PDF reports
src/load_postgis.py Load results into PostgreSQL/PostGIS (deployment layer)
app/                Flask web application (dashboard, maps, alerts, reports)
db/schema.sql       PostGIS schema (wetlands, scenes, runs, alerts, field checks)
data/aoi/           Wetland boundaries (OSM)
data/processed/     Offline Sentinel-2 cubes + scene catalogues
data/field/         Field-verification template (gold-standard reference)
notebooks/          Analysis notebook (retrains + presents results)
outputs/            Metrics, figures, map overlays, change polygons (GeoJSON)
docs/               Methods/data notes, screenshots
tests/              Core engine tests
```

## Data & reproducibility

Sentinel-2 L2A from the public Element84 Earth Search STAC catalogue; one best scene per quarter 2017–2026 (cloud/shadow masked, ≥55% clear). All scene IDs are in `data/processed/*_catalog.csv`. Design decisions, alternatives and **known limitations** (including the silver-standard reference and how field verification replaces it) are documented honestly in `docs/METHODS.md` — read that before the defence.

## Deployment plan

Phase 1 (this delivery) runs locally and offline from stored cubes. The same app ships in Docker with PostgreSQL/PostGIS for a small cloud VM in phase 2; scheduled cube refreshes, user accounts and ingestion for additional wetlands are natural next steps because the system is site-agnostic (add a boundary + one registry entry).

## Author

David Birenzi — Mission Capstone Project.
