# Supervisor Briefing — Project Status, Data and Results

**Project:** Machine Learning-Based Monitoring and Detection of Persistent Changes in Wetlands: A Case Study of Kigali, Rwanda
**Student:** David Birenzi, BSc Software Engineering · **Supervisor:** Dirac Murairi
**Date of this briefing:** 22 October 2026 · **Status:** Working end-to-end system with evaluated results (not a mock-up)

---

## 1. What the project set out to do (proposal → what was approved)

The proposal's gap: existing wetland monitoring in Rwanda maps extent over defined periods but does **not distinguish persistent change from temporary variation**. This system learns each pixel's **normal seasonal behaviour**, tracks deviation from that normal, and reports a change only when it **persists across consecutive clear observations** — with onset date, type, area (ha) and confidence.

Proposal objectives, as delivered:

| Proposal objective | Delivered |
|---|---|
| 1. Review approaches; analyse Sentinel-2 patterns/indicators of persistent change | Literature-based design + 112-scene Sentinel-2 analysis, six spectral indices (below) |
| 2. Develop an ML system that learns normal patterns and detects/quantifies/maps persistent deviations | Two normal-behaviour models + persistence engine + typed change polygons/alerts + interactive maps |
| 3. Evaluate with technical and problem-specific metrics | Precision/recall/F1/accuracy/IoU on spatially held-out blocks, vs a two-date baseline, at three wetlands |

**Scope change to note (needs supervisor awareness, not a problem):** the proposal named *one* wetland. The system was built wetland-agnostic and validated on **three** contrasting wetlands, which strengthens Objective 3. The proposal's core question, data and method are unchanged.

## 2. Study sites

| Wetland | AOI & source | Area | Normal period → Monitoring | Story |
|---|---|---|---|---|
| Nyandungu | OSM eco-park polygon (Nominatim) | 127.5 ha | before Sep 2018 → Sep 2018–2026 | Restoration: works Sep 2018, park opened 8 Jul 2022 |
| Gikondo | OSM wetland way 1209698095 (Overpass) | 104.9 ha | 2017–2019 → 2020–2026 | Former industrial park; industries relocated from 2018; REMA rehabilitation Feb 2024 |
| Rwampara | Study delineation: 180 m buffer of OSM Rwampara stream way 1117393177 — **labelled as a delineation, not an official boundary** | 134.7 ha | 2017–2019 → 2020–2026 | Decades of urban/agricultural pressure; REMA rehabilitation started later in 2024 |

Land outside each AOI within the study window is mapped as a **control ring** for context (in urbanising Kigali it genuinely changes too, so formal accuracy is measured on labelled stable pixels — see §6).

## 3. Data — exactly what was used

- **Source:** Sentinel-2 L2A, Copernicus via the public Element84 Earth Search STAC catalogue (`sentinel-2-l2a`, tile area MGRS-35MRT). Nothing is scraped or hand-downloaded; every scene ID is recorded in `data/processed/<site>_catalog.csv`, so any result can be reproduced scene-by-scene.
- **Selection:** one best scene per quarter, 2017–2026: candidates under 35% catalogue cloud cover are tested in order until one has ≥55% clear pixels over the window (SCL classes 2, 4, 5, 6, 7 kept — vegetation, bare, water, unclassified, dark). **112 scenes used: Nyandungu 38 (7 normal + 31 monitoring), Gikondo 37 (11 + 26), Rwampara 37 (11 + 26).**
- **Bands/indices at 10 m, EPSG:32736:** B02, B03, B04, B08, B11, B12, SCL → **NDVI, NDWI (McFeeters), MNDWI, NDMI, SAVI, BSI**. Cubes are stored locally (`data/processed/<site>_sentinel2.npz`); the app and analysis run fully offline.
- **Radiometric validation (important, and a finding in itself):** STAC metadata advertises the −0.1 baseline-≥04.00 reflectance offset. We checked empirically (script `scripts/check_offset3.py`): over identical ground, median DN is ~850 (green) / ~2100 (NIR) for baseline 00.01–05.09 scenes alike, and MNDWI is stable (≈ −0.45) only under reflectance = DN × scale. Applying the advertised offset again drives green negative over vegetation and collapses MNDWI to −1 from 2022. The pipeline therefore uses DN × scale, documented in `docs/METHODS.md`.

## 4. Method

1. **Normal-behaviour models (two, compared head-to-head):**
   - *Seasonal harmonic:* per-pixel level + pooled seasonal shape (sin/cos, sin2/cos2 of day-of-year). Pooling matters: a purely per-pixel 5-parameter harmonic overfits Nyandungu's 7-scene normal period (we demonstrated this; it is in the methods note).
   - *MLP regressor* (scikit-learn): 64–32 ReLU, Adam, squared error, early stopping; inputs = seasonal harmonics + the pixel's normal-period median NDVI/NDWI/MNDWI; output = expected NDVI. Trained on 8,000 pixels sampled across each whole study window (54,932–85,553 training samples/site) — training only inside the wetland caused false alarms elsewhere, another checked-and-fixed design point.
2. **Persistence engine:** residual = observed − expected NDVI. A pixel is reported when |residual| ≥ threshold for **k consecutive** clear monitoring observations. Threshold and k are tuned on tuning blocks only. Per detected pixel: onset date (first scene of the first qualifying run), longest run, mean magnitude, 0–1 confidence.
3. **Change typing (rule-based, explainable):** monitoring-vs-normal shifts in NDVI/MNDWI/NDMI/BSI → vegetation gain/loss, water gain/loss, vegetation & water gain/loss, built-up/bare encroachment, mixed. Connected pixels ≥9 px (0.09 ha) become polygons (4-connected labelling matched to polygonisation — an earlier 8-connected version triple-counted areas; caught because PostGIS area sums were reconciled against raster areas).
4. **Baseline (proposal-required comparator):** two-date |NDVI difference| between the clearest normal-period scene and the clearest of the latest monitoring scenes (2018-08-16 vs 2025-07-30 at all sites), threshold tuned identically. No seasonal model, no persistence.

## 5. Evaluation design (how the numbers were earned)

- **Reference ("silver standard"):** a pixel is a reference positive when its monitoring median NDVI shifts ≥0.08 from its normal median on ≥60% of clear monitoring observations (either direction); negatives are clearly stable pixels. It is built from raw medians, independent of both ML models — but it is still index-derived, so we call it *silver*, not gold, everywhere including the app.
- **Spatial hold-out:** the map is split into 150 m blocks; whole blocks go to tuning or test (35% test). Random pixel splits would leak neighbours and inflate scores.
- **Metrics:** precision, recall, F1, accuracy, IoU + full confusion matrices — exactly the proposal's metric set.

## 6. Results

### 6.1 Model comparison (held-out test blocks)

| Wetland | Model | Precision | Recall | **F1** | Accuracy | IoU | thr / k |
|---|---|---|---|---|---|---|---|
| Nyandungu | **MLP + persistence ★** | 0.558 | 0.824 | **0.666** | 0.956 | 0.499 | 0.12 / 4 |
| | Harmonic + persistence | 0.549 | 0.839 | 0.664 | 0.955 | 0.496 | 0.10 / 4 |
| | Two-date baseline | 0.459 | 0.833 | 0.592 | 0.939 | 0.420 | 0.18 / – |
| Gikondo | **Harmonic + persistence ★** | 0.801 | 0.967 | **0.876** | 0.979 | 0.779 | 0.12 / 4 |
| | MLP + persistence | 0.575 | 0.929 | 0.710 | 0.943 | 0.551 | 0.18 / 3 |
| | Two-date baseline | 0.596 | 0.654 | 0.623 | 0.941 | 0.453 | 0.18 / – |
| Rwampara | **Harmonic + persistence ★** | 0.454 | 0.877 | **0.598** | 0.974 | 0.427 | 0.10 / 4 |
| | MLP + persistence | 0.396 | 0.708 | 0.508 | 0.970 | 0.341 | 0.12 / 4 |
| | Two-date baseline | 0.144 | 0.541 | 0.228 | 0.919 | 0.129 | 0.12 / – |

Confusion matrices (TN/FP/FN/TP): Nyandungu chosen 21639/817/221/1033 (support 23,710, positives 1,254); Gikondo chosen 31686/632/86/2538 (34,942; 2,624); Rwampara chosen 35303/861/100/715 (36,979; 815).

**Reading:** the proposed approach beats the baseline at all three sites — decisively at Rwampara (F1 0.60 vs 0.23), where one scene pair cannot separate lasting encroachment from seasonality. Accuracy is high everywhere partly because most pixels are stable; F1/IoU are the honest headline (this is stated in the app).

### 6.2 What was detected (inside each wetland)

| Wetland | Detected | Alerts (polygons) | Dominant types (ha) | Onset pattern |
|---|---|---|---|---|
| Nyandungu (127.5 ha) | **51.5 ha** | 38 | Vegetation gain 28.8, vegetation & water recovery 7.4, mixed 9.4, vegetation loss 2.6 | Peaks in works years 2019–2020 (1,237 px) and a second greening wave 2024 (2,538 px) — vegetation maturation after opening |
| Gikondo (104.9 ha) | **81.7 ha** | 12 | Vegetation gain 42.5, water loss 18.4, mixed 11.3, recovery 6.1 | Concentrated 2021–2022 (4,676 px) — regrowth after industrial relocation, before formal rehabilitation |
| Rwampara (134.7 ha) | **56.6 ha** | 62 | Mixed 21.9, built-up/bare encroachment 18.2, vegetation loss 10.4, vegetation gain 4.2 | Spread 2020–2025 with the largest count in 2025 (1,316 px) — continuing pressure + rehabilitation-era disturbance |

Total: **189.7 ha** persistent change inside the three wetlands; 112 alert polygons (their areas sum to 178.4 ha; the small difference is sub-0.09 ha fragments below the mapping unit). For context, detections in the whole study windows are 928.6 / 622.7 / 620.0 ha — Kigali's surroundings are changing too; the product is the inside-wetland alerts.

**Independent plausibility checks:** Nyandungu's median NDVI (inside) climbs through the monitoring period while its control ring stays flat; Gikondo's dominant signal flips to vegetation gain only after the documented industrial relocation; Rwampara's signature (built-up/bare + vegetation loss) matches the published degradation studies of that wetland (reported loss of wetland area to urban growth). The system was not told any of this — it emerges from the residuals.

## 7. System as implemented

- **Stack (as proposed, with documented substitutions):** Python, scikit-learn, Rasterio/Shapely, Flask + Leaflet + Chart.js, PostgreSQL/PostGIS for deployment, Docker Compose. Proposal said Google Earth Engine; STAC was used to build the identical cube layout because it needs no account and yields an offline, inspectable artefact — the substitution and reason are recorded in `docs/METHODS.md`/`docs/DATA.md`.
- **App (http://127.0.0.1:5000, running locally):** Dashboard, Wetlands, per-wetland Explorer (interactive NDVI/MNDWI series, wetland vs control ring), Change Map (typed polygons; **click any pixel to load its own history**), Change Alerts register (filters + CSV export), Data (index definitions + full scene catalogue), Model & Method, Evaluation, System Status. Per-wetland **PDF reports** generated on demand.
- **Database:** PostGIS schema (`db/schema.sql`) + loader, **tested live**: alert areas queried from the database reconcile with raster hectares exactly (Gikondo 80.1 ha, Nyandungu 47.7 ha, Rwampara 50.6 ha polygon sums).
- **Verification:** 8 automated tests pass; all 19 app routes return HTTP 200; Docker web image builds and the container serves the app; the Jupyter notebook (`notebooks/kigali_wetland_persistent_change.ipynb`) has been executed end-to-end and reproduces all tables.
- **Reproducibility:** `python scripts/make_aois.py` → `python -m src.download --all` → `python -m src.analysis --all` → `python -m app.app`.

## 8. Limitations — stated, not hidden

1. **Reference is silver-standard** (index-derived). Absolute accuracy against true land-cover change requires field verification or digitised high-resolution reference. A field template (`data/field/field_verification_template.csv`) and the alerts CSV are ready for this; the proposal budgeted field visits.
2. **Rwampara's AOI is a delineation** (stream buffer), so its areas are study-area statistics; Gikondo/Nyandungu use OSM polygons, not official REMA cadastral boundaries.
3. **Onset precision is quarterly** (one scene per quarter); denser compositing would sharpen onset dates.
4. **Precision is the weak metric at Nyandungu/Rwampara:** some flagged pixels are real lasting change that falls outside the reference rule (e.g. slow change below the 0.08 shift) — the maps should be read beside the scores.
5. Sentinel-2 limits: persistent cloud (one Nyandungu scene, 2025-12-27, is only 58% clear and shows a one-scene NDVI dip — persistence correctly suppresses it as a false change) and changes under ~0.09 ha are not mapped. Sentinel-1 SAR fusion is future work.

## 9. What we ask of the supervisor at this meeting

1. **Approve the three-wetland validation** (expansion of the proposal's single-wetland scope) — or advise if the write-up should foreground Nyandungu and present the others as generalisation tests.
2. **Advise on the reference standard:** accept the silver-standard evaluation plus planned field verification of the largest alerts, or require digitised reference polygons before final submission.
3. **Confirm the write-up framing:** model choice is per-site best-of-two (MLP at Nyandungu, harmonic at Gikondo/Rwampara) selected on tuning blocks — we present both models everywhere for transparency.

## 10. Immediate next steps (student)

1. Field verification of top alerts (GPS + photos) → gold-standard spot-check of precision.
2. Write Chapters 4 (implementation/results) and 5 (discussion/conclusions) from the notebook + `docs/METHODS.md`.
3. Record the 6–8 minute demo (`docs/DEMO_SCRIPT.md`), commit and push to GitHub (changes are currently uncommitted for supervisor review first).

*Key files for the supervisor to inspect: `README.md` (overview + headline results), `docs/METHODS.md` (decisions & limitations), `docs/SUPERVISOR_BRIEFING.md` (this file), `outputs/metrics_all.json` (all numbers), the executed notebook, and the live app.*
