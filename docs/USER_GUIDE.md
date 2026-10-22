# User guide

## Who this is for
- **Wetland / environment officer** — use Dashboard, Change map and Change alerts to find, rank and verify persistent changes.
- **Researcher** — use Data, Model & method, Evaluation and the notebook to reproduce or extend the analysis.
- **System administrator** — use System status to check pipeline readiness and add wetlands.

## Everyday workflow for an officer
1. Open **Dashboard** for the headline picture across wetlands.
2. Open **Change alerts**, filter to one wetland and High confidence, sort by area (default).
3. Click **Map** on an alert to see it in context; click the polygon for onset/magnitude; click nearby pixels to inspect their histories.
4. Download the **CSV** (or the wetland's **PDF report**) as the field worklist.
5. On site, complete `data/field/field_verification_template.csv` (GPS, photo, observed cover, confirmed?). Confirmed points upgrade the evaluation from silver- to gold-standard.

## Reading the outputs
- **Onset date** = first scene of the persistent run (quarter precision).
- **Confidence** combines run length, share of anomalous observations after onset, and deviation magnitude. High ≥ 0.66.
- **Change types** are rule-based from NDVI/MNDWI/NDMI/BSI shifts — treat "Mixed/other" as "real persistent deviation, type unclear; verify first".
- Areas are study-area statistics from the AOI in use (see boundary source on each Wetland page), not cadastral figures.

## Adding a wetland (admin)
1. Put its boundary GeoJSON in `data/aoi/` (official polygon preferred; a documented delineation is acceptable if labelled as such).
2. Add one entry to `WETLANDS` in `src/config.py`: file, study window, normal-period end, story.
3. Run `python -m src.download --site <name>` then `python -m src.analysis --site <name>`, restart the app. It appears everywhere automatically.
