# Data

## Sources
- **Sentinel-2 L2A** via Element84 Earth Search STAC (`sentinel-2-l2a`),
  tile area MGRS-35MRT. One best scene per quarter, 2017–2026: lowest cloud
  candidates tested until a scene with ≥55% clear pixels (SCL classes
  2,4,5,6,7) over the study window is found. Scene IDs are kept in
  `data/processed/<site>_catalog.csv` — the analysis is reproducible from them.
- **Boundaries**: Nyandungu — OSM park polygon; Gikondo — OSM wetland way
  1209698095; Rwampara — documented 180 m buffer delineation of the OSM
  Rwampara stream (see `docs/METHODS.md`). Rebuild with `scripts/make_aois.py`.

## Cube layout (`data/processed/<site>_sentinel2.npz`)
Per scene (10 m, EPSG:32736): `ndvi, ndwi, mndwi, ndmi, savi, bsi`
(float16, NaN where not clear), `clear` mask, true-colour preview for the
earliest/latest clear scenes, plus `dates`, `transform`, `bounds_wgs84`.

Bands used: B02, B03, B04, B08, B11, B12, SCL. Index formulas are listed in
the app's Data page and `src/download.py:compute_indices`.

## Alternative acquisition (hybrid design)
Google Earth Engine can export the same quarterly index stacks
(collection `COPERNICUS/S2_SR_HARMONIZED`, SCL-equivalent QA masking) into
the same NPZ layout; the analysis and app are source-agnostic because they
only read the cube. STAC was used for the delivered cubes because it needs
no account and produces a fully offline, inspectable artefact.

## Field verification
`data/field/field_verification_template.csv` — complete on site visits for
the largest alerts (Dashboard → Change alerts). Confirmed/refuted points
supersede the silver-standard reference for evaluation.
