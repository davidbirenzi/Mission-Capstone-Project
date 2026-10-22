# Methods note — design decisions, rationale and limitations

This note records *why* the system is built the way it is. It is the source
text for Chapter 3/4 justifications in the dissertation.

## The core claim
Conventional change detection compares two dates. Wetlands vary naturally
(season, water level), so two-date differences fire on temporary states.
This system instead (1) learns each pixel's **normal seasonal behaviour**,
(2) tracks the **residual** (observed − expected NDVI), and (3) reports a
change only when the residual stays large for **k consecutive clear
observations**. Persistence is the contribution; everything else supports it.

## Design decisions and alternatives considered
| Decision | Chosen | Alternative(s) | Reason |
|---|---|---|---|
| Data source | Hybrid: STAC (Element84 Earth Search, Sentinel-2 L2A) builds an offline cube; app never calls a live service | Google Earth Engine only | Demo reliability (no account/internet dependency at presentation), full reproducibility. GEE remains a valid alternative acquisition route to the same cube layout. |
| Normal model | Two models compared: per-pixel harmonic regression AND an MLP | Deep learning (LSTM/UNet) | Almost no labelled data; cloudy quarterly series are short. Harmonic is transparent; MLP captures non-linear level/season interactions. Comparing them is itself a result. |
| Detection unit | Pixel residual + persistence, then connected-component polygons (≥9 px) | Object-based segmentation | Simpler, reproducible, and the polygon step gives the area/alert outputs managers need. |
| Change typing | Rule-based from NDVI/MNDWI/NDMI/BSI monitoring-vs-normal shifts | Trained classifier | No labelled change-type samples exist yet; rules are explainable per alert. Field labels can later train a Random Forest. |
| Reference for evaluation | "Silver standard": persistent multi-observation NDVI shift (≥0.08 on ≥60% of clear monitoring obs) defines positives; clearly stable pixels define negatives | Manual digitising only; random pixel split | Independent of both ML models (it uses raw medians, not residuals), but still index-derived — therefore called *silver*, not gold. Gold standard = field verification (template in `data/field/`) and/or digitised high-resolution interpretation. This limitation must be stated in the report. |
| Evaluation split | Spatial blocks (150 m), whole blocks held out | Random pixels | Neighbouring pixels are near-duplicates; random splits inflate scores. Block hold-out is the honest estimate. |
| Control / context | Land outside the AOI inside each study window ("control ring") is mapped for context; formal accuracy uses labelled stable pixels in held-out spatial blocks | A separate "stable wetland" | A truly unchanged Kigali wetland could not be verified from available data, and in a fast-growing city the ring genuinely changes too — so ring area is reported as context, never as the accuracy measure. |
| Database | PostgreSQL/PostGIS for deployment; file store (JSON/GeoJSON/NPZ) for the local demo | PostGIS-only | Keeps the evaluated demo runnable anywhere offline; `src/load_postgis.py` loads identical results into PostGIS. |

## Radiometric harmonisation (checked empirically — do not skip this)
Sentinel-2 processing baseline ≥04.00 added a −0.1 reflectance offset to the
original Copernicus products, and the STAC `raster:bands` metadata of Earth
Search scenes still advertises that offset. It would be natural to apply it —
and wrong for this catalogue. We verified directly
(`scripts/check_offset3.py`): over identical ground, median DN is ~850
(green) / ~2100 (NIR) for baseline 00.01, 04.00, 05.00 and 05.09 scenes
alike, i.e. the COG DNs are already harmonised; and MNDWI stays temporally
stable at ≈ −0.45 only under reflectance = DN × scale, whereas applying the
metadata offset drives green negative over vegetation (clipped to 0) and
collapses MNDWI to −1 from 2022 onward. The pipeline therefore uses DN ×
scale only (`src/download.py:_asset_scale_offset`). Mention this in the
defence: the radiometry was validated against the data, not assumed from
metadata — in either direction.

## Known limitations (state these in the defence — they are strengths if owned)
1. The reference labels are index-derived (silver standard). Absolute
   accuracy against true land-cover change awaits field verification.
2. Precision is penalised by real persistent change that falls outside the
   reference definition (e.g. lasting greening outside a park boundary, or
   change slower than the 0.08 shift rule). Read maps + control-ring areas
   next to the scores.
3. Quarterly best-scene compositing limits onset dating to the quarter of
   the first anomalous scene; denser compositing would sharpen onsets.
4. Rwampara's AOI is a documented study delineation (180 m stream buffer),
   not an official boundary; Gikondo's is the OSM wetland polygon; areas are
   therefore study-area statistics, not cadastral figures.
5. Sentinel-2 cannot see change under persistent cloud or very small
   (< ~0.1 ha) encroachments; Sentinel-1 SAR fusion is future work.

## Ethics / data
All imagery is open Copernicus data; boundaries are OpenStreetMap
(ODbL). No personal data is processed. Field photos should avoid faces and
private property detail.
