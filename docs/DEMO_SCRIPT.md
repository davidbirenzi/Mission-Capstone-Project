# Demo script (6–8 minutes, single take)

Start the app first (`python -m app.app`, open http://127.0.0.1:5000). Everything runs from stored data — no internet needed except map tiles; if tiles fail, the figures on the Wetlands pages carry the same story.

1. **Dashboard (45 s).** "Most wetland change detection compares two satellite dates and fires on seasonal change. This system learns each wetland's normal seasonal behaviour and reports only changes that persist across successive observations — each with an onset date, type, hectares and confidence." Point at the three-wetland table and the model-vs-baseline F1 column.
2. **Wetlands → Nyandungu (1 min).** Before/recent imagery; NDVI inside the park rises ~0.65 → ~0.83 while the control ring stays at ~0.45. Onsets concentrate in 2018–2020, the restoration-works period. That alignment with known history is the first validation.
3. **Gikondo / Rwampara (1 min).** The degraded cases: water loss dominates, built-up/bare encroachment appears at Rwampara, onset years 2020–2022 precede the 2024 REMA rehabilitation — the system detects the degradation era, not just the headline site.
4. **Change map (1.5 min).** Pan the typed polygons, click one: area, onset, magnitude, confidence. Then click any pixel: its own NDVI/MNDWI history loads. "An officer can go from a city-wide alert to a single pixel's evidence in two clicks."
5. **Change alerts (45 s).** Filter by wetland/confidence; download CSV — this is the field-verification worklist. Mention the field template that turns these into gold-standard labels.
6. **Evaluation (1 min).** Per wetland: harmonic + MLP vs two-date baseline on spatially held-out blocks. Be direct about precision limits and the silver-standard reference (Methods note) — and about the radiometric-offset verification, which shows the data pipeline was validated, not assumed.
7. **Notebook + close (45 s).** Scroll the notebook: it retrains everything and reproduces the tables. Close with scale-up: adding a wetland is a boundary file plus one registry entry; PostGIS/Docker deployment is included.
