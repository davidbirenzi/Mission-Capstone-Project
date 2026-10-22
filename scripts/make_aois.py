"""Build AOI GeoJSON files for the study wetlands.

Sources (accessed Oct 2026):
- Gikondo: OpenStreetMap way 1209698095 (natural=wetland), fetched via Overpass.
- Rwampara: OpenStreetMap way 1117393177 (Rwampara stream, waterway), fetched via
  Overpass. The study AOI is delineated as a 180 m buffer around the stream reach
  inside the study window -- a documented study-area delineation, not an official
  boundary. This is stated wherever the AOI is used.
- Nyandungu: existing data/aoi/nyandungu.geojson (OSM relation via Nominatim).
"""
import json, urllib.request, urllib.parse
from pathlib import Path
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
AOI = ROOT / "data" / "aoi"
AOI.mkdir(parents=True, exist_ok=True)

def overpass_geom(way_id):
    q = f"[out:json][timeout:90]; way({way_id}); out geom;"
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": q}).encode(),
        headers={"User-Agent": "MissionCapstone/1.0 (student research)"},
    )
    d = json.load(urllib.request.urlopen(req, timeout=120))
    pts = [(p["lon"], p["lat"]) for p in d["elements"][0]["geometry"]]
    return pts

def save(name, geom, props):
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": props, "geometry": mapping(geom)}]}
    p = AOI / name
    p.write_text(json.dumps(fc), encoding="utf-8")
    print("wrote", p)

# --- Gikondo wetland polygon ---
pts = overpass_geom(1209698095)
poly = Polygon(pts)
assert poly.is_valid and len(pts) > 10
save("gikondo.geojson", poly, {
    "name": "Gikondo Wetland", "source": "OpenStreetMap way 1209698095 (natural=wetland) via Overpass",
    "osm_id": 1209698095, "story": "degradation (former industrial park) and rehabilitation from 2024",
})

# --- Rwampara stream-buffer AOI ---
pts = overpass_geom(1117393177)
line = LineString(pts)
print("rwampara stream pts", len(pts), "bounds", line.bounds)
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32736", always_xy=True).transform
to_wgs = Transformer.from_crs("EPSG:32736", "EPSG:4326", always_xy=True).transform
buf = shp_transform(to_utm, line).buffer(180)
aoi_wgs = shp_transform(to_wgs, buf)
save("rwampara.geojson", aoi_wgs, {
    "name": "Rwampara Wetland (study reach)", "source": "Delineated as 180 m buffer of OSM Rwampara stream way 1117393177 via Overpass; study delineation, not an official boundary",
    "osm_id": 1117393177, "story": "degradation from urban growth, agriculture and waste; rehabilitation started 2024 (late)",
})
print("done")
