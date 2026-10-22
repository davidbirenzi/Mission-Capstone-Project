"""Load file-based results into PostgreSQL/PostGIS (deployment layer).

Requires a running database (see docker-compose.yml):
    DATABASE_URL=postgresql://wetland:wetland@localhost:5433/wetland python -m src.load_postgis

The local demonstration does not depend on this; it keeps the system fully
usable offline, which is a deliberate design decision documented in the report.
"""
import json
import os

from src.config import OUTPUT_DIR, WETLANDS, site_paths


def main():
    import psycopg2
    url = os.environ.get("DATABASE_URL", "postgresql://wetland:wetland@localhost:5433/wetland")
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    schema = (OUTPUT_DIR.parent / "db" / "schema.sql").read_text()
    cur.execute(schema)

    metrics_all = {}
    p = OUTPUT_DIR / "metrics_all.json"
    if p.exists():
        metrics_all = json.loads(p.read_text())

    for site, cfg in WETLANDS.items():
        # Idempotent reload: clear this site's previous runs/alerts/scenes.
        cur.execute("DELETE FROM change_alert WHERE site=%s", (site,))
        cur.execute("DELETE FROM model_run WHERE site=%s", (site,))
        cur.execute("DELETE FROM scene_catalog WHERE site=%s", (site,))
        aoi = json.loads(site_paths(site)["aoi"].read_text())
        geom = json.dumps(aoi["features"][0]["geometry"])
        cur.execute(
            "INSERT INTO wetland(site,label,subtitle,story,aoi_source,intervention,geom) "
            "VALUES (%s,%s,%s,%s,%s,%s, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326))) "
            "ON CONFLICT (site) DO UPDATE SET geom=EXCLUDED.geom",
            (site, cfg["label"], cfg["subtitle"], cfg["story"], cfg["aoi_source"],
             cfg["intervention_label"], geom))
        m = metrics_all.get(site)
        run_id = None
        if m:
            cur.execute(
                "INSERT INTO model_run(site,chosen_model,train_end,monitor_start,scenes,metrics) "
                "VALUES (%s,%s,%s,%s,%s,%s) RETURNING run_id",
                (site, m.get("chosen_model"), m.get("train_end"), m.get("monitor_start"),
                 m.get("scenes"), json.dumps(m)))
            run_id = cur.fetchone()[0]
            changes = json.loads(site_paths(site)["changes_geojson"].read_text()) \
                if site_paths(site)["changes_geojson"].exists() else {"features": []}
            for f in changes["features"]:
                pr = f["properties"]
                cur.execute(
                    "INSERT INTO change_alert(run_id,site,change_type,area_ha,onset_date,mean_magnitude,"
                    "max_run,confidence,confidence_level,geom) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                    "ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326)))",
                    (run_id, site, pr["change_type"], pr["area_ha"], pr["onset_date"] or None,
                     pr["mean_magnitude"], pr["max_run"], pr["confidence"], pr["confidence_level"],
                     json.dumps(f["geometry"])))
        cat = site_paths(site)["catalog"]
        if cat.exists():
            import pandas as pd
            for r in pd.read_csv(cat).to_dict("records"):
                cur.execute(
                    "INSERT INTO scene_catalog(site,date,item_id,mgrs,cloud_cover,clear_fraction) "
                    "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (site,date) DO NOTHING",
                    (site, r["date"], r["item_id"], r.get("mgrs"), r.get("cloud_cover"), r.get("clear_fraction")))
        print(f"[{site}] loaded (run_id={run_id})")
    conn.commit()
    conn.close()
    print("PostGIS load complete.")


if __name__ == "__main__":
    main()
