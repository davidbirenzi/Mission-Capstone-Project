-- PostgreSQL/PostGIS schema for the deployed wetland monitoring database.
-- The local demo runs from files in outputs/; src/load_postgis.py loads the
-- same results into these tables when a database is available (docker compose up).
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS wetland (
    site            TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    subtitle        TEXT,
    story           TEXT,
    aoi_source      TEXT,
    intervention    TEXT,
    geom            GEOMETRY(MultiPolygon, 4326)
);

CREATE TABLE IF NOT EXISTS scene_catalog (
    site            TEXT REFERENCES wetland(site),
    date            DATE NOT NULL,
    item_id         TEXT,
    mgrs            TEXT,
    cloud_cover     DOUBLE PRECISION,
    clear_fraction  DOUBLE PRECISION,
    PRIMARY KEY (site, date)
);

CREATE TABLE IF NOT EXISTS model_run (
    run_id          SERIAL PRIMARY KEY,
    site            TEXT REFERENCES wetland(site),
    created_at      TIMESTAMPTZ DEFAULT now(),
    chosen_model    TEXT,
    train_end       DATE,
    monitor_start   DATE,
    scenes          INT,
    metrics         JSONB
);

CREATE TABLE IF NOT EXISTS change_alert (
    alert_id        SERIAL PRIMARY KEY,
    run_id          INT REFERENCES model_run(run_id),
    site            TEXT REFERENCES wetland(site),
    change_type     TEXT NOT NULL,
    area_ha         DOUBLE PRECISION,
    onset_date      DATE,
    mean_magnitude  DOUBLE PRECISION,
    max_run         INT,
    confidence      DOUBLE PRECISION,
    confidence_level TEXT,
    geom            GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS change_alert_geom_idx ON change_alert USING GIST (geom);
CREATE INDEX IF NOT EXISTS change_alert_site_idx ON change_alert (site, change_type);

CREATE TABLE IF NOT EXISTS field_verification (
    id              SERIAL PRIMARY KEY,
    site            TEXT REFERENCES wetland(site),
    alert_id        INT,
    visited_at      DATE,
    lon             DOUBLE PRECISION,
    lat             DOUBLE PRECISION,
    photo_path      TEXT,
    observed_cover  TEXT,
    confirmed       BOOLEAN,
    notes           TEXT,
    geom            GEOMETRY(Point, 4326)
);
