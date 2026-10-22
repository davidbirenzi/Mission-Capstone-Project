"""PDF summary report per wetland (ReportLab), generated from stored outputs."""
import io
from pathlib import Path

from src.config import WETLANDS, site_paths


def build_report(site: str, metrics: dict, changes: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    cfg = WETLANDS[site]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Persistent Wetland Change Report — {cfg['label']}", styles["Title"]),
             Paragraph(cfg["subtitle"], styles["Heading3"]),
             Paragraph(f"Context: {cfg['intervention_label']}. Expected signal: {cfg['expected_change']}.", styles["BodyText"]),
             Spacer(1, 0.3 * cm)]
    if not metrics:
        story.append(Paragraph("Analysis has not been run for this wetland yet. Run <b>python -m src.analysis --all</b> first.", styles["BodyText"]))
        doc.build(story)
        return buf.getvalue()

    chosen = metrics.get("chosen_model", "")
    model_scores = metrics.get("models", {}).get(chosen, {})
    baseline = metrics.get("models", {}).get("Two-date baseline", {})
    rows = [["Measure", "Value"],
            ["Scenes used", f"{metrics['scenes']} ({metrics['first_date']} to {metrics['last_date']})"],
            ["Normal / monitoring scenes", f"{metrics['train_scenes']} / {metrics['monitor_scenes']}"],
            ["Wetland (AOI) area", f"{metrics['aoi_hectares']} ha"],
            ["Detected persistent change inside AOI", f"{metrics['area_hectares']['detected_inside_aoi']} ha"],
            ["Detected in whole study window", f"{metrics['area_hectares']['detected_window']} ha"],
            ["Detected in control ring (outside AOI)", f"{metrics['area_hectares']['detected_control_ring']} ha"],
            ["Change polygons (alerts)", str(metrics.get("n_change_polygons", 0))],
            ["Proposed model", chosen],
            ["Proposed F1 / Precision / Recall / IoU", " / ".join(str(round(model_scores.get(k, 0), 3)) for k in ("f1", "precision", "recall", "iou"))],
            ["Baseline F1 / Precision / Recall / IoU", " / ".join(str(round(baseline.get(k, 0), 3)) for k in ("f1", "precision", "recall", "iou"))]]
    t = Table(rows, colWidths=[7 * cm, 9.5 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4e89")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4fb")]),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story += [t, Spacer(1, 0.4 * cm), Paragraph("Persistent change by type (inside AOI, hectares)", styles["Heading2"])]
    type_rows = [["Change type", "Hectares"]] + [[k, v] for k, v in sorted(metrics.get("change_by_type_inside_aoi", {}).items(), key=lambda kv: -kv[1])]
    t2 = Table(type_rows, colWidths=[10 * cm, 6.5 * cm])
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15803d")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story.append(t2)

    for fname, caption in [("change_map.png", "Persistent-change map"), ("ndvi_timeseries.png", "Median NDVI: wetland vs control ring")]:
        p = Path(site_paths(site)["figures"]) / fname
        if p.exists():
            story += [Spacer(1, 0.4 * cm), Paragraph(caption, styles["Heading2"]),
                      Image(str(p), width=15.5 * cm, height=11 * cm, kind="proportional")]

    story += [Spacer(1, 0.4 * cm), Paragraph("Largest detected changes", styles["Heading2"])]
    feats = sorted(changes.get("features", []), key=lambda f: -f["properties"]["area_ha"])[:10]
    arows = [["#", "Type", "Area (ha)", "Onset", "Confidence"]] + [
        [f["properties"]["id"], f["properties"]["change_type"], f["properties"]["area_ha"],
         f["properties"]["onset_date"], f["properties"]["confidence_level"]] for f in feats]
    t3 = Table(arows, colWidths=[1 * cm, 7 * cm, 3 * cm, 3 * cm, 2.5 * cm])
    t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(t3)
    story += [Spacer(1, 0.4 * cm),
              Paragraph("Method in brief: seasonal normal-behaviour models (harmonic regression and an MLP) learn expected NDVI from the normal period. A pixel is reported only when its deviation persists for consecutive clear Sentinel-2 observations. Changes are typed from NDVI/MNDWI/NDMI/BSI shifts, dated by onset, scored for confidence and evaluated on spatially held-out blocks against a persistent-shift reference and a two-date baseline. See docs/METHODS.md.", styles["BodyText"]),
              Paragraph(f"AOI source: {cfg['aoi_source']}. Generated by the Kigali Wetland Persistent-Change Monitor (David Birenzi).", styles["BodyText"])]
    doc.build(story)
    return buf.getvalue()
