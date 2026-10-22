import nbformat as nbf
from pathlib import Path
nb = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "name": "python3"}})
def md(s): nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s): nb.cells.append(nbf.v4.new_code_cell(s))

md("""# Kigali Wetland Persistent-Change Monitor
**David Birenzi — BSc Software Engineering capstone**

Machine-learning monitoring of persistent change in Kigali wetlands from multi-temporal Sentinel-2 imagery.

**Research idea.** Two-date change detection fires on seasonality and temporary states. This notebook learns each wetland pixel's *normal seasonal behaviour*, tracks the deviation (residual) from that normal, and reports a change only when the deviation **persists across consecutive clear observations** — with an onset date, change type, area in hectares and confidence.

**Study wetlands** (contrasting cases): Nyandungu (restoration), Gikondo (industrial degradation → rehabilitation), Rwampara (urban/agricultural degradation → rehabilitation). Land outside each boundary inside the study window is the control ring.

Run the data pipeline first if cubes are missing: `python -m src.download --all` (internet, resumable). Everything below then runs offline.""")
code("""import json, sys
sys.path.append('..')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from src.config import WETLANDS
pd.set_option('display.max_columns', 20)
print('Sites:', {k: v['label'] for k, v in WETLANDS.items()})""")
md("""## 1 · Data
Sentinel-2 L2A, one best scene per quarter (2017–2026), SCL cloud/shadow masking, six indices at 10 m: NDVI, NDWI, MNDWI, NDMI, SAVI, BSI. Scene catalogues are stored per site so the work is reproducible from scene IDs.""")
code("""from src.config import site_paths
for site in WETLANDS:
    cat = site_paths(site)['catalog']
    if cat.exists():
        df = pd.read_csv(cat)
        print(site, len(df), 'scenes', df.date.min(), '->', df.date.max())
    else:
        print(site, 'NO CUBE YET - run: python -m src.download --site', site)""")
md("""## 2 · Models
Two normal-behaviour models are fitted per wetland on its normal period and compared:
- **Harmonic regression** per pixel (seasonal sin/cos terms) — transparent, per-pixel level+seasonality.
- **MLP regressor** (64–32 ReLU, Adam, early stopping): seasonal harmonics + the pixel's normal NDVI/NDWI/MNDWI → expected NDVI.

The **persistence engine** flags a pixel when |observed − expected NDVI| ≥ threshold for *k* consecutive clear monitoring observations (threshold and *k* tuned on tuning blocks only). Changes are typed from NDVI/MNDWI/NDMI/BSI shifts. The **baseline** is conventional two-date NDVI differencing.""")
code("""# Architecture of the MLP (one of the two normal-behaviour models)
display(Image('../outputs/figures/nyandungu/architecture.png'))""")
md("""## 3 · Run the full analysis (all wetlands)
Trains both models per site, tunes the persistence rule, evaluates on spatially held-out blocks, builds change polygons/alerts, figures and metrics.""")
code("""from src.analysis import run_all
all_metrics = run_all()""")
md("""## 4 · Results — model vs baseline (held-out spatial blocks)""")
code("""rows = []
for site, m in all_metrics.items():
    for name, s in m['models'].items():
        rows.append({'site': site, 'method': name, **{k: round(s[k], 3) for k in ['precision','recall','f1','accuracy','iou']}})
pd.DataFrame(rows)""")
code("""for site, m in all_metrics.items():
    print(f\"{m['label']}: chosen={m['chosen_model']} | detected inside={m['area_hectares']['detected_inside_aoi']} ha of {m['aoi_hectares']} ha | control-ring={m['area_hectares']['detected_control_ring']} ha | alerts={m['n_change_polygons']}\")
    print('   by type:', m['change_by_type_inside_aoi'])""")
md("""## 5 · Maps and time series""")
code("""for site in all_metrics:
    print(site)
    display(Image(f'../outputs/figures/{site}/change_map.png'))
    display(Image(f'../outputs/figures/{site}/ndvi_timeseries.png'))""")
md("""## 6 · Discussion (write-up prompts)
Answer these in the dissertation from the tables/maps above:
1. Does the proposed (persistent) model beat the two-date baseline on F1/IoU at every site? Where is precision lost, and does the map suggest those are real changes outside the reference definition?
2. Do onset years line up with the known histories (Nyandungu works 2018–2022; Gikondo/Rwampara rehabilitation from 2024)?
3. Does the control ring stay comparatively quiet, i.e. is the signal wetland-specific?
4. Which change types dominate each wetland, and does that match its story?
5. Limitations: silver-standard reference, quarterly onset precision, Rwampara delineated AOI, <0.1 ha changes invisible — see `docs/METHODS.md`. Field verification (`data/field/`) is the gold-standard next step.""")
code("""# Alerts register preview (all sites)
import glob
rows=[]
for f in glob.glob('../outputs/changes_*.geojson'):
    d=json.load(open(f))
    rows += [x['properties'] for x in d['features']]
pd.DataFrame(rows).sort_values('area_ha', ascending=False).head(15)""")

out = Path("notebooks/kigali_wetland_persistent_change.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
