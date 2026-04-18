import geopandas as gpd
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyproj import Transformer
from io import BytesIO
import time
import argparse
import configparser
import csv
import os
import json

# ================= AUTH =================

def authenticate(client_id, client_secret):
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )
    return token["access_token"]

# ================= AREA =================

def get_polygon(path):
    gdf = gpd.read_file(path, driver="KML")
    return list(gdf.geometry[0].exterior.coords)

# ================= SIZE =================

def get_bbox_size(coords):
    lon = [c[0] for c in coords]
    lat = [c[1] for c in coords]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(lon, lat)
    return max(xs) - min(xs), max(ys) - min(ys)

# ================= TREE MASK (DOI: 10.1016/j.compag.2020.105500) =================

TREE_MASK_DEFAULTS = {
    "cvi_threshold":    2.0,
    "ndvi_threshold":   0.4,
    "shadow_threshold": 0.08,
}

EVALSCRIPT_TREE_BANDS = """//VERSION=3
function setup() {
    return {
        input: ["B03","B04","B05","B08","SCL","dataMask"],
        output: { bands: 4, sampleType: "FLOAT32" },
        mosaicking: "ORBIT"
    };
}
function evaluatePixel(samples) {
    for (let s of samples) {
        if (s.dataMask === 0) continue;
        if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
        return [s.B03, s.B04, s.B05, s.B08];
    }
    return [NaN, NaN, NaN, NaN];
}
"""

def compute_tree_mask(b03, b04, b08,
                      cvi_threshold=2.0,
                      ndvi_threshold=0.4,
                      shadow_threshold=0.08):
    with np.errstate(divide="ignore", invalid="ignore"):
        cvi  = np.where(b03 > 0, (b08 / b03) * (b04 / b03), np.nan)
        ndvi = np.where((b08 + b04) > 0, (b08 - b04) / (b08 + b04), np.nan)

    mask = (~np.isnan(cvi)) & (~np.isnan(ndvi))
    mask &= (cvi  >= cvi_threshold)
    mask &= (ndvi >= ndvi_threshold)
    if shadow_threshold > 0:
        mask &= (b08 >= shadow_threshold)

    return mask, cvi, ndvi

# ================= CSV: SINGOLE RILEVAZIONI =================

CSV_FIELDNAMES = ["timestamp", "start", "end", "tree_focused", "index", "value"]

def save_datapoint(filepath, start, end, tree_focused, index_name, value):
    """Salva una singola rilevazione (un indice, una finestra temporale) nel CSV."""
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "start":        start,
            "end":          end,
            "tree_focused": tree_focused,
            "index":        index_name,
            "value":        value,
        })


def load_datapoints(filepath, tree_focused=None):
    """
    Carica tutte le rilevazioni dal CSV.
    Ritorna dict: { index_name: [ {"start": ..., "end": ..., "value": float}, ... ] }
    Se tree_focused non è None filtra per quel valore.
    """
    if not os.path.exists(filepath):
        return {}

    result = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if tree_focused is not None:
                if str(row["tree_focused"]).lower() != str(tree_focused).lower():
                    continue
            idx = row["index"]
            val = row["value"]
            if val == "" or val is None:
                continue
            result.setdefault(idx, []).append({
                "start": row["start"],
                "end":   row["end"],
                "value": float(val),
            })
    return result


def load_period_means(filepath, start, end, tree_focused):
    """
    Carica le medie per un periodo specifico (start/end esatti).
    Usato per evitare di riscaricaire dati già presenti.
    Ritorna dict: { index_name: float } oppure None se non trovato.
    """
    if not os.path.exists(filepath):
        return None

    found = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["start"] == start and
                    row["end"] == end and
                    str(row["tree_focused"]).lower() == str(tree_focused).lower()):
                val = row["value"]
                if val != "":
                    found[row["index"]] = float(val)

    return found if found else None


# ================= HTML CHART =================

def generate_html_chart(filepath, data_file, tree_focused, output_html="chart.html"):
    """
    Genera un file HTML con grafici a linee interattivi e scorrevoli per tutti
    gli indici presenti nel CSV. Il grafico mostra un tooltip con la data al
    passaggio del mouse.
    """
    datapoints = load_datapoints(data_file, tree_focused=tree_focused)
    if not datapoints:
        print("⚠ Nessun dato nel CSV per generare il grafico HTML.")
        return

    # Prepara i dataset per Chart.js
    palette = [
        "#5588ff", "#ff7055", "#55ddaa", "#ffcc44",
        "#cc88ff", "#44ddff", "#ff99cc", "#88ff66",
    ]

    datasets = []
    for i, (idx_name, points) in enumerate(sorted(datapoints.items())):
        # Ordina per data di inizio
        points_sorted = sorted(points, key=lambda p: p["start"])
        color = palette[i % len(palette)]
        datasets.append({
            "label": idx_name.upper(),
            "data": [
                {"x": p["start"], "y": round(p["value"], 5), "end": p["end"]}
                for p in points_sorted
            ],
            "borderColor": color,
            "backgroundColor": color + "22",
            "pointBackgroundColor": color,
            "pointRadius": 5,
            "pointHoverRadius": 8,
            "tension": 0.35,
            "fill": False,
        })

    datasets_json = json.dumps(datasets, ensure_ascii=False)
    tree_label = "chiome arboree" if str(tree_focused).lower() == "true" else "campo intero"

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentinel-2 · Analisi Indici Vegetazione</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #07080f;
    --surface:   #0e1020;
    --border:    #1e2240;
    --accent:    #5588ff;
    --text:      #c8d0f0;
    --muted:     #5a6080;
    --highlight: #ffffff;
  }}

  html, body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
  }}

  header {{
    padding: 2.5rem 2rem 1rem;
    border-bottom: 1px solid var(--border);
  }}

  header h1 {{
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.1rem, 2.5vw, 1.6rem);
    font-weight: 700;
    color: var(--highlight);
    letter-spacing: -0.02em;
  }}

  header p {{
    margin-top: .35rem;
    font-size: .85rem;
    color: var(--muted);
    font-weight: 300;
  }}

  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: .6rem;
    padding: 1.2rem 2rem;
    border-bottom: 1px solid var(--border);
    align-items: center;
  }}

  .btn-idx {{
    font-family: 'Space Mono', monospace;
    font-size: .72rem;
    padding: .35rem .85rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    transition: all .18s ease;
    letter-spacing: .03em;
  }}

  .btn-idx.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    box-shadow: 0 0 12px var(--accent)55;
  }}

  .btn-idx:hover:not(.active) {{
    border-color: var(--accent);
    color: var(--accent);
  }}

  .btn-all {{
    font-family: 'Space Mono', monospace;
    font-size: .72rem;
    padding: .35rem .85rem;
    border-radius: 999px;
    border: 1px solid #ffffff22;
    background: #ffffff0a;
    color: var(--text);
    cursor: pointer;
    margin-left: auto;
    transition: all .18s ease;
  }}

  .btn-all:hover {{ background: #ffffff18; }}

  .chart-wrap {{
    position: relative;
    padding: 1.5rem 2rem 1rem;
    overflow-x: auto;
  }}

  .chart-inner {{
    min-width: 700px;
    height: 420px;
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: .8rem;
    padding: .8rem 2rem 2rem;
  }}

  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .9rem 1rem;
    transition: border-color .2s;
  }}

  .stat-card:hover {{ border-color: var(--accent); }}

  .stat-card .idx-name {{
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    color: var(--muted);
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .3rem;
  }}

  .stat-card .idx-val {{
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--highlight);
    line-height: 1;
  }}

  .stat-card .idx-label {{
    margin-top: .25rem;
    font-size: .72rem;
    color: var(--muted);
  }}

  .tooltip-box {{
    background: #0d1020ee !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: .6rem .9rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .75rem !important;
    color: var(--text) !important;
    box-shadow: 0 4px 24px #00000088 !important;
  }}

  footer {{
    text-align: center;
    padding: 1.2rem;
    font-size: .72rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
  }}

  ::-webkit-scrollbar {{ height: 4px; background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
</style>
</head>
<body>

<header>
  <h1>&#9632; Sentinel-2 — Analisi Indici Vegetazione</h1>
  <p>Modalità: <strong>{tree_label}</strong> &nbsp;·&nbsp; File dati: <code>{os.path.basename(data_file)}</code></p>
</header>

<div class="controls" id="controls">
  <button class="btn-all" onclick="toggleAll()">Tutti / Nessuno</button>
</div>

<div class="chart-wrap">
  <div class="chart-inner">
    <canvas id="mainChart"></canvas>
  </div>
</div>

<div class="stats-grid" id="statsGrid"></div>

<footer>generato il {datetime.now().strftime("%d/%m/%Y %H:%M")} &nbsp;·&nbsp; Sentinel-2 L2A</footer>

<script>
const RAW_DATASETS = {datasets_json};

// Calcola statistiche per le card
function buildStats(datasets) {{
  const grid = document.getElementById('statsGrid');
  grid.innerHTML = '';
  datasets.forEach(ds => {{
    const vals = ds.data.map(p => p.y).filter(v => v !== null && !isNaN(v));
    if (!vals.length) return;
    const last  = vals[vals.length - 1];
    const avg   = vals.reduce((a,b) => a+b, 0) / vals.length;
    const trend = vals.length > 1 ? last - vals[vals.length - 2] : 0;
    const arrow = trend > 0 ? '▲' : trend < 0 ? '▼' : '—';
    const tcolor = trend > 0 ? '#55dd88' : trend < 0 ? '#ff6655' : '#888';

    const card = document.createElement('div');
    card.className = 'stat-card';
    card.innerHTML = `
      <div class="idx-name">${{ds.label}}</div>
      <div class="idx-val">${{last.toFixed(4)}}</div>
      <div class="idx-label" style="color:${{tcolor}}">${{arrow}} ${{Math.abs(trend).toFixed(4)}} &nbsp;·&nbsp; media ${{avg.toFixed(4)}}</div>
    `;
    grid.appendChild(card);
  }});
}}

// Pulsanti filtro
const activeSet = new Set(RAW_DATASETS.map(d => d.label));

function renderButtons() {{
  const ctrl = document.getElementById('controls');
  // rimuovi vecchi btn
  ctrl.querySelectorAll('.btn-idx').forEach(b => b.remove());
  const all = ctrl.querySelector('.btn-all');
  RAW_DATASETS.forEach(ds => {{
    const btn = document.createElement('button');
    btn.className = 'btn-idx' + (activeSet.has(ds.label) ? ' active' : '');
    btn.textContent = ds.label;
    btn.style.setProperty('--c', ds.borderColor);
    btn.addEventListener('click', () => {{
      if (activeSet.has(ds.label)) activeSet.delete(ds.label);
      else activeSet.add(ds.label);
      updateChart();
      renderButtons();
    }});
    ctrl.insertBefore(btn, all);
  }});
}}

function toggleAll() {{
  if (activeSet.size === RAW_DATASETS.length) activeSet.clear();
  else RAW_DATASETS.forEach(d => activeSet.add(d.label));
  updateChart();
  renderButtons();
}}

function updateChart() {{
  const visible = RAW_DATASETS.filter(d => activeSet.has(d.label));
  chart.data.datasets = visible;
  chart.update('active');
  buildStats(visible);
}}

// Chart.js
const ctx = document.getElementById('mainChart').getContext('2d');
const chart = new Chart(ctx, {{
  type: 'line',
  data: {{ datasets: RAW_DATASETS }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{
      mode: 'index',
      intersect: false,
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#0d1020ee',
        borderColor: '#1e2240',
        borderWidth: 1,
        titleFont: {{ family: 'Space Mono', size: 11 }},
        bodyFont: {{ family: 'DM Sans', size: 12 }},
        titleColor: '#c8d0f0',
        bodyColor: '#8899cc',
        padding: 12,
        callbacks: {{
          title: function(items) {{
            if (!items.length) return '';
            const pt = items[0].raw;
            return pt.end
              ? `${{pt.x}}  →  ${{pt.end}}`
              : pt.x;
          }},
          label: function(item) {{
            return `  ${{item.dataset.label}}: ${{item.parsed.y.toFixed(5)}}`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        type: 'category',
        ticks: {{
          color: '#5a6080',
          font: {{ family: 'Space Mono', size: 10 }},
          maxRotation: 45,
        }},
        grid: {{ color: '#1e224044' }},
        title: {{
          display: true,
          text: 'Inizio finestra',
          color: '#5a6080',
          font: {{ family: 'DM Sans', size: 11 }},
        }}
      }},
      y: {{
        ticks: {{
          color: '#5a6080',
          font: {{ family: 'Space Mono', size: 10 }},
        }},
        grid: {{ color: '#1e224088' }},
        title: {{
          display: true,
          text: 'Valore indice',
          color: '#5a6080',
          font: {{ family: 'DM Sans', size: 11 }},
        }}
      }}
    }}
  }}
}});

renderButtons();
buildStats(RAW_DATASETS);
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → Grafico HTML salvato in: {output_html}")


# ================= DESCRIZIONI INDICI =================

INDEX_DESCRIPTIONS = {
    "rgb": {
        "title": "RGB – Immagine a colori naturali",
        "formula": "Bande: B04 (Rosso) · B03 (Verde) · B02 (Blu)",
        "desc": (
            "Composizione a colori reali che riproduce ciò che vedrebbe l'occhio umano dall'alto. "
            "Non è un indice quantitativo: serve come riferimento visivo per interpretare il contesto "
            "geografico e confrontare le mappe degli indici con la realtà del terreno."
        ),
        "diff": "Unico indice non numerico — base di confronto visivo per tutti gli altri."
    },
    "ndvi": {
        "title": "NDVI – Normalized Difference Vegetation Index",
        "formula": "(B08 - B04) / (B08 + B04)     Range: -1 → +1",
        "desc": (
            "Misura la quantità di biomassa verde sfruttando il forte assorbimento del rosso (B04) "
            "da parte della clorofilla e l'alta riflessione del vicino infrarosso (B08) "
            "da parte della struttura cellulare fogliare."
        ),
        "diff": (
            "Il più diffuso e generale. Può saturare su canopie molto dense e sovrastimare "
            "la vegetazione su suoli scuri. Punto di partenza per tutti gli altri indici."
        )
    },
    "ndre": {
        "title": "NDRE – Normalized Difference Red-Edge Index",
        "formula": "(B08 - B05) / (B08 + B05)     Range: -1 → +1",
        "desc": (
            "Sfrutta la banda Red-Edge (B05, ~705 nm), zona spettrale molto sensibile alle variazioni "
            "di clorofilla. Rileva stress iniziali ancora invisibili all'NDVI, "
            "ed è meno soggetto a saturazione su vegetazione densa."
        ),
        "diff": (
            "Rispetto all'NDVI: più preciso su stress precoci e canopie mature. "
            "Rispetto al GNDVI: usa il red-edge invece del verde, più sensibile a piccole variazioni di clorofilla."
        )
    },
    "gndvi": {
        "title": "GNDVI – Green Normalized Difference Vegetation Index",
        "formula": "(B08 - B03) / (B08 + B03)     Range: -1 → +1",
        "desc": (
            "Variante dell'NDVI che usa il verde (B03) al posto del rosso. "
            "Il canale verde è più sensibile alle variazioni di concentrazione di clorofilla, "
            "rendendolo utile per stimare la qualità fisiologica della vegetazione."
        ),
        "diff": (
            "Rispetto all'NDVI: misura qualità della vegetazione, non solo quantità. "
            "Rispetto al NDRE: usa il verde standard invece del red-edge; meno preciso su stress precoci."
        )
    },
    "gci": {
        "title": "GCI – Green Chlorophyll Index",
        "formula": "(B08 / B03) - 1     Range: 0 → ~10+",
        "desc": (
            "Stima diretta del contenuto di clorofilla nella chioma. "
            "Usa un rapporto semplice (non normalizzato), risultando più lineare "
            "rispetto alla concentrazione di clorofilla. Molto utile per guidare "
            "la fertilizzazione azotata in agricoltura di precisione."
        ),
        "diff": (
            "A differenza di NDVI/GNDVI non è limitato al range -1/+1: "
            "valori alti (>8) indicano clorofilla molto elevata. "
            "Più interpretabile come stima assoluta di clorofilla rispetto agli indici normalizzati."
        )
    },
    "savi": {
        "title": "SAVI – Soil Adjusted Vegetation Index",
        "formula": "((B08 - B04) / (B08 + B04 + 0.5)) × 1.5     Range: -1 → +1",
        "desc": (
            "Versione corretta dell'NDVI che introduce il fattore L=0.5 per ridurre "
            "l'influenza del suolo sul segnale vegetale. Quando la copertura è scarsa (<30%), "
            "il suolo altera significativamente l'NDVI; il SAVI compensa questo effetto."
        ),
        "diff": (
            "Rispetto all'NDVI: più affidabile su suoli esposti, aree semi-aride e pascoli radi. "
            "Se SAVI ≈ NDVI l'effetto suolo è trascurabile; se divergono, fidarsi del SAVI."
        )
    },
    "pri": {
        "title": "PRI – Photochemical Reflectance Index",
        "formula": "(B02 - B03) / (B02 + B03)     Range: -0.1 → +0.1",
        "desc": (
            "Misura l'efficienza fotosintetica attraverso il ciclo delle xantofille: "
            "pigmenti che cambiano configurazione in risposta all'intensità luminosa. "
            "Valori positivi → alta efficienza; valori negativi → stress fotosintetico."
        ),
        "diff": (
            "Unico indice che misura non la quantità o la clorofilla, ma COME la pianta "
            "sta usando la luce in quel momento. Un NDVI alto con PRI negativo segnala "
            "vegetazione densa ma in stress fotosintetico attivo."
        )
    },
    "mcari": {
        "title": "MCARI – Modified Chlorophyll Absorption in Reflectance Index",
        "formula": "((B05 - B04) - 0.2 × (B05 - B03)) × (B05 / B04)",
        "desc": (
            "Stima la concentrazione di clorofilla minimizzando l'influenza della radiazione "
            "non fotosinteticamente attiva. Usa tre bande (verde, rosso, red-edge) "
            "per isolare il segnale della clorofilla fogliare."
        ),
        "diff": (
            "Rispetto al TCARI: meno aggressivo nell'amplificazione, più stabile su vegetazione rada. "
            "Particolarmente robusto su colture erbacee e cereali."
        )
    },
    "tcari": {
        "title": "TCARI – Transformed Chlorophyll Absorption in Reflectance Index",
        "formula": "3 × ((B05 - B04) - 0.2 × (B05 - B03) × (B05 / B04))",
        "desc": (
            "Versione amplificata del MCARI (fattore ×3). Più sensibile su range bassi di clorofilla, "
            "ideale per rilevare stress clorofilliani precoci e leggeri ancora non evidenti ad altri indici."
        ),
        "diff": (
            "Rispetto al MCARI: tripla sensibilità → più preciso su stress iniziali ma più rumoroso "
            "su vegetazione rada. Se TCARI < MCARI (in score) indica stress precoce in atto."
        )
    },
}

# ================= PESI PER VALUTAZIONE FINALE =================

INDEX_WEIGHTS = {
    "ndvi":  0.20,
    "ndre":  0.20,
    "gndvi": 0.15,
    "gci":   0.15,
    "savi":  0.10,
    "pri":   0.10,
    "mcari": 0.05,
    "tcari": 0.05,
}

# ================= NORMALIZZAZIONE A SCORE 0-100 =================

def normalize_to_score(name, value):
    if value is None or np.isnan(value):
        return None
    ranges = {
        "ndvi":  (-0.2, 1.0),
        "ndre":  (0.0,  0.6),
        "gndvi": (0.0,  0.8),
        "gci":   (0.0,  10.0),
        "savi":  (0.0,  0.8),
        "pri":   (-0.1, 0.1),
        "mcari": (0.0,  1.0),
        "tcari": (0.0,  1.5),
    }
    lo, hi = ranges.get(name, (0, 1))
    score = (value - lo) / (hi - lo) * 100
    return float(np.clip(score, 0, 100))

# ================= EVALSCRIPTS (indici singoli) =================

EVALSCRIPTS = {
    "rgb": """//VERSION=3
    function setup() {
        return {
            input: ["B02","B03","B04","SCL","dataMask"],
            output: { bands: 3, sampleType: "UINT8" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [
                Math.min(255, s.B04 * 255 * 3.5),
                Math.min(255, s.B03 * 255 * 3.5),
                Math.min(255, s.B02 * 255 * 3.5)
            ];
        }
        return [0, 0, 0];
    }
    """,
    "ndvi": """//VERSION=3
    function setup() {
        return {
            input: ["B04","B08","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [(s.B08 - s.B04) / (s.B08 + s.B04)];
        }
        return [NaN];
    }
    """,
    "ndre": """//VERSION=3
    function setup() {
        return {
            input: ["B05","B08","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [(s.B08 - s.B05) / (s.B08 + s.B05)];
        }
        return [NaN];
    }
    """,
    "gndvi": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B08","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [(s.B08 - s.B03) / (s.B08 + s.B03)];
        }
        return [NaN];
    }
    """,
    "gci": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B08","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [(s.B08 / s.B03) - 1.0];
        }
        return [NaN];
    }
    """,
    "savi": """//VERSION=3
    function setup() {
        return {
            input: ["B04","B08","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            let L = 0.5;
            return [((s.B08 - s.B04) / (s.B08 + s.B04 + L)) * (1 + L)];
        }
        return [NaN];
    }
    """,
    "pri": """//VERSION=3
    function setup() {
        return {
            input: ["B02","B03","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [(s.B02 - s.B03) / (s.B02 + s.B03)];
        }
        return [NaN];
    }
    """,
    "mcari": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B05","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [((s.B05 - s.B04) - 0.2 * (s.B05 - s.B03)) * (s.B05 / s.B04)];
        }
        return [NaN];
    }
    """,
    "tcari": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B05","SCL","dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            let tcari = 3 * ((s.B05 - s.B04) - 0.2 * (s.B05 - s.B03) * (s.B05 / s.B04));
            return [tcari];
        }
        return [NaN];
    }
    """,
}

# ================= EVALSCRIPTS MULTI-BANDA (tree-focused) =================

EVALSCRIPTS_TREE = {
    "ndvi": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [s.B03, s.B04, s.B08, (s.B08 - s.B04) / (s.B08 + s.B04)];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "ndre": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B05","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [s.B03, s.B04, s.B08, (s.B08 - s.B05) / (s.B08 + s.B05)];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "gndvi": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [s.B03, s.B04, s.B08, (s.B08 - s.B03) / (s.B08 + s.B03)];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "gci": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [s.B03, s.B04, s.B08, (s.B08 / s.B03) - 1.0];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "savi": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            let L = 0.5;
            return [s.B03, s.B04, s.B08, ((s.B08 - s.B04) / (s.B08 + s.B04 + L)) * (1 + L)];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "pri": """//VERSION=3
    function setup() {
        return {
            input: ["B02","B03","B04","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            return [s.B03, s.B04, s.B08, (s.B02 - s.B03) / (s.B02 + s.B03)];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "mcari": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B05","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            let mcari = ((s.B05 - s.B04) - 0.2 * (s.B05 - s.B03)) * (s.B05 / s.B04);
            return [s.B03, s.B04, s.B08, mcari];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
    "tcari": """//VERSION=3
    function setup() {
        return {
            input: ["B03","B04","B05","B08","SCL","dataMask"],
            output: { bands: 4, sampleType: "FLOAT32" },
            mosaicking: "ORBIT"
        };
    }
    function evaluatePixel(samples) {
        for (let s of samples) {
            if (s.dataMask === 0) continue;
            if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10) continue;
            let tcari = 3 * ((s.B05 - s.B04) - 0.2 * (s.B05 - s.B03) * (s.B05 / s.B04));
            return [s.B03, s.B04, s.B08, tcari];
        }
        return [NaN, NaN, NaN, NaN];
    }
    """,
}

# ================= VALUTAZIONE TESTUALE =================

def evaluate_index(name, value):
    if value is None or np.isnan(value):
        return "Nessun dato"

    if name == "ndvi":
        if value < 0:    return "Acqua o superfici artificiali"
        if value < 0.2:  return "Suolo nudo o vegetazione molto scarsa"
        if value < 0.4:  return "Vegetazione rada o stress elevato"
        if value < 0.6:  return "Vegetazione moderata"
        if value < 0.8:  return "Vegetazione sana e densa"
        return                  "Vegetazione molto densa e vigorosa"

    if name == "ndre":
        if value < 0:    return "Suolo nudo o stress severo"
        if value < 0.2:  return "Stress elevato, clorofilla molto bassa"
        if value < 0.35: return "Stress moderato"
        if value < 0.5:  return "Vegetazione discreta"
        return                  "Vegetazione sana, buona clorofilla"

    if name == "gndvi":
        if value < 0.1:  return "Suolo nudo"
        if value < 0.3:  return "Vegetazione scarsa"
        if value < 0.5:  return "Vegetazione moderata"
        return                  "Vegetazione densa e sana"

    if name == "gci":
        if value < 1:    return "Clorofilla molto bassa, stress severo"
        if value < 3:    return "Clorofilla bassa"
        if value < 5:    return "Clorofilla nella norma"
        if value < 8:    return "Clorofilla elevata, buona salute"
        return                  "Clorofilla molto elevata"

    if name == "savi":
        if value < 0:    return "Suolo nudo o acqua"
        if value < 0.2:  return "Vegetazione molto scarsa"
        if value < 0.4:  return "Vegetazione rada"
        if value < 0.6:  return "Vegetazione moderata"
        return                  "Vegetazione densa"

    if name == "pri":
        if value < -0.05: return "Efficienza fotosintetica bassa, stress"
        if value < 0:     return "Efficienza fotosintetica nella norma"
        if value < 0.05:  return "Buona efficienza fotosintetica"
        return                   "Efficienza fotosintetica molto alta"

    if name == "mcari":
        if value < 0.1:  return "Clorofilla molto bassa"
        if value < 0.5:  return "Clorofilla moderata"
        return                  "Clorofilla elevata"

    if name == "tcari":
        if value < 0.1:  return "Clorofilla molto bassa"
        if value < 0.5:  return "Clorofilla moderata"
        return                  "Clorofilla elevata"

    return "N/D"

# ================= COLORMAP PER INDICE =================

def get_colormap_and_range(name):
    if name in ("ndvi", "ndre", "gndvi", "savi"):
        cmap = LinearSegmentedColormap.from_list("veg", [
            (0.0,  (0.55, 0.0,  0.0)),
            (0.25, (0.82, 0.41, 0.12)),
            (0.5,  (1.0,  1.0,  0.0)),
            (0.75, (0.56, 0.93, 0.56)),
            (1.0,  (0.0,  0.39, 0.0)),
        ])
        return cmap, -0.2, 1.0

    if name == "gci":
        cmap = LinearSegmentedColormap.from_list("gci", [
            (0.0,  (0.55, 0.0,  0.0)),
            (0.3,  (0.82, 0.41, 0.12)),
            (0.6,  (1.0,  1.0,  0.0)),
            (0.8,  (0.56, 0.93, 0.56)),
            (1.0,  (0.0,  0.39, 0.0)),
        ])
        return cmap, 0, 10

    if name == "pri":
        cmap = LinearSegmentedColormap.from_list("pri", [
            (0.0,  (0.8, 0.2, 0.0)),
            (0.5,  (1.0, 1.0, 0.0)),
            (1.0,  (0.0, 0.6, 0.0)),
        ])
        return cmap, -0.1, 0.1

    if name in ("mcari", "tcari"):
        cmap = LinearSegmentedColormap.from_list("chloro", [
            (0.0,  (0.55, 0.0,  0.0)),
            (0.5,  (1.0,  1.0,  0.0)),
            (1.0,  (0.0,  0.39, 0.0)),
        ])
        return cmap, 0, 1

    return plt.cm.RdYlGn, -1, 1

# ================= REQUEST =================

def make_request(token, polygon, width, height, start, end, mode, tree_focused=False):
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    resolution = 10
    width_px  = max(64, min(2500, int(width  / resolution)))
    height_px = max(64, min(2500, int(height / resolution)))

    if tree_focused and mode in EVALSCRIPTS_TREE:
        evalscript = EVALSCRIPTS_TREE[mode]
    elif mode in EVALSCRIPTS:
        evalscript = EVALSCRIPTS[mode]
    else:
        raise ValueError(f"Nessun evalscript per la modalità: {mode}")

    payload = {
        "input": {
            "bounds": {"geometry": {"type": "Polygon", "coordinates": [polygon]}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": start, "to": end},
                    "maxCloudCoverage": 50
                }
            }]
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [{"format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    return url, headers, payload

# ================= DOWNLOAD =================

def download_image(url, headers, payload):
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code != 200:
        print(f"Errore API: {r.status_code} - {r.text}")
        return None
    return BytesIO(r.content)

# ================= ESTRAI ARRAY, MASCHERA E MEDIA =================

def extract_array_and_mean(img_bytes, mode, tree_focused, tree_params):
    if img_bytes is None:
        return None, None, None, "campo intero"

    with rasterio.open(img_bytes) as src:
        data = src.read().astype(np.float32)

    if tree_focused and data.shape[0] >= 4:
        b03, b04, b08 = data[0], data[1], data[2]
        index_arr = data[3]
        tree_mask, _, _ = compute_tree_mask(
            b03, b04, b08,
            cvi_threshold    = tree_params["cvi_threshold"],
            ndvi_threshold   = tree_params["ndvi_threshold"],
            shadow_threshold = tree_params["shadow_threshold"],
        )
        map_array  = np.where(tree_mask, index_arr, np.nan)
        mask_array = tree_mask
        mean_label = "chiome arboree"
    else:
        map_array  = data[0]
        mask_array = None
        mean_label = "campo intero"

    valid = map_array[~np.isnan(map_array)]
    mean_val = float(np.mean(valid)) if len(valid) > 0 else None

    return map_array, mask_array, mean_val, mean_label


# ================= HOVER INTERATTIVO =================

def _attach_hover(fig, ax, array_2d, label_prefix="val"):
    annot = ax.annotate(
        "", xy=(0, 0),
        xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a3a", ec="#4488ff", lw=0.8, alpha=0.92),
        fontsize=7.5, color="#e0e8ff",
        arrowprops=dict(arrowstyle="->", color="#4488ff", lw=0.7),
    )
    annot.set_visible(False)

    def on_move(event):
        if event.inaxes != ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        col = int(round(event.xdata)) if event.xdata is not None else -1
        row = int(round(event.ydata)) if event.ydata is not None else -1
        h, w = array_2d.shape
        if 0 <= row < h and 0 <= col < w:
            val = array_2d[row, col]
            if np.isnan(val):
                txt = f"({col}, {row})\n— fuori maschera —"
            else:
                txt = f"({col}, {row})\n{label_prefix}: {val:.4f}"
            annot.set_text(txt)
            annot.xy = (event.xdata, event.ydata)
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def _attach_hover_mask(fig, ax, mask_2d):
    annot = ax.annotate(
        "", xy=(0, 0),
        xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1a0d", ec="#44bb44", lw=0.8, alpha=0.92),
        fontsize=7.5, color="#ccffcc",
        arrowprops=dict(arrowstyle="->", color="#44bb44", lw=0.7),
    )
    annot.set_visible(False)

    def on_move(event):
        if event.inaxes != ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        col = int(round(event.xdata)) if event.xdata is not None else -1
        row = int(round(event.ydata)) if event.ydata is not None else -1
        h, w = mask_2d.shape
        if 0 <= row < h and 0 <= col < w:
            is_tree = bool(mask_2d[row, col])
            lbl = "Chioma arborea" if is_tree else "Non chioma"
            annot.set_text(f"({col}, {row})\n{lbl}")
            annot.xy = (event.xdata, event.ydata)
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


# ================= FIGURA UNIFICATA: MAPPA + COPERTURA + TREND =================

def show_index_unified(token, polygon, width, height,
                       mode, start, end,
                       step_days=10,
                       tree_focused=False,
                       tree_params=None,
                       data_file=None):
    if tree_params is None:
        tree_params = dict(TREE_MASK_DEFAULTS)

    info = INDEX_DESCRIPTIONS.get(mode, {})
    cmap, vmin, vmax = get_colormap_and_range(mode)

    # ------------------------------------------------------------------ #
    # 1. Mappa media sull'intero periodo
    # ------------------------------------------------------------------ #
    s_full = f"{start}T00:00:00Z" if "T" not in start else start
    e_full = f"{end}T23:59:59Z"   if "T" not in end   else end

    print(f"\n[{mode.upper()}] Scarico mappa periodo intero {start} → {end} ...")
    url, headers, payload = make_request(
        token, polygon, width, height,
        s_full, e_full, mode,
        tree_focused=tree_focused,
    )
    img_full = download_image(url, headers, payload)
    map_array, mask_array, mean_full, mean_label = extract_array_and_mean(
        img_full, mode, tree_focused, tree_params
    )

    # ------------------------------------------------------------------ #
    # 2. Dati temporali: un punto ogni step_days giorni
    # ------------------------------------------------------------------ #
    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)

    dates_pts  = []
    values_pts = []
    current    = start_dt

    while current <= end_dt:
        win_end = min(current + relativedelta(days=step_days), end_dt)
        s_w = current.isoformat() + "Z"
        e_w = win_end.isoformat() + "Z"
        s_w_date = current.strftime("%Y-%m-%d")
        e_w_date = win_end.strftime("%Y-%m-%d")

        # Controlla se già salvato nel CSV
        cached = None
        if data_file:
            cached_dict = load_period_means(data_file, s_w_date, e_w_date, tree_focused)
            if cached_dict and mode in cached_dict:
                cached = cached_dict[mode]
                print(f"  trend {s_w_date} → {e_w_date} ... {cached:.4f}  [da CSV]")

        if cached is None:
            print(f"  trend {s_w_date} → {e_w_date} ...", end=" ", flush=True)
            url_w, hdr_w, pay_w = make_request(
                token, polygon, width, height,
                s_w, e_w, mode,
                tree_focused=tree_focused,
            )
            img_w = download_image(url_w, hdr_w, pay_w)
            _, _, mean_w, _ = extract_array_and_mean(img_w, mode, tree_focused, tree_params)
            mean_w_val = mean_w if mean_w is not None else np.nan
            print(f"{mean_w_val:.4f}" if not np.isnan(mean_w_val) else "no data")

            # Salva nel CSV
            if data_file and not np.isnan(mean_w_val):
                save_datapoint(data_file, s_w_date, e_w_date, tree_focused, mode, mean_w_val)

            cached = mean_w_val

        dates_pts.append(current)
        values_pts.append(cached)
        current += relativedelta(days=step_days)
        time.sleep(0.4)

    # ------------------------------------------------------------------ #
    # 3. Costruzione figura matplotlib
    # ------------------------------------------------------------------ #
    period_label = f"{start} → {end}"
    tree_tag     = "  [chiome arboree]" if tree_focused else ""

    fig = plt.figure(figsize=(16, 10) if tree_focused else (14, 10))
    fig.patch.set_facecolor("#0a0a14")

    if tree_focused and mask_array is not None:
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1.6, 1],
            hspace=0.42, wspace=0.12,
            left=0.05, right=0.97,
            top=0.93, bottom=0.07,
        )
        ax_map  = fig.add_subplot(gs[0, 0])
        ax_mask = fig.add_subplot(gs[0, 1])
        ax_line = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[1.6, 1],
            hspace=0.42,
            left=0.06, right=0.97,
            top=0.93, bottom=0.07,
        )
        ax_map  = fig.add_subplot(gs[0])
        ax_mask = None
        ax_line = fig.add_subplot(gs[1])

    fig.suptitle(
        f"{info.get('title', mode.upper())}   —   {period_label}{tree_tag}",
        fontsize=11, color="#ffffff", y=0.97,
    )

    # Pannello A: mappa indice
    ax_map.set_facecolor("#0a0a14")
    ax_map.axis("off")
    map_title = (
        f"Mappa {mode.upper()} — solo chiome arboree"
        if tree_focused else
        f"Mappa {mode.upper()} — periodo completo"
    )
    ax_map.set_title(map_title, fontsize=9, color="#a8d8ff", pad=5)

    if map_array is not None:
        # Normalizzazione robusta: usa percentile 2-98 sui pixel validi
        # per evitare che outlier schiaccino tutti i colori su un valore uniforme
        valid_px = map_array[~np.isnan(map_array)]
        if len(valid_px) > 0:
            p2  = float(np.percentile(valid_px, 2))
            p98 = float(np.percentile(valid_px, 98))
            # Se la varianza è troppo bassa forza un range minimo
            if (p98 - p2) < 0.01:
                mid = (p98 + p2) / 2
                p2, p98 = mid - 0.05, mid + 0.05
            # Clamp ai limiti teorici dell'indice
            disp_vmin = max(vmin, p2)
            disp_vmax = min(vmax, p98)
        else:
            disp_vmin, disp_vmax = vmin, vmax

        im = ax_map.imshow(map_array, cmap=cmap, vmin=disp_vmin, vmax=disp_vmax, aspect="auto")
        cbar = fig.colorbar(im, ax=ax_map, fraction=0.025, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color="#aaa")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccc", fontsize=7)
        # Aggiungi label range reale sotto la colorbar
        cbar.ax.text(
            0.5, -0.02, f"[{disp_vmin:.3f} – {disp_vmax:.3f}]",
            transform=cbar.ax.transAxes, ha="center", va="top",
            fontsize=6, color="#888899",
        )
        _attach_hover(fig, ax_map, map_array, label_prefix=mode.upper())

        if mean_full is not None:
            score   = normalize_to_score(mode, mean_full)
            sc_col  = ("#66dd66" if (score or 0) >= 70
                       else "#eecc44" if (score or 0) >= 45
                       else "#ee4444")
            score_str = f"  |  score {score:.0f}/100" if score is not None else ""
            ax_map.text(
                0.5, -0.025,
                f"Media {mean_label}: {mean_full:.4f}{score_str}"
                f"   —   {evaluate_index(mode, mean_full)}",
                transform=ax_map.transAxes, ha="center", va="top",
                fontsize=8.5, color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=sc_col, alpha=0.88, edgecolor="none"),
            )

        formula = info.get("formula", "")
        if formula:
            ax_map.text(
                0.01, 0.98, formula,
                transform=ax_map.transAxes, ha="left", va="top",
                fontsize=7, color="#ffe08a", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a3a",
                          edgecolor="#4455aa", lw=0.8, alpha=0.85),
            )
    else:
        ax_map.text(0.5, 0.5, "Nessuna immagine disponibile",
                    transform=ax_map.transAxes, ha="center", va="center",
                    fontsize=10, color="#888899")

    # Pannello B: mappa copertura chiome
    if ax_mask is not None and mask_array is not None:
        ax_mask.set_facecolor("#0a0a14")
        ax_mask.axis("off")
        total_px  = mask_array.size
        tree_px   = int(np.sum(mask_array))
        cover_pct = tree_px / total_px * 100.0
        ax_mask.set_title(
            f"Copertura chiome  ({cover_pct:.1f}% del raster)",
            fontsize=9, color="#88ffaa", pad=5,
        )
        mask_display = np.where(mask_array, 1.0, np.nan)
        bg = np.zeros_like(mask_array, dtype=float)
        ax_mask.imshow(bg, cmap="gray", vmin=0, vmax=1, aspect="auto", alpha=0.25)
        cmap_tree = LinearSegmentedColormap.from_list(
            "tree_cover", [(0.0, (0.1, 0.45, 0.1)), (1.0, (0.35, 0.95, 0.35))]
        )
        ax_mask.imshow(mask_display, cmap=cmap_tree, vmin=0, vmax=1,
                       aspect="auto", alpha=0.9)
        _attach_hover_mask(fig, ax_mask, mask_array)
        legend_elements = [
            Patch(facecolor="#44bb44", edgecolor="none", label="Chioma arborea"),
            Patch(facecolor="#333344", edgecolor="none", label="Non chioma"),
        ]
        ax_mask.legend(
            handles=legend_elements, loc="lower right",
            facecolor="#1a1a3a", edgecolor="#3355aa",
            labelcolor="#d0d0f0", fontsize=7,
        )
        ax_mask.text(
            0.5, -0.025,
            f"Pixel chiome: {tree_px:,} / {total_px:,}  ({cover_pct:.1f}%)",
            transform=ax_mask.transAxes, ha="center", va="top",
            fontsize=8, color="#bbffbb",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#0d2a0d", alpha=0.88, edgecolor="none"),
        )

    # Pannello C: grafico trend INTERATTIVO
    ax_line.set_facecolor("#0d0d20")
    ax_line.tick_params(colors="#888899", labelsize=8)
    ax_line.set_xlabel("Data", color="#888899", fontsize=8)
    trend_label = (
        f"Valore medio {mode.upper()}  (chiome arboree)"
        if tree_focused else
        f"Valore medio {mode.upper()}  (campo intero)"
    )
    ax_line.set_ylabel(trend_label, color="#888899", fontsize=8)
    for spine in ax_line.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax_line.grid(True, color="#1e1e3a", linewidth=0.6)
    ax_line.set_title(
        f"Andamento temporale — finestre di {step_days} giorni"
        + ("  [calcolato su chiome]" if tree_focused else "")
        + "   [🔍 zoom: scroll · pan: tasto centrale]",
        fontsize=9, color="#a8d8ff", pad=4,
    )

    vals_arr = np.array(values_pts, dtype=float)
    mask_ok  = ~np.isnan(vals_arr)
    dates_ok = [d for d, ok in zip(dates_pts, mask_ok) if ok]
    vals_ok  = vals_arr[mask_ok]

    if len(vals_ok) > 0:
        line_obj, = ax_line.plot(
            dates_ok, vals_ok,
            color="#5588ff", lw=2, marker="o", ms=6,
            zorder=3, label=f"{mode.upper()} {'chiome' if tree_focused else 'campo'}",
            picker=5,
        )
        ax_line.fill_between(dates_ok, vals_ok, alpha=0.10, color="#5588ff")

        if mean_full is not None:
            ax_line.axhline(
                mean_full, color="#eecc44", lw=1.2,
                linestyle="--", alpha=0.75,
                label=f"Media periodo ({mean_full:.4f})",
            )

        # Annotazioni statiche sui punti
        for d, v in zip(dates_ok, vals_ok):
            ax_line.annotate(
                f"{v:.3f}", xy=(d, v), xytext=(0, 9),
                textcoords="offset points",
                ha="center", fontsize=6.5, color="#ccccee",
            )

        ax_line.set_ylim(
            min(vals_ok) - abs(vmax - vmin) * 0.15,
            max(vals_ok) + abs(vmax - vmin) * 0.22,
        )
        ax_line.legend(
            facecolor="#1a1a3a", edgecolor="#3355aa",
            labelcolor="#d0d0f0", fontsize=8,
        )

        # ---- Tooltip interattivo con mplcursors ----
        try:
            import mplcursors
            cursor = mplcursors.cursor(line_obj, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                d   = dates_ok[idx]
                v   = vals_ok[idx]
                # Finestra temporale corrispondente
                win_end_dt = min(d + relativedelta(days=step_days), end_dt)
                sel.annotation.set_text(
                    f"📅 {d.strftime('%d/%m/%Y')} → {win_end_dt.strftime('%d/%m/%Y')}\n"
                    f"{mode.upper()}: {v:.5f}\n"
                    f"Score: {normalize_to_score(mode, v):.1f}/100"
                )
                sel.annotation.get_bbox_patch().set(
                    fc="#0d1020", ec="#4488ff", alpha=0.95,
                )
                sel.annotation.set_color("#e0e8ff")
                sel.annotation.set_fontsize(8.5)
        except ImportError:
            # mplcursors non installato: tooltip via motion_notify_event
            annot_line = ax_line.annotate(
                "", xy=(0, 0), xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc="#0d1020", ec="#4488ff", lw=0.9, alpha=0.94),
                fontsize=8, color="#e0e8ff",
                arrowprops=dict(arrowstyle="->", color="#4488ff", lw=0.7),
            )
            annot_line.set_visible(False)

            # Converti date in numeri matplotlib per il nearest-point
            import matplotlib.dates as mdates
            dates_num = mdates.date2num(dates_ok)

            def on_move_line(event):
                if event.inaxes != ax_line:
                    annot_line.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                if event.xdata is None:
                    return
                # Trova il punto più vicino sull'asse X
                diffs = np.abs(dates_num - event.xdata)
                idx   = int(np.argmin(diffs))
                # Mostra solo se abbastanza vicino (entro step_days/2 giorni)
                threshold = step_days / 2.0
                if diffs[idx] > threshold:
                    annot_line.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                d = dates_ok[idx]
                v = vals_ok[idx]
                win_end_dt = min(d + relativedelta(days=step_days), end_dt)
                annot_line.set_text(
                    f"📅 {d.strftime('%d/%m/%Y')} → {win_end_dt.strftime('%d/%m/%Y')}\n"
                    f"{mode.upper()}: {v:.5f}\n"
                    f"Score: {normalize_to_score(mode, v):.1f}/100"
                )
                annot_line.xy = (dates_ok[idx], vals_ok[idx])
                annot_line.set_visible(True)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", on_move_line)

        # ---- Zoom con scroll ----
        def on_scroll(event):
            if event.inaxes != ax_line:
                return
            factor = 0.85 if event.button == "up" else 1.15
            cur_xlim = ax_line.get_xlim()
            cur_ylim = ax_line.get_ylim()
            xdata, ydata = event.xdata, event.ydata
            new_xlim = [xdata + (x - xdata) * factor for x in cur_xlim]
            new_ylim = [ydata + (y - ydata) * factor for y in cur_ylim]
            ax_line.set_xlim(new_xlim)
            ax_line.set_ylim(new_ylim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", on_scroll)

        # ---- Pan con click sinistro ----
        _pan_state = {"active": False, "x0": None, "xlim0": None}

        def on_press_pan(event):
            if event.inaxes != ax_line or event.button != 1:
                return
            _pan_state["active"] = True
            _pan_state["x0"]    = event.xdata
            _pan_state["xlim0"] = ax_line.get_xlim()

        def on_release_pan(event):
            _pan_state["active"] = False

        def on_motion_pan(event):
            if not _pan_state["active"] or event.inaxes != ax_line:
                return
            if event.xdata is None:
                return
            dx = event.xdata - _pan_state["x0"]
            ax_line.set_xlim([x - dx for x in _pan_state["xlim0"]])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event",   on_press_pan)
        fig.canvas.mpl_connect("button_release_event", on_release_pan)
        fig.canvas.mpl_connect("motion_notify_event",  on_motion_pan)

    else:
        ax_line.text(
            0.5, 0.5, "Nessun dato temporale disponibile",
            transform=ax_line.transAxes, ha="center", va="center",
            fontsize=9, color="#888899",
        )

    fname = f"analisi_{mode}{'_tree' if tree_focused else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
    plt.show()
    print(f"  → Salvata: {fname}")

    return mean_full

# ================= VALUTAZIONE FINALE =================

def show_final_report(collected_means, period_label):
    available = {k: v for k, v in collected_means.items() if v is not None}
    if not available:
        print("Nessun dato disponibile per la valutazione finale.")
        return

    scores = {}
    for name, val in available.items():
        s = normalize_to_score(name, val)
        if s is not None:
            scores[name] = s

    total_weight = sum(INDEX_WEIGHTS.get(n, 0) for n in scores)
    global_score = (
        sum(scores[n] * INDEX_WEIGHTS.get(n, 0) for n in scores) / total_weight
        if total_weight > 0 else 0
    )

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#f5f5f0")
    fig.suptitle(
        f"RAPPORTO DI VALUTAZIONE FINALE  —  {period_label}",
        fontsize=14, fontweight="bold", color="#1a1a2e", y=0.98
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.35,
                          left=0.05, right=0.97, top=0.92, bottom=0.05)
    ax_tab  = fig.add_subplot(gs[0, 0])
    ax_bar  = fig.add_subplot(gs[0, 1])
    ax_conc = fig.add_subplot(gs[1, :])

    ax_tab.axis("off")
    ax_tab.set_title("Riepilogo per indice", fontsize=11, fontweight="bold", color="#1a1a2e")
    col_labels = ["Indice", "Valore medio", "Score /100", "Peso", "Valutazione"]
    rows = []
    for name in scores:
        val = available[name]
        rows.append([
            name.upper(),
            f"{val:.4f}",
            f"{scores[name]:.1f}",
            f"{INDEX_WEIGHTS.get(name, 0):.0%}",
            evaluate_index(name, val)
        ])

    table = ax_tab.table(cellText=rows, colLabels=col_labels,
                         loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#1a1a2e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, name in enumerate(scores, start=1):
        color = ("#c8f0c8" if scores[name] >= 70
                 else "#f0f0c0" if scores[name] >= 45
                 else "#f0c8c8")
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    ax_bar.set_facecolor("#fafaf5")
    names  = list(scores.keys())
    values = [scores[n] for n in names]
    colors = ["#2e8b57" if v >= 70 else "#e6b800" if v >= 45 else "#cc2200"
              for v in values]
    bars = ax_bar.barh(names, values, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_xlim(0, 100)
    ax_bar.axvline(global_score, color="#1a1a2e", linewidth=2,
                   linestyle="--", alpha=0.7)
    ax_bar.text(global_score + 1, len(names) - 0.3,
                f"Score globale: {global_score:.1f}",
                fontsize=8.5, color="#1a1a2e", fontweight="bold")
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8, color="#333333")
    ax_bar.set_xlabel("Score normalizzato (0–100)", fontsize=9)
    ax_bar.set_title("Score per indice", fontsize=11, fontweight="bold", color="#1a1a2e")
    ax_bar.invert_yaxis()
    ax_bar.spines[["top", "right"]].set_visible(False)

    ax_conc.axis("off")
    ax_conc.set_title("Conclusione correlata", fontsize=11, fontweight="bold", color="#1a1a2e")
    conclusion = build_conclusion(available, scores, global_score)
    ax_conc.text(
        0.01, 0.95, conclusion,
        transform=ax_conc.transAxes, ha="left", va="top",
        fontsize=9, color="#1a1a1a", wrap=True, multialignment="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef4ee",
                  edgecolor="#4a7a4a", linewidth=1.5)
    )

    plt.savefig("report_finale.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Report salvato in: report_finale.png")

# ================= CONCLUSIONE =================

def build_conclusion(available, scores, global_score):
    lines = []

    if global_score >= 75:
        stato, intro = "ECCELLENTE", "La vegetazione mostra uno stato di salute eccellente."
    elif global_score >= 60:
        stato, intro = "BUONO", "La vegetazione è in buono stato complessivo."
    elif global_score >= 45:
        stato, intro = "DISCRETO", "La vegetazione mostra un quadro discreto, con alcune aree di attenzione."
    elif global_score >= 30:
        stato, intro = "MEDIOCRE", "La vegetazione evidenzia stress diffusi che richiedono attenzione."
    else:
        stato, intro = "CRITICO", "Stato critico: stress severi e/o copertura molto scarsa."

    lines.append(f"► STATO GENERALE: {stato} (score ponderato: {global_score:.1f}/100)")
    lines.append(f"  {intro}\n")

    if "ndvi" in available and "savi" in available:
        ndvi_v, savi_v = available["ndvi"], available["savi"]
        lines.append(f"► COPERTURA VEGETALE: NDVI={ndvi_v:.3f}, SAVI={savi_v:.3f}.")
        if abs(ndvi_v - savi_v) > 0.1:
            lines.append("  Differenza NDVI/SAVI significativa → suolo esposto influente; riferirsi al SAVI.")
        else:
            lines.append("  NDVI e SAVI concordano: effetto suolo trascurabile.")
        lines.append("")
    elif "ndvi" in available:
        lines.append(
            f"► COPERTURA VEGETALE: NDVI={available['ndvi']:.3f}"
            f"  ({evaluate_index('ndvi', available['ndvi'])}).\n"
        )

    chloro = [n for n in ("ndre", "gndvi", "gci", "mcari", "tcari") if n in scores]
    if chloro:
        mean_c = np.mean([scores[n] for n in chloro])
        lines.append(f"► CLOROFILLA (media indici: {mean_c:.1f}/100):")
        for n in chloro:
            lines.append(f"  • {n.upper()}={available[n]:.3f}: {evaluate_index(n, available[n])}")
        if "mcari" in available and "tcari" in available:
            if abs(scores["mcari"] - scores["tcari"]) > 15:
                lines.append("  ⚠ Divergenza MCARI/TCARI: possibile stress clorofilliano precoce in atto.")
        lines.append("")

    if "pri" in available:
        pri_v = available["pri"]
        lines.append(f"► FOTOSINTESI (PRI={pri_v:.4f}): {evaluate_index('pri', pri_v)}.")
        if "ndvi" in available and available["ndvi"] > 0.5 and pri_v < -0.02:
            lines.append(
                "  ⚠ NDVI elevato + PRI negativo → vegetazione densa ma in stress fotosintetico attivo."
            )
        elif "ndvi" in available and available["ndvi"] > 0.5 and pri_v >= 0:
            lines.append(
                "  ✔ NDVI e PRI concordano: vegetazione densa e fotosinteticamente efficiente."
            )
        lines.append("")

    lines.append("► RACCOMANDAZIONI:")
    if global_score >= 70:
        lines.append("  Monitoraggio periodico standard. Nessuna azione urgente.")
    elif global_score >= 50:
        lines.append("  Verificare zone con score basso. Irrigazione/fertilizzazione azotata se GCI/NDRE bassi.")
    elif global_score >= 35:
        lines.append("  Intervenire con fertilizzazione e/o irrigazione. Ripetere acquisizione tra 2-3 settimane.")
    else:
        lines.append(
            "  Intervento agronomico immediato. Valutare cause strutturali (siccità, malattie, infestanti)."
        )

    return "\n".join(lines)


# ================= MAIN =================

def _show_rgb(img_bytes, title):
    with rasterio.open(img_bytes) as src:
        data = src.read().astype(np.float32)

    rgb = np.stack([data[0], data[1], data[2]], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.axis("off")
    ax.set_title(title, fontsize=10, color="#ffffff", pad=6)
    ax.imshow(rgb)

    annot = ax.annotate(
        "", xy=(0, 0),
        xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a3a", ec="#ff8844", lw=0.8, alpha=0.92),
        fontsize=7.5, color="#ffe8cc",
        arrowprops=dict(arrowstyle="->", color="#ff8844", lw=0.7),
    )
    annot.set_visible(False)

    def on_move_rgb(event):
        if event.inaxes != ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        col = int(round(event.xdata)) if event.xdata is not None else -1
        row = int(round(event.ydata)) if event.ydata is not None else -1
        h, w, _ = rgb.shape
        if 0 <= row < h and 0 <= col < w:
            r, g, b = rgb[row, col]
            annot.set_text(f"({col}, {row})\nR:{r}  G:{g}  B:{b}")
            annot.xy = (event.xdata, event.ydata)
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move_rgb)
    plt.tight_layout(pad=1.5)
    plt.savefig("analisi_rgb.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print("  → Salvata: analisi_rgb.png")


def run(client_id, client_secret, kml, start, end, mode, step_days,
        report_only=False, tree_focused=False, tree_params=None,
        data_file=None, chart_only=False, chart_output="chart.html"):

    if tree_params is None:
        tree_params = dict(TREE_MASK_DEFAULTS)

    period_label = f"{start} → {end}"
    if tree_focused:
        period_label += "  [chiome arboree]"

    # ---- Modalità chart-only ----
    if chart_only:
        if not data_file or not os.path.exists(data_file):
            print("⚠ --chart-only richiede --data-file con un CSV esistente.")
            return
        generate_html_chart(None, data_file, tree_focused, output_html=chart_output)
        return

    # ---- Lazy init API: autenticazione solo se serve ----
    _api = {"token": None, "polygon": None, "width": None, "height": None}

    def get_api():
        if _api["token"] is None:
            print("→ Autenticazione Copernicus...")
            _api["token"]                 = authenticate(client_id, client_secret)
            _api["polygon"]               = get_polygon(kml)
            _api["width"], _api["height"] = get_bbox_size(_api["polygon"])
        return _api["token"], _api["polygon"], _api["width"], _api["height"]

    # ---- Punti del trend attesi dati start/end/step ----
    def expected_trend_keys():
        start_dt = datetime.fromisoformat(start)
        end_dt   = datetime.fromisoformat(end)
        keys = []
        cur  = start_dt
        while cur <= end_dt:
            win_end = min(cur + relativedelta(days=step_days), end_dt)
            keys.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
            cur += relativedelta(days=step_days)
        return keys

    # ---- Legge il CSV per un indice: ritorna trend completo o None ----
    def read_from_csv(m):
        if not data_file or not os.path.exists(data_file):
            return None, None

        all_trend   = load_datapoints(data_file, tree_focused=tree_focused)
        trend_for_m = all_trend.get(m, [])

        if not trend_for_m:
            return None, None

        trend_sorted = sorted(trend_for_m, key=lambda p: p["start"])
        vals         = [p["value"] for p in trend_sorted]
        mean_val     = float(np.mean(vals)) if vals else None
        return mean_val, trend_sorted

    # ---- Grafico trend da CSV (senza mappa raster) ----
    def plot_trend_from_csv(m, mean_val, trend_points):
        info             = INDEX_DESCRIPTIONS.get(m, {})
        _, vmin, vmax    = get_colormap_and_range(m)
        tree_tag         = "  [chiome arboree]" if tree_focused else ""

        dates_pts = [datetime.fromisoformat(p["start"]) for p in trend_points]
        vals_arr  = np.array([p["value"] for p in trend_points], dtype=float)
        mask_ok   = ~np.isnan(vals_arr)
        dates_ok  = [d for d, ok in zip(dates_pts, mask_ok) if ok]
        vals_ok   = vals_arr[mask_ok]

        fig = plt.figure(figsize=(14, 6))
        fig.patch.set_facecolor("#0a0a14")
        fig.suptitle(
            f"{info.get('title', m.upper())}   —   {start} → {end}{tree_tag}"
            f"\n[dati da CSV]",
            fontsize=11, color="#ffffff", y=0.99,
        )

        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#0d0d20")
        ax.tick_params(colors="#888899", labelsize=8)
        ax.set_xlabel("Data", color="#888899", fontsize=8)
        ax.set_ylabel(
            f"Valore medio {m.upper()}  ({'chiome arboree' if tree_focused else 'campo intero'})",
            color="#888899", fontsize=8,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.grid(True, color="#1e1e3a", linewidth=0.6)
        ax.set_title(
            f"Andamento temporale — finestre di {step_days} giorni  [da CSV]",
            fontsize=9, color="#a8d8ff", pad=4,
        )

        if len(vals_ok) > 0:
            ax.plot(dates_ok, vals_ok,
                    color="#5588ff", lw=2, marker="o", ms=6, zorder=3,
                    label=f"{m.upper()} {'chiome' if tree_focused else 'campo'}")
            ax.fill_between(dates_ok, vals_ok, alpha=0.10, color="#5588ff")

            if mean_val is not None:
                ax.axhline(mean_val, color="#eecc44", lw=1.2,
                           linestyle="--", alpha=0.75,
                           label=f"Media periodo ({mean_val:.4f})")

            for d, v in zip(dates_ok, vals_ok):
                ax.annotate(f"{v:.3f}", xy=(d, v), xytext=(0, 9),
                            textcoords="offset points",
                            ha="center", fontsize=6.5, color="#ccccee")

            ax.set_ylim(
                min(vals_ok) - abs(vmax - vmin) * 0.15,
                max(vals_ok) + abs(vmax - vmin) * 0.22,
            )
            ax.legend(facecolor="#1a1a3a", edgecolor="#3355aa",
                      labelcolor="#d0d0f0", fontsize=8)

        if mean_val is not None:
            score     = normalize_to_score(m, mean_val)
            sc_col    = ("#66dd66" if (score or 0) >= 70
                         else "#eecc44" if (score or 0) >= 45
                         else "#ee4444")
            score_str = f"  |  score {score:.0f}/100" if score is not None else ""
            fig.text(
                0.5, 0.01,
                f"Media: {mean_val:.4f}{score_str}   —   {evaluate_index(m, mean_val)}",
                ha="center", fontsize=9, color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=sc_col, alpha=0.88, edgecolor="none"),
            )

        fname = f"analisi_{m}{'_tree' if tree_focused else ''}_csv.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0a0a14")
        plt.show()
        print(f"  → Salvata: {fname}")

    # ================================================================== #
    #  LOOP PRINCIPALE
    # ================================================================== #
    all_means = {}

    for m in mode:

        # RGB: sempre via API, niente da cacheare
        if m == "rgb":
            token, polygon, width, height = get_api()
            s_full = start + "Z" if not start.endswith("Z") else start
            e_full = end   + "Z" if not end.endswith("Z")   else end
            print(f"\n[RGB] Scarico immagine colori naturali {start} → {end} ...")
            url, headers, payload = make_request(
                token, polygon, width, height,
                s_full, e_full, "rgb", tree_focused=False,
            )
            img = download_image(url, headers, payload)
            if img:
                _show_rgb(img, f"RGB — {period_label}")
            continue

        print(f"\n[{m.upper()}] Controllo CSV...")
        mean_val, trend_points = read_from_csv(m)

        # ---- CASO 1: CSV completo → nessuna API ----
        if trend_points is not None:
            print(f"  → {len(trend_points)} punti trovati nel CSV. Nessuna chiamata API.")
            all_means[m] = mean_val
            if not report_only:
                plot_trend_from_csv(m, mean_val, trend_points)

        # ---- CASO 2: CSV incompleto o assente → scarica via API ----
        else:
            print(f"  → Dati non disponibili nel CSV. Scarico via API...")
            token, polygon, width, height = get_api()

            if report_only:
                s_full = start + "Z" if not start.endswith("Z") else start
                e_full = end   + "Z" if not end.endswith("Z")   else end
                url, headers, payload = make_request(
                    token, polygon, width, height,
                    s_full, e_full, m,
                    tree_focused=tree_focused,
                )
                img = download_image(url, headers, payload)
                _, _, mean_val, _ = extract_array_and_mean(img, m, tree_focused, tree_params)
                if mean_val is not None:
                    all_means[m] = mean_val
                    print(f"  → media: {mean_val:.4f}")
                    if data_file:
                        save_datapoint(data_file, start, end, tree_focused, m, mean_val)
            else:
                mean_val = show_index_unified(
                    token=token,
                    polygon=polygon,
                    width=width,
                    height=height,
                    mode=m,
                    start=start,
                    end=end,
                    step_days=step_days,
                    tree_focused=tree_focused,
                    tree_params=tree_params,
                    data_file=data_file,
                )
                if mean_val is not None:
                    all_means[m] = mean_val
                    if data_file:
                        save_datapoint(data_file, start, end, tree_focused, m, mean_val)

        time.sleep(0.2)

    if all_means:
        show_final_report(all_means, period_label)

    if data_file and os.path.exists(data_file):
        generate_html_chart(None, data_file, tree_focused, output_html=chart_output)

# ================= CLI =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisi satellite Sentinel-2 — mappa + copertura chiome + trend per indice."
    )
    parser.add_argument("--area",    required=False,
                        help="Percorso al file KML dell'area di interesse")
    parser.add_argument("--start",   required=False,
                        help="Data inizio periodo  YYYY-MM-DD")
    parser.add_argument("--end",     required=False,
                        help="Data fine periodo    YYYY-MM-DD")
    parser.add_argument("--mode",    required=False,
                        default="ndvi,ndre,gndvi,gci,savi,pri,mcari,tcari",
                        help="Indici separati da virgola oppure 'all'")
    parser.add_argument("--step",    required=False, type=int, default=10,
                        help="Ampiezza in giorni di ogni finestra temporale (default: 10)")
    parser.add_argument("--report-only", action="store_true",
                        help="Mostra solo il report finale senza figure per indice")
    parser.add_argument("--data-file", default=None,
                        help="File CSV per salvare/caricare le rilevazioni")
    parser.add_argument("--chart-only", action="store_true",
                        help="Genera solo il grafico HTML dal CSV esistente, senza chiamate API")
    parser.add_argument("--chart-output", default="chart.html",
                        help="Nome del file HTML del grafico (default: chart.html)")

    # ---- flag chiome arboree ----
    parser.add_argument("--tree-focus", action="store_true",
                        help="Abilita il filtro chiome arboree")
    parser.add_argument("--cvi-threshold",  type=float, default=1.5)
    parser.add_argument("--ndvi-threshold", type=float, default=0.25)
    parser.add_argument("--shadow-thresh",  type=float, default=0.08)

    args   = parser.parse_args()
    config = configparser.ConfigParser()
    config.read("../conf/conf.ini")

    modes = (
        ["rgb", "ndvi", "ndre", "gndvi", "gci", "savi", "pri", "mcari", "tcari"]
        if args.mode == "all"
        else [m.strip() for m in args.mode.split(",")]
    )

    if args.report_only and "rgb" in modes:
        modes = [m for m in modes if m != "rgb"]

    tree_params = {
        "cvi_threshold":    args.cvi_threshold,
        "ndvi_threshold":   args.ndvi_threshold,
        "shadow_threshold": args.shadow_thresh,
    }

    run(
        client_id     = config["AUTH"]["CLIENT_ID"],
        client_secret = config["AUTH"]["CLIENT_SECRET"],
        kml           = args.area,
        start         = args.start,
        end           = args.end,
        mode          = modes,
        step_days     = args.step,
        report_only   = args.report_only,
        tree_focused  = args.tree_focus,
        tree_params   = tree_params,
        data_file     = args.data_file,
        chart_only    = args.chart_only,
        chart_output  = args.chart_output,
    )