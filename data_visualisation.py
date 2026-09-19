"""
Data visualisation for the three ICS-SimLab captures (Intelligent Electronic
Device, Smart Grid, Water Bottle Factory). Two analyses, both on the full
(not downsampled) MODBUS rows of each dataset:

1. Temporal distribution - per-minute normal vs attack packet counts across
   each session, plus a 1-second zoom of the first 120 s (where the data flood
   attack sits). The raw CSVs have no date column, only a wall-clock "time"
   and the session-relative "frame_time_relative", so everything is placed in
   session seconds.
2. Detection-window experiment - the LLM pipeline computes packet_rate over a
   hardcoded k = 4 s window (extract_packet_info() in the local_multi_model_*
   scripts). For each candidate k, every packet gets a rolling rate
   [t - k, t] / k (inclusive window, same definition as extract_packet_info)
   and AUC-ROC measures how well that single number separates each attack type
   from normal traffic (1.0 = perfect, 0.5 = chance).

Output goes into data_visualisation/ (all filenames get the optional --tag
suffix so earlier results are not overwritten):
    overview.csv, timeline_per_minute.csv, burst_first_120s.csv,
    window_auc.csv, report.html
report.html is self-contained; only the Google Fonts stylesheet needs internet
(system fonts are used as the fallback). Run with a log, e.g.:
    python data_visualisation.py 2>&1 | tee data_visualisation/run_$(date +%Y%m%d_%H%M).log

The explanatory prose in the HTML describes these three captures (a flood
burst at the start of each session followed by sustained low-rate attacks).
The takeaway box under the heatmaps is computed from the results.
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv

# Created at import time so a shell log redirect works from the first run
# (same convention as the other scripts in this repo).
OUTPUT_DIR = Path("data_visualisation")
OUTPUT_DIR.mkdir(exist_ok=True)

ATTACKS = {
    1: "address scan",
    2: "function code scan",
    3: "device ID attack",
    4: "naive sensor read",
    5: "sporadic sensor injection",
    6: "force listen mode",
    7: "restart communication",
    8: "data flood attack",
}
FLOOD_CODE = 8

CANDIDATE_K = [0.5, 1, 2, 4, 8, 16, 32, 60]   # window sizes in seconds
CURRENT_K = 4                                  # value hardcoded in the pipeline
BUCKET_SEC = 60
BURST_WINDOW_SEC = 120
WEAK_AUC = 0.85          # "best AUC over all k stays below this" = weak signal
SMALL_SAMPLE = 50        # attack types with fewer rows are flagged as noisy

USECOLS = ["frame_time_relative", "attack_specific", "protocol"]


def load_modbus(dataset_name):
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES[dataset_name]), usecols=USECOLS)
    df = df[df["protocol"] == "MODBUS"].copy()
    # in the raw CSV normal traffic is encoded as NaN (or 0) in attack_specific
    df["is_attack"] = (df["attack_specific"].notna() & (df["attack_specific"] != 0)).astype(int)
    return df.sort_values("frame_time_relative").reset_index(drop=True)


def rolling_rate(t_sorted, k):
    """Packets per second in [t - k, t] for every packet (t_sorted ascending)."""
    left = np.searchsorted(t_sorted, t_sorted - k, side="left")
    right = np.searchsorted(t_sorted, t_sorted, side="right")
    return (right - left) / k


def analyse_dataset(dataset_name):
    df = load_modbus(dataset_name)
    t = df["frame_time_relative"].to_numpy()
    duration = float(t.max())

    n_buckets = int(duration // BUCKET_SEC) + 1
    df["minute"] = (df["frame_time_relative"] // BUCKET_SEC).astype(int)

    def per_minute(mask):
        return df[mask].groupby("minute").size().reindex(range(n_buckets), fill_value=0).tolist()

    is_flood = df["attack_specific"] == FLOOD_CODE
    normal_min = per_minute(df["is_attack"] == 0)
    flood_min = per_minute(is_flood)
    other_attack_min = per_minute((df["is_attack"] == 1) & ~is_flood)

    early = df[df["frame_time_relative"] <= BURST_WINDOW_SEC].copy()
    early["second"] = early["frame_time_relative"].astype(int)

    def per_second(mask):
        return early[mask].groupby("second").size().reindex(range(BURST_WINDOW_SEC + 1), fill_value=0).tolist()

    normal_1s = per_second(early["is_attack"] == 0)
    attack_1s = per_second(early["is_attack"] == 1)

    flood = df[is_flood]
    n_normal = int((df["is_attack"] == 0).sum())
    n_attack = int((df["is_attack"] == 1).sum())

    auc_matrix, auc_overall = [], []
    for k in CANDIDATE_K:
        rate = rolling_rate(t, k)
        auc_overall.append(round(float(roc_auc_score(df["is_attack"], rate)), 4))
        row = []
        for code in ATTACKS:
            mask = ((df["attack_specific"] == code) | (df["is_attack"] == 0)).to_numpy()
            y = df["is_attack"].to_numpy()[mask]
            row.append(round(float(roc_auc_score(y, rate[mask])), 4) if len(np.unique(y)) == 2 else None)
        auc_matrix.append(row)

    return {
        "session_duration_sec": duration,
        "n_normal": n_normal,
        "n_attack": n_attack,
        "attack_pct": round(n_attack / (n_normal + n_attack) * 100, 1),
        "bucket_sec": BUCKET_SEC,
        "normal_timeline": normal_min,
        "attack_timeline_excl_flood": other_attack_min,
        "flood_timeline": flood_min,
        "normal_1s": normal_1s,
        "attack_1s": attack_1s,
        "flood_window": [float(flood["frame_time_relative"].min()), float(flood["frame_time_relative"].max())],
        "flood_count": int(len(flood)),
        "flood_peak_per_sec": int(max(attack_1s)),
        "attack_counts": [int((df["attack_specific"] == c).sum()) for c in ATTACKS],
        "auc_matrix": auc_matrix,
        "auc_overall": auc_overall,
    }


def build_callout(datasets):
    labels = list(ATTACKS.values())
    i_cur = CANDIDATE_K.index(CURRENT_K)

    def mean_auc(matrix, ki):
        vals = [v for v in matrix[ki] if v is not None]
        return sum(vals) / len(vals)

    best_lines = []
    for name, d in datasets.items():
        means = [mean_auc(d["auc_matrix"], i) for i in range(len(CANDIDATE_K))]
        b = int(np.argmax(means))
        best_lines.append(
            f"<b>{name}</b>: best k = {CANDIDATE_K[b]}s (mean AUC {means[b]:.3f}); "
            f"k = {CURRENT_K}s gives {means[i_cur]:.3f}; k = {CANDIDATE_K[-1]}s gives {means[-1]:.3f}")

    weak, small = [], []
    for name, d in datasets.items():
        for ti, label in enumerate(labels):
            col = [row[ti] for row in d["auc_matrix"]]
            valid = [(v, i) for i, v in enumerate(col) if v is not None]
            n = d["attack_counts"][ti]
            if valid:
                best, bi = max(valid)
                if best < WEAK_AUC:
                    weak.append(f"{label} in {name} (best AUC {best:.3f} at k = {CANDIDATE_K[bi]}s, n = {n})")
            if 0 < n < SMALL_SAMPLE:
                small.append(f"{label} in {name} (n = {n})")

    html = ("<h3>Takeaway (computed from the results above)</h3>"
            "<p>Mean AUC across attack types, by window size:<br>" + "<br>".join(best_lines) + "</p>")
    if weak:
        html += ("<p>Attack types whose best AUC over every tested window stays below "
                 f"{WEAK_AUC}: " + "; ".join(weak) +
                 ". For these, packet rate alone is a weak signal at any window size.</p>")
    else:
        html += f"<p>Every attack type reaches an AUC of at least {WEAK_AUC} at some window size.</p>"
    if small:
        html += ("<p>Small samples, read their AUC with caution: " + "; ".join(small) + ".</p>")
    return html


def write_csvs(datasets, tag):
    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{tag}.{ext}" if tag else base)

    labels = list(ATTACKS.values())
    overview, timeline, burst, auc_rows = [], [], [], []
    for name, d in datasets.items():
        overview.append({
            "dataset": name, "session_duration_sec": round(d["session_duration_sec"], 1),
            "n_normal": d["n_normal"], "n_attack": d["n_attack"], "attack_pct": d["attack_pct"],
            "flood_count": d["flood_count"], "flood_start_sec": round(d["flood_window"][0], 1),
            "flood_end_sec": round(d["flood_window"][1], 1), "flood_peak_per_sec": d["flood_peak_per_sec"],
        })
        for i, (nm, ao, fl) in enumerate(zip(d["normal_timeline"], d["attack_timeline_excl_flood"],
                                             d["flood_timeline"])):
            timeline.append({"dataset": name, "minute": i, "normal": nm,
                             "attack_excluding_flood": ao, "flood": fl})
        for s, (nm, at) in enumerate(zip(d["normal_1s"], d["attack_1s"])):
            burst.append({"dataset": name, "second": s, "normal": nm, "attack": at})
        for ki, k in enumerate(CANDIDATE_K):
            auc_rows.append({"dataset": name, "k_sec": k, "attack_type": "ALL attacks",
                             "n_attack": d["n_attack"], "auc": d["auc_overall"][ki]})
            for ti, label in enumerate(labels):
                auc_rows.append({"dataset": name, "k_sec": k, "attack_type": label,
                                 "n_attack": d["attack_counts"][ti], "auc": d["auc_matrix"][ki][ti]})

    files = {"overview.csv": overview, "timeline_per_minute.csv": timeline,
             "burst_first_120s.csv": burst, "window_auc.csv": auc_rows}
    for base, rows in files.items():
        pd.DataFrame(rows).to_csv(path(base), index=False)
        print(f"Saved: {path(base)}")
    return path("report.html")


def render_html(payload, standalone=True):
    data_js = json.dumps(payload).replace("</", "<\\/")
    page = HTML_TEMPLATE.replace("__DASHBOARD_DATA__", data_js)
    if not standalone:
        return page
    head, body = page.split("</style>", 1)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<style>html, body { margin: 0; }</style>\n'
            + head + "</style>\n</head>\n<body>" + body + "\n</body>\n</html>\n")


def main():
    parser = argparse.ArgumentParser(description="Attack-timing distribution and detection-window experiment.")
    parser.add_argument("--tag", default=None,
                        help="Suffix added to every output filename so earlier results are kept.")
    args = parser.parse_args()

    start = time.time()
    datasets = {}
    for name in DATASET_FILENAMES:
        print("=" * 60)
        print(f"Dataset: {name}")
        t0 = time.time()
        datasets[name] = analyse_dataset(name)
        d = datasets[name]
        print(f"  session {d['session_duration_sec'] / 60:.1f} min | normal {d['n_normal']} | attack {d['n_attack']} "
              f"({d['attack_pct']}%) | flood {d['flood_count']} rows, peak {d['flood_peak_per_sec']} pkt/s")
        print("  mean AUC per window (k in s):",
              {k: round(sum(v for v in row if v is not None) / len([v for v in row if v is not None]), 3)
               for k, row in zip(CANDIDATE_K, d["auc_matrix"])})
        print(f"  dataset time: {time.time() - t0:.2f}s")

    print("=" * 60)
    callout = build_callout(datasets)
    report_path = write_csvs(datasets, args.tag)
    payload = {"candidate_k": CANDIDATE_K, "current_k": CURRENT_K,
               "attack_labels": list(ATTACKS.values()),
               "datasets": datasets, "callout_html": callout}
    report_path.write_text(render_html(payload, standalone=True), encoding="utf-8")
    print(f"Saved: {report_path}")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>ICS Attack Timing</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  .viz-root {
    color-scheme: light;
    --surface-0: #f9f9f7;
    --surface-1: #fcfcfb;
    --border: #e4e3de;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #8a8980;
    --series-normal: #2a78d6;
    --series-attack: #eb6834;
    --seq-100: #cde2fb; --seq-150: #b7d3f6; --seq-200: #9ec5f4; --seq-250: #86b6ef;
    --seq-300: #6da7ec; --seq-350: #5598e7; --seq-400: #3987e5; --seq-450: #2a78d6;
    --seq-500: #256abf; --seq-550: #1c5cab; --seq-600: #184f95; --seq-650: #104281;
    --seq-700: #0d366b;
    --shadow: 0 1px 2px rgba(20,20,15,0.06), 0 6px 20px rgba(20,20,15,0.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .viz-root {
      color-scheme: dark;
      --surface-0: #0d0d0d;
      --surface-1: #1a1a19;
      --border: #302f2b;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #7d7c74;
      --series-normal: #3987e5;
      --series-attack: #d95926;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-0: #0d0d0d;
    --surface-1: #1a1a19;
    --border: #302f2b;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #7d7c74;
    --series-normal: #3987e5;
    --series-attack: #d95926;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
  }

  .viz-root {
    min-height: 100vh;
    background: var(--surface-0);
    color: var(--text-primary);
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    padding: 32px 20px 64px;
    display: flex;
    flex-direction: column;
    gap: 40px;
  }
  .viz-root * { box-sizing: border-box; }
  .wrap { max-width: 1280px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; gap: 40px; }

  h1, h2, h3 { font-family: "Archivo", system-ui, sans-serif; text-wrap: balance; margin: 0; }
  h1 { font-size: clamp(28px, 4vw, 40px); font-weight: 800; letter-spacing: -0.01em; }
  h2 { font-size: 22px; font-weight: 700; }
  h3 { font-size: 15px; font-weight: 700; }
  .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .eyebrow {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-muted);
  }
  p { color: var(--text-secondary); line-height: 1.55; margin: 0; max-width: 68ch; }

  header.page-head { display: flex; flex-direction: column; gap: 10px; padding-top: 4px; }

  /* ---- stat tiles ---- */
  .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  @media (max-width: 800px) { .stat-grid { grid-template-columns: 1fr; } }
  .stat-card {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; display: flex; flex-direction: column; gap: 10px; box-shadow: var(--shadow);
  }
  .stat-card h3 { color: var(--text-primary); }
  .stat-row { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .stat-label { font-size: 12.5px; color: var(--text-muted); }
  .stat-value { font-family: "IBM Plex Mono", monospace; font-size: 14px; font-variant-numeric: tabular-nums; }
  .bar-split { height: 8px; border-radius: 4px; overflow: hidden; display: flex; background: var(--border); }
  .bar-split .normal { background: var(--series-normal); }
  .bar-split .attack { background: var(--series-attack); }
  .legend-row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-secondary); }
  .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }

  /* ---- section shells ---- */
  section { display: flex; flex-direction: column; gap: 16px; }
  .section-head { display: flex; flex-direction: column; gap: 6px; }

  .panel-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  @media (max-width: 980px) { .panel-grid { grid-template-columns: 1fr; } }
  .dataset-block {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 16px 14px; display: flex; flex-direction: column; gap: 12px; box-shadow: var(--shadow);
  }
  .chart-title { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 2px 8px; }
  .chart-title .name { font-weight: 700; font-size: 13.5px; }
  .chart-title .meta { font-size: 11.5px; color: var(--text-muted); font-family: "IBM Plex Mono", monospace; }

  .chart-box { position: relative; width: 100%; }
  .chart-box svg { display: block; width: 100%; height: auto; overflow: visible; }
  .axis-label { font-size: 9.5px; fill: var(--text-muted); font-family: "IBM Plex Mono", monospace; }
  .gridline { stroke: var(--border); stroke-width: 1; }
  .crosshair { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2 2; opacity: 0; pointer-events: none; }

  .tooltip {
    position: absolute; pointer-events: none; z-index: 5;
    background: var(--text-primary); color: var(--surface-1);
    font-family: "IBM Plex Mono", monospace; font-size: 11px; line-height: 1.5;
    padding: 6px 9px; border-radius: 6px; white-space: nowrap; opacity: 0;
    transform: translate(-50%, -100%); transition: opacity 0.08s ease;
    box-shadow: var(--shadow);
  }
  .tooltip b { font-weight: 700; }

  .callout {
    background: var(--surface-1); border: 1px solid var(--border); border-left: 3px solid var(--series-normal);
    border-radius: 8px; padding: 14px 18px; display: flex; flex-direction: column; gap: 6px;
  }
  .callout p { max-width: none; }
  .callout .k4 { color: var(--series-attack); font-weight: 700; }

  /* ---- heatmap ---- */
  .heatmap-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .heatmap-grid > .dataset-block { min-width: 0; overflow-x: auto; }
  @media (max-width: 980px) { .heatmap-grid { grid-template-columns: 1fr; } }
  table.heatmap { border-collapse: collapse; width: 100%; font-size: 11px; }
  table.heatmap th, table.heatmap td { text-align: center; padding: 0; }
  table.heatmap th { font-family: "IBM Plex Mono", monospace; font-weight: 600; color: var(--text-muted); font-size: 10.5px; padding-bottom: 4px; }
  table.heatmap th.row-label-head { text-align: left; }
  table.heatmap td.row-label {
    text-align: left; font-size: 11px; color: var(--text-secondary); padding: 3px 8px 3px 0;
    white-space: nowrap; max-width: 100px; overflow: hidden; text-overflow: ellipsis;
  }
  .heat-cell {
    height: 24px; border-radius: 3px; border: 2px solid var(--surface-1);
    font-family: "IBM Plex Mono", monospace; color: white; font-size: 10px;
    display: flex; align-items: center; justify-content: center; cursor: default;
  }
  .n-head, .n-cell { font-family: "IBM Plex Mono", monospace; font-size: 10px; color: var(--text-muted); padding-left: 8px; text-align: right; }
  .k4-col { outline: 2px solid var(--series-attack); outline-offset: -2px; border-radius: 3px; }

  .table-toggle {
    font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--text-secondary);
    background: none; border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer;
  }
  .table-toggle:hover { border-color: var(--text-muted); }

  footer { border-top: 1px solid var(--border); padding-top: 20px; }
  footer p { font-size: 12.5px; }
</style>

<div class="viz-root">
<div class="wrap">

  <header class="page-head">
    <span class="eyebrow">ICS-SimLab &middot; Modbus/TCP capture analysis</span>
    <h1>When do attacks actually happen?</h1>
    <p>Three ICS-SimLab captures &mdash; Intelligent Electronic Device, Smart Grid, Water Bottle Factory &mdash; each a single continuous session with no date field in the raw data, only a wall-clock <span class="mono">time</span> and a session-relative <span class="mono">frame_time_relative</span>. This maps where normal and attack traffic actually fall inside each session, then tests whether the packet-rate detection window (hardcoded at k&nbsp;=&nbsp;4s across the pipeline) is well-chosen.</p>
  </header>

  <section id="overview">
    <div class="stat-grid" id="stat-grid"></div>
  </section>

  <section id="timing">
    <div class="section-head">
      <span class="eyebrow">Part 1 &middot; Temporal distribution</span>
      <h2>Two very different attack rhythms</h2>
      <p>Every session opens with an extreme, ~7-second packet flood (<b>data flood attack</b>) that outweighs everything else by two orders of magnitude &mdash; shown zoomed below. The other seven attack types are spread thin across the <em>entire</em> remaining session, interleaved with normal traffic rather than clustered.</p>
    </div>
    <div class="panel-grid" id="burst-grid"></div>
    <div class="panel-grid" id="sustained-grid"></div>
  </section>

  <section id="window">
    <div class="section-head">
      <span class="eyebrow">Part 2 &middot; Detection window experiment</span>
      <h2>Is a 4-second rate window the right choice?</h2>
      <p><span class="mono">extract_packet_info()</span> computes <span class="mono">packet_rate</span> from a hardcoded <span class="mono">k&nbsp;=&nbsp;4</span> second window, with no comment or evaluation anywhere in the notebook or scripts justifying that value. To check it empirically: for each candidate <span class="mono">k</span>, every packet in the full (unsampled) dataset gets a rolling rate &mdash; count of packets in [t&minus;k,&nbsp;t] &divide; k, computed via a sorted-array two-pointer search, not a heuristic &mdash; then <b>AUC-ROC</b> scores how well that single number separates each attack type from normal. 1.0 = perfect separation, 0.5 = no better than chance.</p>
    </div>
    <div class="heatmap-grid" id="heatmap-grid"></div>
    <div class="callout" id="window-callout"></div>
  </section>

  <footer>
    <p>Methodology: MODBUS-protocol rows only, full dataset (no downsampling) for both the timing and AUC analyses. Rolling packet-rate uses an inclusive window <span class="mono">[t&minus;k, t]</span> matching <span class="mono">extract_packet_info()</span>'s own definition. The <span class="k4" style="color:var(--series-attack)">outlined column</span> in each heatmap marks the pipeline's current <span class="mono">k=4</span>.</p>
  </footer>

</div>
</div>

<script>
(function () {
  const DATA = __DASHBOARD_DATA__;

  const isDark = () => {
    const root = document.documentElement;
    if (root.getAttribute('data-theme') === 'dark') return true;
    if (root.getAttribute('data-theme') === 'light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  };

  function css(varName, el) {
    return getComputedStyle(el || document.querySelector('.viz-root')).getPropertyValue(varName).trim();
  }

  function fmt(n) {
    if (n >= 1000) return (n/1000).toFixed(1).replace(/\.0$/,'') + 'k';
    return String(n);
  }

  function secToClock(s) {
    const m = Math.floor(s / 60), sec = Math.round(s % 60);
    return `${m}m${String(sec).padStart(2,'0')}s`;
  }

  // ---------------- stat tiles ----------------
  const statGrid = document.getElementById('stat-grid');
  Object.entries(DATA.datasets).forEach(([name, d]) => {
    const card = document.createElement('div');
    card.className = 'stat-card';
    const durMin = (d.session_duration_sec/60).toFixed(1);
    card.innerHTML = `
      <h3>${name}</h3>
      <div class="stat-row"><span class="stat-label">Session length</span><span class="stat-value">${durMin} min</span></div>
      <div class="stat-row"><span class="stat-label">Normal / Attack rows</span><span class="stat-value">${fmt(d.n_normal)} / ${fmt(d.n_attack)}</span></div>
      <div class="bar-split"><div class="normal" style="width:${100-d.attack_pct}%"></div><div class="attack" style="width:${d.attack_pct}%"></div></div>
      <div class="stat-row"><span class="stat-label">Attack share</span><span class="stat-value">${d.attack_pct}%</span></div>
      <div class="stat-row"><span class="stat-label">Flood burst peak</span><span class="stat-value">${fmt(d.flood_peak_per_sec)} pkt/s</span></div>
    `;
    statGrid.appendChild(card);
  });

  // ---------------- burst-zoom panels (0-120s, 1s buckets) ----------------
  // Uses drawScaledChart (defined below) with a power-scale y-axis so the
  // ~12,500 pkt/s flood spike and the near-zero normal baseline both stay
  // legible on one chart, while tooltips still show the real, unscaled counts.
  const burstGrid = document.getElementById('burst-grid');
  Object.entries(DATA.datasets).forEach(([name, d]) => {
    const block = document.createElement('div');
    block.className = 'dataset-block';
    block.innerHTML = `<div class="chart-title"><span class="name">${name}</span><span class="meta">0&ndash;120s &middot; 1s buckets &middot; power-scaled y</span></div>
      <div class="chart-box"></div>
      <div class="legend-row">
        <div class="legend-item"><span class="swatch" style="background:var(--series-normal)"></span>Normal</div>
        <div class="legend-item"><span class="swatch" style="background:var(--series-attack)"></span>Attack (flood)</div>
      </div>`;
    burstGrid.appendChild(block);
    drawScaledChart(block.querySelector('.chart-box'), {
      seriesRaw: [
        {label:'Normal', values: d.normal_1s, color: css('--series-normal')},
        {label:'Attack', values: d.attack_1s, color: css('--series-attack')},
      ],
      scale: v => Math.pow(v, 0.4),
      xTickFormat: (i, full) => full ? `t = ${i}s` : `${i}s`,
      yTickFormat: v => fmt(Math.round(v)),
    });
  });

  function drawScaledChart(container, {seriesRaw, scale, width=360, height=150, xTickFormat, yTickFormat}) {
    const pad = {top: 10, right: 8, bottom: 20, left: 36};
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const n = seriesRaw[0].values.length;
    const rawMax = Math.max(1, ...seriesRaw.flatMap(s => s.values));
    const scaledMax = scale(rawMax) * 1.12;

    const x = i => pad.left + (i/(n-1)) * innerW;
    const y = raw => pad.top + innerH - (scale(raw)/scaledMax) * innerH;

    const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

    [0, 0.25, 0.55, 1].forEach(frac => {
      const rawTickVal = Math.pow(frac*scaledMax, 1/0.4);
      const gy = pad.top + innerH - frac*innerH;
      const line = document.createElementNS(svg.namespaceURI,'line');
      line.setAttribute('x1', pad.left); line.setAttribute('x2', width-pad.right);
      line.setAttribute('y1', gy); line.setAttribute('y2', gy);
      line.setAttribute('class','gridline');
      svg.appendChild(line);
      const lbl = document.createElementNS(svg.namespaceURI,'text');
      lbl.setAttribute('x', pad.left-5); lbl.setAttribute('y', gy+3);
      lbl.setAttribute('text-anchor','end'); lbl.setAttribute('class','axis-label');
      lbl.textContent = yTickFormat(rawTickVal);
      svg.appendChild(lbl);
    });

    [0, Math.floor((n-1)/2), n-1].forEach(i => {
      const lbl = document.createElementNS(svg.namespaceURI,'text');
      lbl.setAttribute('x', x(i)); lbl.setAttribute('y', height-4);
      lbl.setAttribute('text-anchor', i===0 ? 'start' : (i===n-1 ? 'end' : 'middle'));
      lbl.setAttribute('class','axis-label');
      lbl.textContent = xTickFormat(i);
      svg.appendChild(lbl);
    });

    seriesRaw.forEach(s => {
      let path = `M ${x(0)} ${y(s.values[0])}`;
      for (let i=1;i<n;i++) path += ` L ${x(i)} ${y(s.values[i])}`;
      const areaPath = path + ` L ${x(n-1)} ${pad.top+innerH} L ${x(0)} ${pad.top+innerH} Z`;
      const area = document.createElementNS(svg.namespaceURI,'path');
      area.setAttribute('d', areaPath); area.setAttribute('fill', s.color); area.setAttribute('opacity','0.15');
      svg.appendChild(area);
      const line = document.createElementNS(svg.namespaceURI,'path');
      line.setAttribute('d', path); line.setAttribute('fill','none');
      line.setAttribute('stroke', s.color); line.setAttribute('stroke-width','1.6');
      svg.appendChild(line);
    });

    const crosshair = document.createElementNS(svg.namespaceURI,'line');
    crosshair.setAttribute('y1', pad.top); crosshair.setAttribute('y2', pad.top+innerH);
    crosshair.setAttribute('class','crosshair');
    svg.appendChild(crosshair);
    const hit = document.createElementNS(svg.namespaceURI,'rect');
    hit.setAttribute('x', pad.left); hit.setAttribute('y', pad.top);
    hit.setAttribute('width', innerW); hit.setAttribute('height', innerH);
    hit.setAttribute('fill','transparent');
    svg.appendChild(hit);

    container.style.position = 'relative';
    container.appendChild(svg);
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    container.appendChild(tooltip);

    hit.addEventListener('mousemove', (e) => {
      const rect = svg.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width * width;
      let i = Math.round(((relX - pad.left) / innerW) * (n-1));
      i = Math.max(0, Math.min(n-1, i));
      crosshair.setAttribute('x1', x(i)); crosshair.setAttribute('x2', x(i));
      crosshair.style.opacity = 1;
      const lines = seriesRaw.map(s => `<b style="color:${s.color}">${s.label}</b> ${fmt(s.values[i])}/s`).join('<br>');
      tooltip.innerHTML = `${xTickFormat(i, true)}<br>${lines}`;
      tooltip.style.opacity = 1;
      const containerRect = container.getBoundingClientRect();
      tooltip.style.left = (rect.left - containerRect.left + x(i)) + 'px';
      tooltip.style.top = (rect.top - containerRect.top + Math.min(...seriesRaw.map(s=>y(s.values[i])))) - 8 + 'px';
    });
    hit.addEventListener('mouseleave', () => { crosshair.style.opacity = 0; tooltip.style.opacity = 0; });
  }

  // ---------------- sustained-activity panels (full session, flood excluded, 60s buckets) ----------------
  const sustainedGrid = document.getElementById('sustained-grid');
  Object.entries(DATA.datasets).forEach(([name, d]) => {
    const block = document.createElement('div');
    block.className = 'dataset-block';
    const durMin = Math.round(d.session_duration_sec/60);
    block.innerHTML = `<div class="chart-title"><span class="name">${name}</span><span class="meta">0&ndash;${durMin}m &middot; 60s buckets &middot; flood excluded &middot; power-scaled y</span></div>
      <div class="chart-box"></div>
      <div class="legend-row">
        <div class="legend-item"><span class="swatch" style="background:var(--series-normal)"></span>Normal</div>
        <div class="legend-item"><span class="swatch" style="background:var(--series-attack)"></span>Attack (7 other types)</div>
      </div>`;
    sustainedGrid.appendChild(block);
    drawScaledChart(block.querySelector('.chart-box'), {
      seriesRaw: [
        {label:'Normal', values: d.normal_timeline, color: css('--series-normal')},
        {label:'Attack', values: d.attack_timeline_excl_flood, color: css('--series-attack')},
      ],
      scale: v => Math.pow(v, 0.7),
      xTickFormat: (i, full) => full ? secToClock(i*60) : `${Math.round(i*60/60)}m`,
      yTickFormat: v => fmt(Math.round(v)),
    });
  });

  // ---------------- window-size heatmaps ----------------
  const heatGrid = document.getElementById('heatmap-grid');
  const seqRamp = ['--seq-100','--seq-200','--seq-300','--seq-400','--seq-500','--seq-600','--seq-700'];
  function seqText(auc) {
    const t = Math.max(0, Math.min(1, (auc - 0.5) / 0.5));
    return Math.floor(t * seqRamp.length) <= 2 ? '#0b0b0b' : '#ffffff';
  }
  function seqColor(auc) {
    // map AUC 0.5 (chance) -> 1.0 (perfect) onto the 7-step ramp
    const t = Math.max(0, Math.min(1, (auc - 0.5) / 0.5));
    const idx = Math.min(seqRamp.length-1, Math.floor(t * seqRamp.length));
    return css(seqRamp[idx]);
  }

  const k4Index = DATA.candidate_k.indexOf(DATA.current_k);

  Object.entries(DATA.datasets).forEach(([name, d]) => {
    const wrap = document.createElement('div');
    wrap.className = 'dataset-block';
    const table = document.createElement('table');
    table.className = 'heatmap';
    let thead = '<tr><th class="row-label-head">Attack type</th>' +
      DATA.candidate_k.map((k,ci) => `<th class="${ci===k4Index?'k4-col':''}">${k}s</th>`).join('') + '<th class="n-head">n</th></tr>';
    let rows = '';
    DATA.attack_labels.forEach((label, ri) => {
      rows += `<tr><td class="row-label">${label}</td>`;
      DATA.candidate_k.forEach((k, ci) => {
        const auc = d.auc_matrix[ci][ri];
        const cellClass = ci===k4Index ? 'k4-col' : '';
        if (auc === null) {
          rows += `<td class="${cellClass}"><div class="heat-cell" style="background:var(--border)">&mdash;</div></td>`;
        } else {
          rows += `<td class="${cellClass}"><div class="heat-cell" style="background:${seqColor(auc)};color:${seqText(auc)}" title="${label} @ k=${k}s: AUC ${auc.toFixed(3)}">${auc.toFixed(2)}</div></td>`;
        }
      });
      rows += `<td class="n-cell">${d.attack_counts[ri]}</td></tr>`;
    });
    table.innerHTML = thead + rows;
    wrap.innerHTML = `<h3>${name}</h3>`;
    wrap.appendChild(table);
    heatGrid.appendChild(wrap);
  });

  // callout takeaway
  document.getElementById('window-callout').innerHTML = DATA.callout_html;

})();
</script>
"""

if __name__ == "__main__":
    main()
