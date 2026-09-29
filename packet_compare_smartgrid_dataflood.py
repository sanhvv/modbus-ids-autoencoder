"""
Ninth and last in the per-attack-type series for Smart Grid (after
function-code scan, address scan, device identification attack, naive
sensor read, sporadic sensor measurement injection, force listen mode,
restart communication - see the Smart Grid normal-behavior-baseline
memory), applied to "data flood attack" (attack_specific == 8).

This is the same ~7-second flood burst already found session-wide, across
all 3 datasets, by data_visualisation.py (XAI-15): a single burst at
t~47-54s, peak 12.1-12.5k pkt/s. This script complements that analysis
with packet-level content and Smart Grid-specific detection signals that
data_visualisation.py's session-wide view does not cover.

Unlike every other attack in this series, this attack's request TYPE
(reading a value) DOES have a normal-traffic equivalent, so
extract_normal_pair() from packet_compare_smartgrid_fcscan.py is reused -
same reasoning as naive-sensor-read. The difference from normal is volume
and randomization, not the request category.

Structure, verified from the raw data: t=46.90s-53.92s (7.02s span),
42,465 requests (~6,052 req/s - ~1,528x normal's ~3.96 req/s), spread
across only 10 TCP connections (~4,246 requests per connection on
average). Unlike naive-sensor-read's fixed address-0/max-quantity pattern,
THIS attack randomizes both parameters: address spans the full 0-100
range (101 distinct values) and quantity spans 1-100 (100 distinct
values) - 17,735 distinct request PDUs out of 42,465 requests. All 4
standard read function codes (fc1-4) used in roughly equal proportion,
no writes. The request rate is remarkably constant throughout the burst
(~3,000 requests per 0.5s bin throughout, not ramping up/down).

Also verified: unlike naive-sensor-read/sporadic-injection/force-listen/
restart-comm, this attack has ZERO background-noise co-mingling (0.0% of
its 85,010 rows are unrelated traffic) - the flood is so overwhelming
within its narrow 7-second window that there is no room for incidental
background chatter, unlike the longer-duration, lower-intensity attacks
earlier in this series.

Output goes into data_visualisation/smartgrid_data_flood/ (all filenames
get the optional --tag suffix so earlier results are not overwritten):
packets.json, stats.json, timeline.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_dataflood.py 2>&1 | tee data_visualisation/smartgrid_data_flood/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv
from packet_compare_smartgrid_fcscan import extract_normal_pair, pack

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_data_flood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
TARGET_IP = "192.168.0.31"


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def decode_qty(hexdata):
    s = str(hexdata)[2:] if str(hexdata).startswith("0x") else str(hexdata)
    if len(s) < 5:
        return None, None
    try:
        return int(s[:4], 16), int(s[4:], 16)
    except ValueError:
        return None, None


def extract_attack_examples(df):
    a8 = df[df.attack_specific == 8].sort_values("row")
    mb = a8[a8.protocol == "MODBUS"].sort_values("frame_time_relative").reset_index(drop=True)

    def find_pair_at(idx_from):
        for i in range(idx_from, len(mb) - 1):
            if mb.iloc[i].ip_src == ATTACKER_IP and mb.iloc[i + 1].ip_src == TARGET_IP:
                return {"request": pack(mb.iloc[i]), "response": pack(mb.iloc[i + 1])}
        return None

    early_example = find_pair_at(0)
    mid_example = find_pair_at(len(mb) // 2)

    return {"early_example": early_example, "mid_example": mid_example}


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a8 = df[df.attack_specific == 8]
    mb = a8[a8.protocol == "MODBUS"]
    req = mb[mb.ip_src == ATTACKER_IP]
    resp = mb[mb.ip_src == TARGET_IP]
    dur = df.frame_time_relative.max()

    addrs, qtys = [], []
    for d in req.modbus_data.dropna():
        a, q = decode_qty(d)
        if a is not None:
            addrs.append(a)
        if q is not None:
            qtys.append(q)
    addrs = np.array(addrs)
    qtys = np.array(qtys)

    t = req.sort_values("frame_time_relative").frame_time_relative.to_numpy()
    attack_start, attack_end = float(t.min()), float(t.max())
    attack_dur = attack_end - attack_start

    noise = a8[a8.protocol.isin(["DATA", "WEBSOCKET", "HTTP"])
              & a8.ip_src.isin([ATTACKER_IP, "192.168.0.111"]) & a8.ip_dst.isin([ATTACKER_IP, "192.168.0.111"])]

    normal_req = n[(n.protocol == "MODBUS") & (n.ip_src == "192.168.0.40")]
    normal_rate = round(len(normal_req) / dur, 2)
    normal_max_resp = int(n[(n.protocol == "MODBUS") & (n.tcp_len != 12)].ip_len.max())
    normal_addrs = set()
    for d in n[(n.protocol == "MODBUS") & (n.tcp_len == 12)].modbus_data:
        a, _ = decode_qty(d)
        if a is not None:
            normal_addrs.add(a)

    fc_counts = req.modbus_func_code.value_counts().to_dict()

    return {
        "session_duration_sec": round(dur, 1),
        "attack_total_rows": int(len(a8)),
        "attack_noise_rows": int(len(noise)),
        "attack_noise_pct": round(len(noise) / len(a8) * 100, 1) if len(a8) else 0.0,
        "n_requests": int(len(req)),
        "n_responses": int(len(resp)),
        "attack_window_start_sec": round(attack_start, 1),
        "attack_window_end_sec": round(attack_end, 1),
        "attack_duration_sec": round(attack_dur, 2),
        "attack_rate_per_sec": round(len(req) / attack_dur, 1),
        "normal_rate_per_sec": normal_rate,
        "rate_ratio": round((len(req) / attack_dur) / normal_rate, 1) if normal_rate else None,
        "n_tcp_connections": int(req.tcp_stream.nunique()),
        "requests_per_connection": round(len(req) / req.tcp_stream.nunique(), 1),
        "address_min": int(addrs.min()) if len(addrs) else None,
        "address_max": int(addrs.max()) if len(addrs) else None,
        "address_distinct": int(len(set(addrs))),
        "quantity_min": int(qtys.min()) if len(qtys) else None,
        "quantity_max": int(qtys.max()) if len(qtys) else None,
        "quantity_distinct": int(len(set(qtys))),
        "distinct_pdus": int(req.modbus_data.nunique()),
        "normal_addresses": sorted(normal_addrs),
        "normal_max_qty": 1,
        "normal_max_resp_bytes": normal_max_resp,
        "attack_max_resp_bytes": int(resp.ip_len.max()),
        "function_codes_used": {str(int(k)): int(v) for k, v in fc_counts.items()},
    }


def compute_timeline(df, bin_width=10.0):
    """Per-bin Modbus request counts across the whole session, normal
    (master -> RTU) vs. this attack (attacker -> RTU) - same treatment as
    packet_compare_smartgrid_naivesensorread.py. Unlike every other attack
    in this series, this burst is so short (~7s) it mostly falls within a
    single bin.
    """
    dur = float(df.frame_time_relative.max())
    edges = np.arange(0, dur + bin_width, bin_width)
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a8 = df[df.attack_specific == 8]

    normal_times = n[(n.protocol == "MODBUS") & (n.ip_src == "192.168.0.40")].frame_time_relative.to_numpy()
    attack_times = a8[(a8.protocol == "MODBUS") & (a8.ip_src == ATTACKER_IP)].frame_time_relative.to_numpy()

    normal_counts, _ = np.histogram(normal_times, bins=edges)
    attack_counts, _ = np.histogram(attack_times, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    return {
        "bin_width_sec": bin_width,
        "session_duration_sec": round(dur, 1),
        "bin_centers": [round(float(x), 1) for x in bin_centers],
        "normal_counts": [int(x) for x in normal_counts],
        "attack_counts": [int(x) for x in attack_counts],
        "normal_avg_per_bin": round(float(normal_counts.mean()), 1),
        "attack_max_per_bin": int(attack_counts.max()),
        "attack_active_bin_pct": round(float((attack_counts > 0).mean() * 100), 1),
    }


def render_html(payload):
    data_js = json.dumps(payload).replace("</", "<\\/")
    body = HTML_TEMPLATE.replace("__PACKET_DATA__", data_js)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<style>html, body { margin: 0; }</style>\n'
            + body.split("</style>", 1)[0] + "</style>\n</head>\n<body>"
            + body.split("</style>", 1)[1] + "\n</body>\n</html>\n")


def main():
    parser = argparse.ArgumentParser(description="Compare a normal exchange vs. the data-flood attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting normal packet pair (reused from packet_compare_smartgrid_fcscan)...")
    normal_pair = extract_normal_pair(df)
    print("Extracting data-flood examples (early and mid-burst requests)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)
    print("Computing request-rate time series (normal vs. attack, whole session)...")
    timeline = compute_timeline(df)

    packets = {"normal_pair": normal_pair, "attack_examples": attack_examples}
    with open(path("packets.json"), "w") as f:
        json.dump(packets, f, indent=1)
    print(f"Saved: {path('packets.json')}")

    with open(path("stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"Saved: {path('stats.json')}")

    with open(path("timeline.json"), "w") as f:
        json.dump(timeline, f, indent=1)
    print(f"Saved: {path('timeline.json')}")

    payload = {**packets, "stats": stats, "timeline": timeline}
    report_path = path("report.html")
    report_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"Data-flood summary: {stats['n_requests']} requests in {stats['attack_duration_sec']}s "
          f"(~{stats['attack_rate_per_sec']}/s, {stats['rate_ratio']}x normal), addresses "
          f"{stats['address_min']}-{stats['address_max']} and quantities {stats['quantity_min']}-"
          f"{stats['quantity_max']} randomized, {stats['n_tcp_connections']} TCP connections, "
          f"{stats['attack_noise_pct']}% background noise")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Data Flood Diff</title>
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
    --normal: #2a78d6;
    --attack: #eb6834;
    --normal-bg: #eaf2fc;
    --attack-bg: #fdece2;
    --shadow: 0 1px 2px rgba(20,20,15,0.06), 0 6px 20px rgba(20,20,15,0.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .viz-root {
      color-scheme: dark;
      --surface-0: #0d0d0d; --surface-1: #1a1a19; --border: #302f2b;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #7d7c74;
      --normal: #3987e5; --attack: #d95926;
      --normal-bg: #13253d; --attack-bg: #3a2116;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-0: #0d0d0d; --surface-1: #1a1a19; --border: #302f2b;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #7d7c74;
    --normal: #3987e5; --attack: #d95926;
    --normal-bg: #13253d; --attack-bg: #3a2116;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
  }

  .viz-root {
    min-height: 100vh;
    background: var(--surface-0);
    color: var(--text-primary);
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    padding: 32px 20px 64px;
  }
  .viz-root * { box-sizing: border-box; }
  .wrap { max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 36px; }
  h1, h2, h3 { font-family: "Archivo", system-ui, sans-serif; text-wrap: balance; margin: 0; }
  h1 { font-size: clamp(26px, 4vw, 36px); font-weight: 800; letter-spacing: -0.01em; }
  h2 { font-size: 19px; font-weight: 700; }
  h3 { font-size: 13.5px; font-weight: 700; }
  .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .eyebrow {
    font-family: "IBM Plex Mono", monospace; font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted);
  }
  p { color: var(--text-secondary); line-height: 1.55; margin: 0; max-width: 72ch; }

  section { display: flex; flex-direction: column; gap: 14px; }
  .section-head { display: flex; flex-direction: column; gap: 4px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
  @media (max-width: 860px) { .cols { grid-template-columns: 1fr; } }
  .col-head {
    display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 8px 8px 0 0;
    font-family: "Archivo", sans-serif; font-weight: 700; font-size: 15px;
  }
  .col-head.normal { background: var(--normal-bg); color: var(--normal); }
  .col-head.attack { background: var(--attack-bg); color: var(--attack); }
  .col-body { display: flex; flex-direction: column; gap: 14px; border: 1px solid var(--border); border-top: none;
              border-radius: 0 0 10px 10px; padding: 14px; background: var(--surface-1); box-shadow: var(--shadow); }

  .shared-target-banner {
    display: block; line-height: 1.6; padding: 10px 14px; border-radius: 8px;
    background: var(--surface-1); border: 1px dashed var(--border); font-size: 12.5px; color: var(--text-secondary);
  }
  .shared-target-banner b { font-family: "IBM Plex Mono", monospace; color: var(--text-primary); }

  /* ---- bar charts ---- */
  .chart-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  @media (max-width: 860px) { .chart-grid { grid-template-columns: 1fr; } }
  .chart-card { border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1);
                box-shadow: var(--shadow); padding: 14px; display: flex; flex-direction: column; gap: 8px; }
  .chart-card h3 { font-size: 12.5px; color: var(--text-secondary); font-weight: 600; }
  .chart-svg-box { position: relative; }
  .chart-svg-box svg { display: block; width: 100%; height: auto; overflow: visible; }
  .bar-axis-label { font-size: 10px; fill: var(--text-muted); font-family: "IBM Plex Mono", monospace; }
  .bar-cat-label { font-size: 11px; fill: var(--text-secondary); font-family: "IBM Plex Sans", sans-serif; }
  .bar-value-label { font-size: 11px; font-weight: 600; font-family: "IBM Plex Mono", monospace; }
  .bar-gridline { stroke: var(--border); stroke-width: 1; }
  .bar-legend { display: flex; gap: 14px; align-items: center; }
  .bar-legend .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--text-secondary); }
  .bar-legend .swatch { width: 9px; height: 9px; border-radius: 2px; }

  .frame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .frame-label { font-size: 11px; font-weight: 600; color: var(--text-muted); padding: 7px 10px;
                 background: var(--surface-0); border-bottom: 1px solid var(--border); }
  .frame-body { padding: 10px 10px 4px; }
  .fline { display: flex; gap: 8px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px;
           padding: 3px 0; border-bottom: 1px dashed var(--border); }
  .fline:last-child { border-bottom: none; }
  .fk { color: var(--text-muted); min-width: 92px; flex: none; }
  .fv { color: var(--text-primary); word-break: break-all; }
  .fv.hex { color: var(--normal); }
  .attack .fv.hex, .frame.is-attack .fv.hex { color: var(--attack); }
  .decoded { font-family: "IBM Plex Sans", sans-serif; font-size: 12px; color: var(--text-secondary);
             padding: 8px 10px; background: var(--surface-0); border-top: 1px solid var(--border); }
  .decoded b { color: var(--text-primary); }
  .flag { display: inline-block; font-size: 10px; font-weight: 600; padding: 1px 6px; border-radius: 4px;
          background: var(--attack-bg); color: var(--attack); margin-left: 6px; }

  table.stats { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.stats th, table.stats td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  table.stats th { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--text-muted);
                   font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
  table.stats td.metric { color: var(--text-secondary); }
  table.stats td.normal-val { font-family: "IBM Plex Mono", monospace; color: var(--normal); font-weight: 600; }
  table.stats td.attack-val { font-family: "IBM Plex Mono", monospace; color: var(--attack); font-weight: 600; }
  table.stats tr.deviates td.attack-val { position: relative; }
  table.stats tr.deviates td.attack-val::after { content: "\25B2 deviates"; display: block; font-size: 9.5px;
    font-family: "IBM Plex Sans", sans-serif; font-weight: 500; color: var(--attack); opacity: .75; }
  .stats-wrap { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface-1); box-shadow: var(--shadow); overflow-x: auto; }

  /* ---- detection signals ---- */
  .signal-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  @media (max-width: 860px) { .signal-grid { grid-template-columns: 1fr; } }
  .signal-card { border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1);
                 box-shadow: var(--shadow); padding: 14px; display: flex; flex-direction: column; gap: 8px; }
  .signal-card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .signal-title { font-family: "Archivo", sans-serif; font-weight: 700; font-size: 14px; }
  .signal-tag { display: inline-block; font-size: 9.5px; font-weight: 600; padding: 2px 7px; border-radius: 4px;
                font-family: "IBM Plex Mono", monospace; text-transform: uppercase; letter-spacing: .03em; white-space: nowrap; }
  .signal-tag.categorical { background: var(--attack-bg); color: var(--attack); }
  .signal-tag.statistical { background: var(--normal-bg); color: var(--normal); }
  .signal-values { display: flex; gap: 18px; font-family: "IBM Plex Mono", monospace; font-size: 12.5px; }
  .signal-values .val-label { color: var(--text-muted); font-size: 10px; display: block; text-transform: uppercase; letter-spacing: .03em; }
  .signal-values .val-normal { color: var(--normal); font-weight: 600; }
  .signal-values .val-attack { color: var(--attack); font-weight: 600; }
  .signal-note { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }

  /* ---- time series ---- */
  .timeline-card { border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1);
                    box-shadow: var(--shadow); padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .timeline-card .chart-svg-box svg { overflow: visible; }

  footer { border-top: 1px solid var(--border); padding-top: 18px; }
  footer p { font-size: 12px; }
</style>

<div class="viz-root">
<div class="wrap">

  <header>
    <span class="eyebrow">Smart Grid &middot; ICS-SimLab capture &middot; real packets, not synthetic</span>
    <h1 style="margin-top:8px">A wall of randomized reads, ~6,000 a second</h1>
    <p style="margin-top:10px">The last attack in this series is also the most intense: <b id="rate-inline"></b>
      requests per second sustained for <b id="dur-inline"></b> (t&nbsp;=&nbsp;46.9s-53.9s), on only
      <b id="conn-inline"></b> TCP connections. Unlike every other attack here, address and quantity
      aren't fixed or maxed out - they're <b>randomized</b> across the full valid range each time. This
      is the same session-wide burst already found by <code class="mono">data_visualisation.py</code>
      (XAI-15) across all 3 datasets; this page adds the packet-level content.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>The one attack in this series with zero background noise</h2>
      <p style="margin-top:6px">A contrast to naive-sensor-read (30.8%), sporadic-injection (34.4%),
        force-listen (40.3%) and restart-comm (43.2%).</p>
    </div>
    <div class="shared-target-banner">
      <b>0.0%</b> of this attack's 85,010 rows are unrelated background traffic. The flood is so
      overwhelming within its narrow 7-second window that there is simply no room for incidental
      DATA/WEBSOCKET chatter to appear alongside it - unlike the longer, lower-intensity attacks
      earlier in this series, where the attacker's OTHER (legitimate) traffic got swept into the same
      attack_specific window.
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; a routine value poll</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; randomized flood read</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Detection signals: what actually marks this traffic as an attack</h2>
      <p style="margin-top:6px">Each signal below is checked directly against the whole session's
        normal baseline, not assumed. <span class="signal-tag categorical" style="margin:0 4px">categorical</span>
        means the value never occurs at all in normal traffic (zero-ambiguity); <span class="signal-tag statistical" style="margin:0 4px">statistical</span>
        means normal traffic does have this value, but at a very different magnitude.</p>
    </div>
    <div class="signal-grid" id="signal-grid"></div>
  </section>

  <section>
    <div class="section-head">
      <h2>Time series: request rate across the whole session</h2>
      <p style="margin-top:6px">A single towering spike very early in the session, then nothing -
        the opposite shape from naive-sensor-read's 18 recurring bursts or restart-comm's 10 evenly
        scattered rounds.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal (master &rarr; RTU)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Data flood</span>
    </div>
    <div class="timeline-card">
      <div class="chart-svg-box" id="timeline-chart"></div>
      <p style="font-size:12px" id="timeline-caption"></p>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Statistics: this attack type vs. the session's normal baseline</h2>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal baseline</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Data flood</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Data flood</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows labeled
      <code class="mono">attack_specific == 8</code>. Same burst already characterized session-wide
      (all 3 datasets, temporal distribution + detection-window AUC experiment) by
      <code class="mono">data_visualisation.py</code> (XAI-15). Companion page to
      <code class="mono">packet_compare_smartgrid_naivesensorread.py</code> (shares
      <code class="mono">extract_normal_pair()</code>) - last in the Smart Grid per-attack-type
      series.</p>
  </footer>

</div>
</div>

<script>
(function () {
  const DATA = __PACKET_DATA__;

  function esc(s) {
    return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  }

  function frame(label, pkt, isAttack, note) {
    const rows = [
      ['time', pkt.time],
      ['src -> dst', `${pkt.ip_src} -> ${pkt.ip_dst}`],
      ['function code', `${pkt.modbus_func_code != null ? Math.trunc(pkt.modbus_func_code) : 'n/a'}`],
      ['pdu (hex)', pkt.modbus_data || 'n/a', true],
      ['ip_len / tcp_len', `${pkt.ip_len} / ${pkt.tcp_len} bytes`],
    ];
    const lines = rows.map(([k, v, hex]) =>
      `<div class="fline"><span class="fk">${k}</span><span class="fv${hex ? ' hex' : ''}">${esc(v)}</span></div>`
    ).join('');
    return `<div class="frame${isAttack ? ' is-attack' : ''}">
      <div class="frame-label">${esc(label)}</div>
      <div class="frame-body">${lines}</div>
      ${note ? `<div class="decoded">${note}</div>` : ''}
    </div>`;
  }

  const s = DATA.stats;

  // ---- headline callouts ----
  document.getElementById('rate-inline').innerHTML = `<b>${s.attack_rate_per_sec.toLocaleString()}</b>`;
  document.getElementById('dur-inline').textContent = s.attack_duration_sec + 's';
  document.getElementById('conn-inline').innerHTML = `<b>${s.n_tcp_connections}</b>`;

  // ---- normal column ----
  const nCol = document.getElementById('normal-col');
  const np = DATA.normal_pair;
  let normalHtml = '';
  if (np) {
    normalHtml += `<div><h3 style="margin-bottom:8px">A routine value poll<span class="flag" style="background:var(--normal-bg); color:var(--normal)">quantity = 1</span></h3>` +
      frame('REQUEST', np.request, false,
        `Read Coils (fc1), one of 4 fixed addresses, quantity <b>1</b> &mdash; every normal read in
         this session asks for exactly one value at one of a handful of known points.`) +
      frame('RESPONSE', np.response, false,
        `A single value, nothing more.`) +
      `</div>`;
  }
  nCol.innerHTML = normalHtml;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml = '';
  if (ex.early_example) {
    attackHtml += `<div><h3 style="margin-bottom:8px">1. Early in the flood<span class="flag">randomized address/quantity</span></h3>` +
      frame('REQUEST', ex.early_example.request, true,
        `Address and quantity both drawn from a wide range each request &mdash; ${s.address_distinct}
         distinct addresses (0-${s.address_max}) and ${s.quantity_distinct} distinct quantities
         (1-${s.quantity_max}) seen across the burst, not a fixed target like every other attack in
         this series.`) +
      frame('RESPONSE', ex.early_example.response, true,
        `Response size varies with the random quantity &mdash; up to ${s.attack_max_resp_bytes} bytes,
         vs. a ${s.normal_max_resp_bytes}-byte normal maximum.`) +
      `</div>`;
  }
  if (ex.mid_example) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. Mid-flood, same random pattern<span class="flag">still accepted</span></h3>` +
      frame('REQUEST', ex.mid_example.request, true,
        `A different address/quantity pair, taken from roughly the middle of the burst &mdash; the
         randomization and the request rate both stay constant throughout, not ramping up or down.`) +
      frame('RESPONSE', ex.mid_example.response, true,
        `Still answered &mdash; the target keeps responding to every one of the
         ${s.n_requests.toLocaleString()} requests despite the rate.`) +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- detection signals ----
  const signals = [
    {
      title: 'Request rate', tag: 'statistical',
      normal: `${s.normal_rate_per_sec}/s`, attack: `${s.attack_rate_per_sec.toLocaleString()}/s`,
      note: `${s.rate_ratio.toLocaleString()}&times; the normal request rate - by far the largest
             magnitude deviation of any attack in this series.`,
    },
    {
      title: 'Request address range', tag: 'statistical',
      normal: `${s.normal_addresses.length} fixed points (${s.normal_addresses.join(', ')})`, attack: `${s.address_distinct} distinct (0-${s.address_max})`,
      note: `Sweeps the full valid address range instead of the 4 specific points normal traffic ever
             touches - not a fixed target like naive-sensor-read's address 0.`,
    },
    {
      title: 'Read quantity range', tag: 'statistical',
      normal: `always ${s.normal_max_qty}`, attack: `${s.quantity_distinct} distinct (${s.quantity_min}-${s.quantity_max})`,
      note: `Randomized each request, unlike naive-sensor-read's fixed protocol-maximum (2000/125) -
             this attack maximizes REQUEST VOLUME instead of per-request size.`,
    },
    {
      title: 'Requests per TCP connection', tag: 'statistical',
      normal: `~1 per ${(60/24.7).toFixed(1)}s (new connection)`, attack: `~${s.requests_per_connection.toLocaleString()} per connection`,
      note: `Only ${s.n_tcp_connections} TCP connections carry all ${s.n_requests.toLocaleString()}
             requests - connections are reused intensively instead of opened per-request, unlike
             device-id/force-listen/restart-comm's new-connection-per-round pattern.`,
    },
    {
      title: 'Background noise in labeled rows', tag: 'statistical',
      normal: '0%', attack: `${s.attack_noise_pct}%`,
      note: `The only attack in this series with ZERO co-mingled background traffic - the flood is too
             intense and too brief (${s.attack_duration_sec}s) to leave room for anything else.`,
    },
    {
      title: 'Response size range', tag: 'statistical',
      normal: `max ${s.normal_max_resp_bytes} bytes`, attack: `up to ${s.attack_max_resp_bytes} bytes`,
      note: `Varies with the randomized quantity per request, unlike the other attacks' constant
             response sizes.`,
    },
  ];
  document.getElementById('signal-grid').innerHTML = signals.map(sig => `
    <div class="signal-card">
      <div class="signal-card-head">
        <span class="signal-title">${esc(sig.title)}</span>
        <span class="signal-tag ${sig.tag}">${sig.tag}</span>
      </div>
      <div class="signal-values">
        <span><span class="val-label">Normal</span><span class="val-normal">${esc(sig.normal)}</span></span>
        <span><span class="val-label">Attack</span><span class="val-attack">${esc(sig.attack)}</span></span>
      </div>
      <div class="signal-note">${sig.note}</div>
    </div>`).join('');

  // ---- time series (request rate across the whole session) ----
  function drawTimeSeries(tl) {
    const width = 1000, height = 260;
    const pad = {top: 16, right: 16, bottom: 30, left: 46};
    const innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom;
    const dur = tl.session_duration_sec;
    const maxV = Math.max(...tl.normal_counts, ...tl.attack_counts, 1) * 1.15;
    const x = t => pad.left + (t / dur) * innerW;
    const y = v => pad.top + innerH - (v / maxV) * innerH;
    const nColor = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--normal').trim();
    const aColor = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--attack').trim();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

    [0, 0.5, 1].forEach(frac => {
      const gy = pad.top + innerH - frac * innerH;
      const gl = document.createElementNS(svg.namespaceURI, 'line');
      gl.setAttribute('x1', pad.left); gl.setAttribute('x2', width - pad.right);
      gl.setAttribute('y1', gy); gl.setAttribute('y2', gy);
      gl.setAttribute('class', 'bar-gridline');
      svg.appendChild(gl);
      const lbl = document.createElementNS(svg.namespaceURI, 'text');
      lbl.setAttribute('x', pad.left - 6); lbl.setAttribute('y', gy + 3);
      lbl.setAttribute('text-anchor', 'end'); lbl.setAttribute('class', 'bar-axis-label');
      lbl.textContent = fmtNum(Math.round(frac * maxV));
      svg.appendChild(lbl);
    });

    for (let m = 0; m <= dur / 60; m += 10) {
      const gx = x(m * 60);
      const tick = document.createElementNS(svg.namespaceURI, 'text');
      tick.setAttribute('x', gx); tick.setAttribute('y', height - 8);
      tick.setAttribute('text-anchor', 'middle'); tick.setAttribute('class', 'bar-axis-label');
      tick.textContent = m + 'm';
      svg.appendChild(tick);
    }

    function areaPath(counts) {
      const pts = tl.bin_centers.map((t, i) => `${x(t)},${y(counts[i])}`);
      return `M${pad.left},${y(0)} L${pts.join(' L')} L${x(dur)},${y(0)} Z`;
    }
    function linePath(counts) {
      const pts = tl.bin_centers.map((t, i) => `${x(t)},${y(counts[i])}`);
      return `M${pts.join(' L')}`;
    }

    const attackArea = document.createElementNS(svg.namespaceURI, 'path');
    attackArea.setAttribute('d', areaPath(tl.attack_counts));
    attackArea.setAttribute('fill', aColor); attackArea.setAttribute('fill-opacity', '0.55');
    attackArea.setAttribute('stroke', aColor); attackArea.setAttribute('stroke-width', '1');
    svg.appendChild(attackArea);

    const normalLine = document.createElementNS(svg.namespaceURI, 'path');
    normalLine.setAttribute('d', linePath(tl.normal_counts));
    normalLine.setAttribute('fill', 'none');
    normalLine.setAttribute('stroke', nColor); normalLine.setAttribute('stroke-width', '1.75');
    svg.appendChild(normalLine);

    return svg;
  }

  function fmtNum(v) { return v >= 1000 ? Math.round(v).toLocaleString() : (Number.isInteger(v) ? v : v.toFixed(1)); }

  const tl = DATA.timeline;
  document.getElementById('timeline-chart').appendChild(drawTimeSeries(tl));
  document.getElementById('timeline-caption').innerHTML =
    `Normal traffic averages <b class="mono" style="color:var(--normal)">${tl.normal_avg_per_bin}</b>
     requests per 10s bin, continuous throughout. This attack sits at
     <b class="mono" style="color:var(--attack)">0</b> for the rest of the session, then spikes to
     <b class="mono" style="color:var(--attack)">${fmtNum(tl.attack_max_per_bin)}</b> requests in a
     single bin near the very start &mdash; active in only
     <b class="mono" style="color:var(--attack)">${tl.attack_active_bin_pct}%</b> of the session's
     bins, the most extreme single-burst shape in this series.`;

  // ---- bar charts ----
  function drawBarPair(container, {label, normal, attack, unit, logScale}) {
    const width = 300, height = 150;
    const pad = {top: 14, right: 10, bottom: 30, left: 38};
    const innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom;
    const maxV = Math.max(normal, attack, logScale ? 1 : 0.0001);
    const floor = logScale ? Math.max(Math.min(normal || 1, attack || 1) * 0.5, 0.01) : 0;
    function y(v) {
      if (!logScale) return pad.top + innerH - (v / (maxV * 1.15)) * innerH;
      const lv = Math.log10(Math.max(v, floor)), lo = Math.log10(floor), hi = Math.log10(maxV * 1.3);
      return pad.top + innerH - ((lv - lo) / (hi - lo)) * innerH;
    }
    const barW = 46, gap = 26;
    const x0 = pad.left + innerW / 2 - barW - gap / 2;
    const x1 = pad.left + innerW / 2 + gap / 2;
    const nColor = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--normal').trim();
    const aColor = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--attack').trim();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

    [0, 0.5, 1].forEach(frac => {
      const gy = pad.top + innerH - frac * innerH;
      const gl = document.createElementNS(svg.namespaceURI, 'line');
      gl.setAttribute('x1', pad.left); gl.setAttribute('x2', width - pad.right);
      gl.setAttribute('y1', gy); gl.setAttribute('y2', gy);
      gl.setAttribute('class', 'bar-gridline');
      svg.appendChild(gl);
    });

    function bar(x, v, color, label2) {
      const barTop = y(v);
      const rect = document.createElementNS(svg.namespaceURI, 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', barTop);
      rect.setAttribute('width', barW); rect.setAttribute('height', Math.max(pad.top + innerH - barTop, 1.5));
      rect.setAttribute('rx', 3); rect.setAttribute('fill', color);
      svg.appendChild(rect);
      const val = document.createElementNS(svg.namespaceURI, 'text');
      val.setAttribute('x', x + barW / 2); val.setAttribute('y', barTop - 6);
      val.setAttribute('text-anchor', 'middle'); val.setAttribute('class', 'bar-value-label');
      val.setAttribute('fill', color);
      val.textContent = fmtNum(v) + (unit ? ` ${unit}` : '');
      svg.appendChild(val);
      const cat = document.createElementNS(svg.namespaceURI, 'text');
      cat.setAttribute('x', x + barW / 2); cat.setAttribute('y', height - 10);
      cat.setAttribute('text-anchor', 'middle'); cat.setAttribute('class', 'bar-cat-label');
      cat.textContent = label2;
      svg.appendChild(cat);
    }
    bar(x0, normal, nColor, 'Normal');
    bar(x1, attack, aColor, 'Attack');
    return svg;
  }

  const countCharts = [
    {label: 'Request rate (req/sec)', normal: s.normal_rate_per_sec, attack: s.attack_rate_per_sec, logScale: true},
    {label: 'Distinct addresses touched', normal: s.normal_addresses.length, attack: s.address_distinct, logScale: true},
    {label: 'Requests per TCP connection', normal: 1, attack: s.requests_per_connection, logScale: true},
  ];
  const countChartsEl = document.getElementById('count-charts');
  countCharts.forEach(cfg => {
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<h3>${cfg.label}</h3><div class="chart-svg-box"></div>`;
    card.querySelector('.chart-svg-box').appendChild(drawBarPair(card, cfg));
    countChartsEl.appendChild(card);
  });

  // ---- stats table ----
  const rows = [
    ['Requests', '—', `${s.n_requests.toLocaleString()} in ${s.attack_duration_sec}s`, true],
    ['Request rate', `${s.normal_rate_per_sec}/s`, `${s.attack_rate_per_sec.toLocaleString()}/s (${s.rate_ratio.toLocaleString()}x)`, true],
    ['Distinct request addresses', `${s.normal_addresses.length} (${s.normal_addresses.join(', ')})`, `${s.address_distinct} (0-${s.address_max})`, true],
    ['Distinct read quantities', '1', `${s.quantity_distinct} (${s.quantity_min}-${s.quantity_max})`, true],
    ['Distinct request PDUs', '—', `${s.distinct_pdus.toLocaleString()} of ${s.n_requests.toLocaleString()} requests`, false],
    ['TCP connections used', '—', `${s.n_tcp_connections} (~${s.requests_per_connection.toLocaleString()} requests each)`, true],
    ['Largest response', `${s.normal_max_resp_bytes} bytes`, `${s.attack_max_resp_bytes} bytes`, false],
    ['Rows labeled attack_specific=8', '—', `${s.attack_total_rows.toLocaleString()}`, false],
    ['...of which unrelated background traffic', '0%', `${s.attack_noise_pct}%`, false],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
