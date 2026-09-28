"""
Side-by-side comparison of one NORMAL Modbus packet exchange vs. one round of
the "function code scan" attack on the Smart Grid dataset - both request and
response, field by field, plus the statistics that separate them.

Built to answer a concrete question raised while designing the analyst-style
prompt (see the Smart Grid normal-behavior-baseline memory): a single flagged
packet only shows one function code and a 4-second-averaged rate, which hides
what the attacker actually did across the whole scan round (45 function codes
tried in ~22 ms, including a write command the target ACCEPTED and a Report
Slave ID reply that discloses the target's software). This script pulls the
exact raw rows for one normal exchange and 3 representative attack exchanges
from that round, plus the aggregate stats, and renders a static comparison
page - no LLM calls, this is a data-exploration script, not a pipeline run.

Output goes into data_visualisation/smartgrid_function_code_scan/ (all
filenames get the optional --tag suffix so earlier results are not
overwritten): packets.json, stats.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_fcscan.py 2>&1 | tee data_visualisation/smartgrid_function_code_scan/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_function_code_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
STANDARD_FC = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 20, 21, 22, 23, 24, 43}
WRITE_FC = {5, 6, 15, 16}
SCAN_STREAMS = [6064, 6065, 6066, 6067, 6068]   # the 5 TCP connections opened in round 1

FIELDS = ["row", "time", "frame_time_relative", "ip_src", "ip_dst", "ip_len", "ip_ttl",
          "ip_proto", "tcp_len", "tcp_flags", "tcp_window_size", "tcp_analysis_ack_rtt",
          "frame_time_delta", "modbus_func_code", "modbus_data", "tcp_stream"]


def pack(row):
    d = {}
    for f in FIELDS:
        v = row[f]
        if pd.isna(v):
            v = None
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        else:
            v = str(v)
        d[f] = v
    return d


def decode_qty(hexdata):
    """Decode (start_address, quantity) from a read request's PDU hex string."""
    s = str(hexdata)[2:] if str(hexdata).startswith("0x") else str(hexdata)
    if len(s) < 5:
        return None, None
    try:
        return int(s[:4], 16), int(s[4:], 16)
    except ValueError:
        return None, None


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def extract_normal_pair(df):
    """A routine fc1 read from the known master, picked from the middle of the session."""
    n = df[(df.protocol == "MODBUS") & df.attack_specific.isna() & (df.tcp_len == 12)
           & (df.ip_src == "192.168.0.40") & (df.modbus_func_code == 1)]
    req_row = n.iloc[len(n) // 2]
    rsp_row = df.iloc[int(req_row.row) + 1]
    return {"request": pack(req_row), "response": pack(rsp_row)}


def extract_attack_examples(df):
    """3 request/response pairs from scan round 1, each showing a different signal."""
    a = df[(df.attack_specific == 2) & (df.protocol == "MODBUS")
           & df.frame_time_relative.between(549.45, 549.47)].sort_values("row")
    examples = {}
    for name, fc in [("probe_unsupported_fc2", 2), ("write_fc5_accepted", 5), ("identity_leak_fc17", 17)]:
        r = a[a.modbus_func_code == fc]
        if len(r) >= 2:
            examples[name] = {"request": pack(r.iloc[0]), "response": pack(r.iloc[1])}
    return examples


def compute_stats(df):
    n = df[(df.protocol == "MODBUS") & (df.attack_specific.isna() | (df.attack_specific == 0))]
    a_all = df[(df.protocol == "MODBUS") & df.attack_specific.notna() & (df.attack_specific != 0)]
    normal_ips = sorted(set(n.ip_src.unique()) | set(n.ip_dst.unique()))
    attack_ips = sorted(set(a_all.ip_src.unique()) | set(a_all.ip_dst.unique()))
    normal_fc = sorted(int(x) for x in n.modbus_func_code.dropna().unique())
    n_req = n[n.tcp_len == 12]
    normal_qtys = [decode_qty(d)[1] for d in n_req.modbus_data]
    normal_maxq = max(q for q in normal_qtys if q is not None)
    normal_max_resp = int(n[n.tcp_len != 12].ip_len.max())

    n_all = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    minutes = df.frame_time_relative.max() / 60
    normal_conn_per_min = round(n_all.tcp_stream.nunique() / minutes, 1)
    normal_rate = round(len(n) / df.frame_time_relative.max(), 2)

    r1 = df[df.tcp_stream.isin(SCAN_STREAMS)]
    req = r1[(r1.ip_src == ATTACKER_IP) & (r1.protocol == "MODBUS")]
    rsp = r1[(r1.ip_dst == ATTACKER_IP) & (r1.protocol == "MODBUS")]
    fcs = sorted(int(x) for x in req.modbus_func_code.dropna().unique())
    nonstd_fc = [f for f in fcs if f not in STANDARD_FC]
    writes_sent = sorted(f for f in fcs if f in WRITE_FC)

    qs = [decode_qty(d)[1] for d, ln, fc in zip(req.modbus_data, req.tcp_len, req.modbus_func_code)
          if ln == 12 and fc in (1, 2, 3, 4)]
    qs = [q for q in qs if q is not None]
    round1_maxq = max(qs) if qs else None
    round1_max_resp = int(rsp[rsp.tcp_len != 12].ip_len.max()) if (rsp.tcp_len != 12).any() else None
    dur_ms = round((r1.frame_time_relative.max() - r1.frame_time_relative.min()) * 1000, 1)

    rq = req.dropna(subset=["tcp_stream"]).sort_values("frame_time_relative")[
        ["frame_time_relative", "tcp_stream", "modbus_func_code"]]
    rp = rsp.dropna(subset=["tcp_stream"]).sort_values("frame_time_relative")[
        ["frame_time_relative", "tcp_stream", "modbus_func_code", "tcp_len"]]
    m = pd.merge_asof(rq, rp, on="frame_time_relative", by="tcp_stream", direction="forward",
                      tolerance=0.05, suffixes=("_q", "_r"))
    accepted = sorted(int(x) for x in m[(m.modbus_func_code_q.isin(WRITE_FC))
                                        & (m.modbus_func_code_r == m.modbus_func_code_q)
                                        & (m.tcp_len >= 12)].modbus_func_code_q.unique())

    dur_s = dur_ms / 1000
    attack_pkt_rate = round(len(r1) / dur_s, 1)
    attack_req_rate = round(len(req) / dur_s, 1)
    normal_conn_rate = round(normal_conn_per_min / 60, 3)
    attack_conn_rate = round(len(SCAN_STREAMS) / dur_s, 1)

    return {
        "normal_n_rows": int(len(n)),
        "normal_ips": normal_ips,
        "attack_ips": attack_ips,
        "attacker_never_in_normal": ATTACKER_IP not in normal_ips,
        "master_never_in_attack": "192.168.0.40" not in attack_ips,
        "shared_target_ip": "192.168.0.31",
        "normal_fc": normal_fc,
        "normal_maxq": normal_maxq,
        "normal_max_resp": normal_max_resp,
        "normal_conn_per_min": normal_conn_per_min,
        "normal_rate": normal_rate,
        "normal_pkt_rate": normal_rate,
        "normal_req_rate": round(normal_rate / 2, 2),
        "normal_conn_rate": normal_conn_rate,
        "round1_n_pkts_total": int(len(r1)),
        "round1_dur_ms": dur_ms,
        "round1_n_req": int(len(req)),
        "round1_n_connections": len(SCAN_STREAMS),
        "round1_distinct_fc": len(fcs),
        "round1_fc_list": fcs,
        "round1_nonstd_fc": nonstd_fc,
        "round1_maxq": round1_maxq,
        "round1_max_resp": round1_max_resp,
        "round1_writes_sent": writes_sent,
        "round1_writes_accepted": accepted,
        "attack_pkt_rate": attack_pkt_rate,
        "attack_req_rate": attack_req_rate,
        "attack_conn_rate": attack_conn_rate,
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
    parser = argparse.ArgumentParser(description="Compare a normal packet vs. a function-code-scan round on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting normal packet pair...")
    normal_pair = extract_normal_pair(df)
    print("Extracting attack examples (scan round 1, t~549.46s)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)

    packets = {"normal_pair": normal_pair, "attack_examples": attack_examples}
    with open(path("packets.json"), "w") as f:
        json.dump(packets, f, indent=1)
    print(f"Saved: {path('packets.json')}")

    with open(path("stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"Saved: {path('stats.json')}")

    payload = {**packets, "stats": stats}
    report_path = path("report.html")
    report_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"Round 1 summary: {stats['round1_n_req']} requests, {stats['round1_distinct_fc']} distinct "
          f"function codes ({len(stats['round1_nonstd_fc'])} non-standard), writes accepted: "
          f"{stats['round1_writes_accepted'] or 'none'}")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Modbus Packet Diff</title>
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

  /* ---- IP highlight ---- */
  .ip-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 860px) { .ip-grid { grid-template-columns: 1fr; } }
  .ip-card { border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1);
             box-shadow: var(--shadow); overflow: hidden; }
  .ip-card .col-head { border-radius: 0; }
  .ip-card-body { padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .ip-chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .ip-chip { font-family: "IBM Plex Mono", monospace; font-size: 12px; padding: 4px 9px; border-radius: 6px;
             border: 1px solid var(--border); background: var(--surface-0); }
  .ip-chip.role-target { font-weight: 700; }
  .ip-card.normal .ip-chip.role-active { border-color: var(--normal); color: var(--normal); background: var(--normal-bg); }
  .ip-card.attack .ip-chip.role-active { border-color: var(--attack); color: var(--attack); background: var(--attack-bg); }
  .ip-note { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
  .ip-note b { color: var(--text-primary); }
  .shared-target-banner {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 8px;
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
  .bar-tooltip {
    position: absolute; pointer-events: none; z-index: 5; background: var(--text-primary); color: var(--surface-1);
    font-family: "IBM Plex Mono", monospace; font-size: 11px; padding: 5px 8px; border-radius: 6px;
    white-space: nowrap; opacity: 0; transform: translate(-50%, -100%); transition: opacity .08s ease;
    box-shadow: var(--shadow);
  }

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

  footer { border-top: 1px solid var(--border); padding-top: 18px; }
  footer p { font-size: 12px; }
</style>

<div class="viz-root">
<div class="wrap">

  <header>
    <span class="eyebrow">Smart Grid &middot; ICS-SimLab capture &middot; real packets, not synthetic</span>
    <h1 style="margin-top:8px">One normal poll vs. one function-code-scan round</h1>
    <p style="margin-top:10px">Every field below is copied verbatim from the raw CSV &mdash; nothing is
      illustrative. Left: a routine read from the known SCADA master (<code class="mono">192.168.0.40</code>)
      to the RTU (<code class="mono">192.168.0.31</code>), picked from the middle of the session.
      Right: 3 packets from the same 22&nbsp;ms attack burst (<code class="mono">192.168.0.1</code>,
      not the known master), each illustrating a different thing the attacker learned.</p>
  </header>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; routine poll</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; function code scan</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Source IP: who is talking to the RTU</h2>
      <p style="margin-top:6px">The clearest deviation isn't a number at all &mdash; it's <em>who sent the
        packet</em>. Every IP that ever appears in normal traffic vs. every IP that ever appears in
        attack traffic, for this entire session (not just this round).</p>
    </div>
    <div class="ip-grid" id="ip-grid"></div>
    <div class="shared-target-banner" id="shared-target-banner"></div>
  </section>

  <section>
    <div class="section-head">
      <h2>Statistics: this round vs. the session's normal baseline</h2>
      <p style="margin-top:6px">Same look-back window used by the analyst prompt (the 22&nbsp;ms burst / 5
        connections that make up this scan round), compared against normal traffic measured across the
        entire 99.4-minute session.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal baseline</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>This attack round</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <div class="chart-grid" id="rate-chart-row" style="grid-template-columns: 1fr;"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>This attack round</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows filtered to
      <code class="mono">protocol == "MODBUS"</code>. The attack round is the 5 TCP connections
      (streams 6064&ndash;6068) opened by 192.168.0.1 starting at t&nbsp;=&nbsp;549.46s. PDU bytes shown
      exactly as captured; "0x0950796d6f64627573ff" decodes as byte-count&nbsp;9, ASCII "Pymodbus", run
      indicator 0xff &mdash; the standard Modbus Report Slave ID (fc17) response shape.</p>
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

  // ---- normal column ----
  const nCol = document.getElementById('normal-col');
  const s0 = DATA.stats;
  nCol.innerHTML =
    frame('REQUEST', DATA.normal_pair.request, false,
      `<b>Read Coils</b> (fc1), start address 9, quantity 1 &mdash; one coil, one poll cycle.`) +
    frame('RESPONSE', DATA.normal_pair.response, false,
      `Byte count 1, value <code>0x00</code> (coil off). 62 bytes total, answered in 0.57&nbsp;ms.`) +
    `<div class="frame"><div class="frame-label">CONTEXT &mdash; this pair over the full session</div>
      <div class="decoded" style="border-top:none">
        This exact request/response shape repeats <b>${s0.normal_n_rows.toLocaleString()}</b>
        times across the 99.4-minute session &mdash; only 3 function codes ever appear
        (fc${s0.normal_fc.join(', fc')}), quantity is always ${s0.normal_maxq}, and no reply exceeds
        ${s0.normal_max_resp} bytes. One master (192.168.0.40), one slave (192.168.0.31), nothing else
        ever speaks Modbus on this network under normal conditions.
      </div>
    </div>`;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  aCol.innerHTML =
    `<div><h3 style="margin-bottom:8px">1. Probing an unsupported code<span class="flag">fc2, generic reply</span></h3>` +
    frame('REQUEST', ex.probe_unsupported_fc2.request, true,
      `Read Discrete Inputs (fc2) &mdash; never used in this system's normal traffic.`) +
    frame('RESPONSE', ex.probe_unsupported_fc2.response, true,
      `9-byte generic reply (<code>0x00</code>) &mdash; identical shape to 29 other rejected codes in this round; this is how the attacker fingerprints what ISN'T supported.`) +
    `</div>` +
    `<div><h3 style="margin-bottom:8px">2. A write command the target ACCEPTED<span class="flag">fc5, echoed back</span></h3>` +
    frame('REQUEST', ex.write_fc5_accepted.request, true,
      `Write Single Coil (fc5) &mdash; 0% of this system's normal traffic is a write.`) +
    frame('RESPONSE', ex.write_fc5_accepted.response, true,
      `Target echoes the exact same PDU back &mdash; Modbus's own confirmation that the write was
       <b>accepted</b>, not rejected. This is the MITRE ATT&amp;CK "Unauthorized Command Message" pattern.`) +
    `</div>` +
    `<div><h3 style="margin-bottom:8px">3. Device identity disclosed<span class="flag">fc17, info leak</span></h3>` +
    frame('REQUEST', ex.identity_leak_fc17.request, true,
      `Report Slave ID (fc17) &mdash; asks the device to identify itself.`) +
    frame('RESPONSE', ex.identity_leak_fc17.response, true,
      `PDU decodes to ASCII <b>"Pymodbus"</b> &mdash; the target discloses the software library it runs,
       narrowing down exploitable CVEs for the attacker.`) +
    `</div>`;

  // ---- IP highlight cards ----
  const s = DATA.stats;
  const ipGrid = document.getElementById('ip-grid');
  function ipChip(ip, activeSet) {
    const isTarget = ip === s.shared_target_ip;
    const cls = activeSet.includes(ip) ? ' role-active' : '';
    return `<span class="ip-chip${cls}${isTarget ? ' role-target' : ''}">${ip}${isTarget ? ' (target)' : ''}</span>`;
  }
  ipGrid.innerHTML = `
    <div class="ip-card normal">
      <div class="col-head normal">Normal traffic</div>
      <div class="ip-card-body">
        <div class="ip-chip-row">${s.normal_ips.map(ip => ipChip(ip, s.normal_ips)).join('')}</div>
        <div class="ip-note">Only <b>${s.normal_ips.length} IPs</b> ever speak Modbus here across
          ${s.normal_n_rows.toLocaleString()} normal packets: the master
          (<b>192.168.0.40</b>) and the RTU (<b>192.168.0.31</b>). No other address appears.</div>
      </div>
    </div>
    <div class="ip-card attack">
      <div class="col-head attack">Attack traffic (all 8 attack types)</div>
      <div class="ip-card-body">
        <div class="ip-chip-row">${s.attack_ips.map(ip => ipChip(ip, s.attack_ips)).join('')}</div>
        <div class="ip-note"><b>192.168.0.1</b> ${s.attacker_never_in_normal ? 'never appears in normal traffic at all' : 'rarely appears in normal traffic'}
          &mdash; every attack packet originates from it. The legitimate master
          (192.168.0.40) ${s.master_never_in_attack ? 'never sends a single attack packet' : 'is also seen in some attack rows'}.</div>
      </div>
    </div>`;
  document.getElementById('shared-target-banner').innerHTML =
    `Only <b>${s.shared_target_ip}</b> (the RTU) appears on both sides &mdash; it is the one constant:
     everyone talks to it, but who talks TO it is the tell.`;

  // ---- bar charts ----
  function fmtNum(v) { return v >= 1000 ? Math.round(v).toLocaleString() : (Number.isInteger(v) ? v : v.toFixed(1)); }

  function drawBarPair(container, {label, normal, attack, unit, logScale}) {
    const width = 300, height = 150;
    const pad = {top: 14, right: 10, bottom: 30, left: 38};
    const innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom;
    const maxV = Math.max(normal, attack, logScale ? 1 : 0.0001);
    const floor = logScale ? Math.max(Math.min(normal, attack) * 0.5, 0.01) : 0;
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
    {label: 'Distinct function codes', normal: s.normal_fc.length, attack: s.round1_distinct_fc},
    {label: 'Non-standard function codes', normal: 0, attack: s.round1_nonstd_fc.length},
    {label: 'Largest reply (bytes)', normal: s.normal_max_resp, attack: s.round1_max_resp},
  ];
  const countChartsEl = document.getElementById('count-charts');
  countCharts.forEach(cfg => {
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<h3>${cfg.label}</h3><div class="chart-svg-box"></div>`;
    card.querySelector('.chart-svg-box').appendChild(drawBarPair(card, cfg));
    countChartsEl.appendChild(card);
  });

  const rateChart = {
    label: 'Rate (log scale, events/sec)',
    groups: [
      {name: 'Packets/sec', normal: s.normal_pkt_rate, attack: s.attack_pkt_rate},
      {name: 'Modbus requests/sec', normal: s.normal_req_rate, attack: s.attack_req_rate},
      {name: 'New connections/sec', normal: s.normal_conn_rate, attack: s.attack_conn_rate},
    ],
  };
  const rateCard = document.createElement('div');
  rateCard.className = 'chart-card';
  rateCard.innerHTML = `<h3>${rateChart.label} &mdash; same round, log scale so both ends of a 3-order-of-magnitude gap stay visible</h3><div class="chart-svg-box" id="rate-svg-box"></div>`;
  document.getElementById('rate-chart-row').appendChild(rateCard);

  (function drawRateChart() {
    const width = 640, height = 170;
    const pad = {top: 16, right: 16, bottom: 32, left: 46};
    const innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom;
    const groups = rateChart.groups;
    const allV = groups.flatMap(g => [g.normal, g.attack]).filter(v => v > 0);
    const lo = Math.log10(Math.min(...allV) * 0.6), hi = Math.log10(Math.max(...allV) * 1.4);
    function y(v) { return pad.top + innerH - ((Math.log10(Math.max(v, 0.001)) - lo) / (hi - lo)) * innerH; }
    const nColor = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--normal').trim();
    const aColor = getComputedStyle(document.querySelector('.viz-root')).getPropertyValue('--attack').trim();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    [0, 0.25, 0.5, 0.75, 1].forEach(frac => {
      const gy = pad.top + innerH - frac * innerH;
      const gl = document.createElementNS(svg.namespaceURI, 'line');
      gl.setAttribute('x1', pad.left); gl.setAttribute('x2', width - pad.right);
      gl.setAttribute('y1', gy); gl.setAttribute('y2', gy);
      gl.setAttribute('class', 'bar-gridline');
      svg.appendChild(gl);
      const lbl = document.createElementNS(svg.namespaceURI, 'text');
      lbl.setAttribute('x', pad.left - 6); lbl.setAttribute('y', gy + 3);
      lbl.setAttribute('text-anchor', 'end'); lbl.setAttribute('class', 'bar-axis-label');
      lbl.textContent = fmtNum(Math.pow(10, lo + frac * (hi - lo)));
      svg.appendChild(lbl);
    });

    const groupW = innerW / groups.length, barW = 34, gap = 18;
    groups.forEach((g, i) => {
      const cx = pad.left + i * groupW + groupW / 2;
      const x0 = cx - barW - gap / 2, x1 = cx + gap / 2;
      [[x0, g.normal, nColor], [x1, g.attack, aColor]].forEach(([x, v, color]) => {
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
        val.textContent = fmtNum(v);
        svg.appendChild(val);
      });
      const cat = document.createElementNS(svg.namespaceURI, 'text');
      cat.setAttribute('x', cx); cat.setAttribute('y', height - 10);
      cat.setAttribute('text-anchor', 'middle'); cat.setAttribute('class', 'bar-cat-label');
      cat.textContent = g.name;
      svg.appendChild(cat);
    });
    document.getElementById('rate-svg-box').appendChild(svg);
  })();

  // ---- stats table ----
  const rows = [
    ['Known-good peer?', 'yes (192.168.0.40)', 'no (192.168.0.1, never seen in normal traffic)', true],
    ['Distinct function codes used', `${s.normal_fc.length} (fc ${s.normal_fc.join(', ')})`, `${s.round1_distinct_fc} (${s.round1_fc_list[0]}–${s.round1_fc_list[s.round1_fc_list.length-1]})`, true],
    ['... outside the standard public set', '0', `${s.round1_nonstd_fc.length} (fc ${s.round1_nonstd_fc.slice(0,6).join(', ')}, ...)`, true],
    ['Write-type commands sent', 'none (0.0% of traffic)', `fc ${s.round1_writes_sent.join(', ')} — and ACCEPTED`, true],
    ['Largest read quantity requested', `${s.normal_maxq}`, `${s.round1_maxq}`, false],
    ['Largest reply size', `${s.normal_max_resp} bytes`, `${s.round1_max_resp} bytes`, true],
    ['New TCP connections', `${s.normal_conn_per_min}/min (1 every ${(60/s.normal_conn_per_min).toFixed(1)}s)`, `${s.round1_n_connections} opened within ${s.round1_dur_ms} ms`, true],
    ['Packets in this exchange', `—`, `${s.round1_n_pkts_total} packets in ${s.round1_dur_ms} ms (≈${Math.round(s.round1_n_pkts_total/(s.round1_dur_ms/1000))} pkt/s)`, true],
    ['Modbus requests sent by this host', `${s.normal_rate.toFixed(1)} req/s (session average)`, `${s.round1_n_req} requests in ${s.round1_dur_ms} ms`, true],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
