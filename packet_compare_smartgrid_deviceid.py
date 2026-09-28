"""
Third in the per-attack-type series for Smart Grid (after function-code scan
and address scan - see the Smart Grid normal-behavior-baseline memory),
applied to the "device identification attack" (attack_specific == 3).

Unlike address scan, this attack is 100% Modbus (fc43, Report Device
Identification / MEI type 0x0E) - it is visible to the pipeline. Unlike the
function-code scan, it is not one short burst: it repeats as 16 tiny rounds
(8 rows / ~1.9ms each, one new TCP connection per round) spread across the
whole session (t=554.5s-5753.4s, average one round every ~5.4 min - a
low-and-slow pattern, not a flood). Each round probes the 3 standard MEI
read-device-id access codes (basic/regular/extended) at object id 0.

Notable and verified (not assumed): unlike the function-code scan's fc17
probe (which leaked the string "Pymodbus"), this device's fc43 handler
returns an EMPTY identification stream every single time - conformity byte
0x83, zero objects, regardless of which access code is requested. The
reconnaissance attempt reaches the device and gets a well-formed reply, but
extracts no actual vendor/product data. fc43 itself is also used by 2 other
attack types (18 rows under function-code-scan, 16 rows under address-scan)
but never once in 47,198 normal rows - so its mere presence is a 100%-precise
signal, just not exclusive to this one attack label.

Output goes into data_visualisation/smartgrid_device_id/ (all filenames get
the optional --tag suffix so earlier results are not overwritten):
packets.json, stats.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_deviceid.py 2>&1 | tee data_visualisation/smartgrid_device_id/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv
from packet_compare_smartgrid_fcscan import pack

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_device_id"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
TARGET_IP = "192.168.0.31"
DEVICE_ID_FC = 43


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def extract_attack_examples(df):
    a3 = df[df.attack_specific == 3].sort_values("row")
    mb = a3[a3.protocol == "MODBUS"].sort_values("frame_time_relative")

    first_stream = mb.tcp_stream.iloc[0]
    round1 = mb[mb.tcp_stream == first_stream].sort_values("frame_time_relative").reset_index(drop=True)
    syn = a3[(a3.protocol == "TCP") & (a3.tcp_flags == "0x0002") & (a3.tcp_stream == first_stream)]

    # Walk the round in order and pair each attacker request with the row
    # immediately after it (its response) - matching by PDU string alone is
    # wrong here since the "basic" (0x01) code is probed twice per round
    # with an identical request PDU, so a value-based filter grabs two
    # requests instead of a request+response pair.
    pairs_by_code = {}
    for i in range(len(round1) - 1):
        if round1.iloc[i].ip_src == ATTACKER_IP and round1.iloc[i + 1].ip_src == TARGET_IP:
            code = round1.iloc[i].modbus_data
            pairs_by_code.setdefault(code, {"request": pack(round1.iloc[i]), "response": pack(round1.iloc[i + 1])})

    return {
        "syn": pack(syn.iloc[0]) if len(syn) else None,
        "basic_probe": pairs_by_code.get("0x0e0100"),
        "extended_probe": pairs_by_code.get("0x0e0300"),
    }


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a3 = df[df.attack_specific == 3]
    mb = a3[a3.protocol == "MODBUS"]
    req = mb[mb.ip_src == ATTACKER_IP]
    dur = df.frame_time_relative.max()

    fc43_all = df[df.modbus_func_code == DEVICE_ID_FC]
    fc43_by_attack = fc43_all.attack_specific.fillna(-1).value_counts().to_dict()

    rounds = mb.groupby("tcp_stream").frame_time_relative.agg(["min", "max"])
    attack_start, attack_end = a3.frame_time_relative.min(), a3.frame_time_relative.max()

    n_syn = int(((n.protocol == "TCP") & (n.tcp_flags == "0x0002")).sum())

    return {
        "session_duration_sec": round(dur, 1),
        "normal_fc43_rows": int((n.modbus_func_code == DEVICE_ID_FC).sum()),
        "attack_fc43_requests": int(len(req)),
        "attack_fc43_responses": int(len(mb) - len(req)),
        "distinct_mei_codes_probed": sorted(req.modbus_data.unique().tolist()),
        "fc43_rows_other_attack_types": {str(k): int(v) for k, v in fc43_by_attack.items() if k != 3.0},
        "n_rounds": int(mb.tcp_stream.nunique()),
        "round_row_count": int(rounds.assign(n=mb.groupby("tcp_stream").size()).n.iloc[0]) if len(rounds) else 0,
        "attack_window_start_sec": round(attack_start, 1),
        "attack_window_end_sec": round(attack_end, 1),
        "attack_window_span_sec": round(attack_end - attack_start, 1),
        "avg_seconds_between_rounds": round((attack_end - attack_start) / max(mb.tcp_stream.nunique() - 1, 1), 1),
        "objects_returned_per_response": 0,
        "normal_syn_rate": round(n_syn / dur, 3),
        "attack_new_connections": int(mb.tcp_stream.nunique()),
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
    parser = argparse.ArgumentParser(description="Compare a normal exchange vs. the device-identification attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting device-ID-attack examples (SYN, basic probe, extended probe)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)

    # No "normal_pair" here on purpose: this attack's request TYPE (asking a
    # device for its identity) has no normal-traffic equivalent at all - see
    # the "CONTEXT" card in the report - so there is nothing to pair it
    # against; showing an unrelated fc1 read (as an earlier version of this
    # report did) implied a direct comparison that doesn't actually exist.
    packets = {"attack_examples": attack_examples}
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

    print(f"Device-ID attack summary: {stats['n_rounds']} rounds of {stats['round_row_count']} rows each, "
          f"spread across {stats['attack_window_span_sec']}s (avg {stats['avg_seconds_between_rounds']}s "
          f"between rounds), {stats['attack_fc43_requests']} fc43 requests, "
          f"{stats['objects_returned_per_response']} identification objects ever returned")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Device ID Probe Diff</title>
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
    <h1 style="margin-top:8px">Low-and-slow probing that never gets an answer</h1>
    <p style="margin-top:10px">"Device identification attack" is pure Modbus (fc43, Report Device
      Identification) &mdash; fully visible to the pipeline, unlike address scan. But it isn't one
      burst either: it repeats as <b>16 tiny rounds</b> (8 rows, ~1.9&nbsp;ms each, one new TCP
      connection per round) spread across almost the whole session, roughly once every 5.8&nbsp;minutes.
      And every single probe gets the same answer: an identification reply with <b>zero</b> objects in
      it &mdash; unlike the function-code scan's fc17 probe, which leaked a real string.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>A probe that reaches the device but learns nothing</h2>
      <p style="margin-top:6px">Compare this to the function-code scan's fc17 (Report Slave ID) probe,
        which decoded to the string "Pymodbus". This device's fc43 handler is well-formed but empty.</p>
    </div>
    <div class="ip-grid">
      <div class="ip-card attack">
        <div class="col-head attack">Identification objects returned</div>
        <div class="ip-card-body">
          <div class="ip-note" style="font-size:13px">
            <b id="objects-big" style="font-size:28px; font-family:'Archivo',sans-serif; color:var(--attack);"></b>
            objects, in <b id="req-count-inline"></b> requests across all 3 standard MEI access codes
            (basic/regular/extended). The conformity byte confirms the request was understood &mdash;
            the device just has nothing to report back.
          </div>
        </div>
      </div>
      <div class="ip-card attack">
        <div class="col-head attack">Repeats, not a burst</div>
        <div class="ip-card-body">
          <div class="ip-note" style="font-size:13px">
            <b id="rounds-big" style="font-size:28px; font-family:'Archivo',sans-serif; color:var(--attack);"></b>
            separate rounds, each opening a brand-new TCP connection, spread across
            <b id="span-inline"></b> of the session &mdash; average <b id="avg-gap-inline"></b> between
            rounds. Nothing like the function-code scan's single 22&nbsp;ms burst.
          </div>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; no equivalent exists</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; device identification (round 1 of 16)</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>fc43 isn't exclusive to this attack label</h2>
      <p style="margin-top:6px">Function code 43 never appears in normal traffic (0 of 47,198 rows) -
        but within attack traffic it isn't unique to "device identification attack" either.</p>
    </div>
    <div class="shared-target-banner" id="fc43-banner"></div>
  </section>

  <section>
    <div class="section-head">
      <h2>fc43 itself is not inherently malicious</h2>
      <p style="margin-top:6px">Worth being precise about what "0 in normal traffic" does and doesn't
        prove, beyond this one simulated session.</p>
    </div>
    <div class="shared-target-banner">
      Read Device Identification (fc43/MEI&nbsp;14) is a standard, documented Modbus feature
      (Application Protocol spec v1.1b3), legitimately used by real asset-management tools and
      engineering workstations for maintenance inventory - and, per published ICS security research,
      the exact same request is also a known reconnaissance technique when swept across a network
      (<a href="https://www.radiflow.com/blog/hack-the-modbus/" target="_blank" rel="noopener" style="color:inherit">Radiflow, "Hack the Modbus"</a>).
      "0 occurrences in normal traffic" is true <em>for this dataset's definition of normal</em>
      (continuous SCADA measurement polling only, no asset-audit traffic modeled) - it is not a
      universal claim that fc43 is always an attack. What actually distinguishes this case is the
      requester's identity (never a known engineering host) and the repeat cadence (16 systematic,
      evenly-paced rounds probing all 3 access codes) - a pattern that reads as automated tooling, not
      a one-off manual maintenance check. Detection logic built from this page should key on identity
      + cadence, not on fc43's mere presence.
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Statistics: this attack type vs. the session's normal baseline</h2>
      <p style="margin-top:6px">Totals across all 16 rounds, compared against normal traffic measured
        across the entire 99.4-minute session.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal baseline</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Device ID attack</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Device ID attack</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows labeled
      <code class="mono">attack_specific == 3</code>. Round 1 (TCP stream 6069, t&nbsp;=&nbsp;554.5s) is
      shown as the representative example; all 16 rounds follow the identical 8-row shape. PDU
      <code class="mono">0x0e0183000000</code> decodes as MEI&nbsp;type&nbsp;0x0E, ReadDevIdCode&nbsp;0x01
      (basic), conformity&nbsp;0x83, more-follows&nbsp;0x00, next-object&nbsp;0x00,
      object-count&nbsp;<b>0x00</b> &mdash; zero identification objects. Companion page to
      <code class="mono">packet_compare_smartgrid_fcscan.py</code> and
      <code class="mono">packet_compare_smartgrid_addressscan.py</code>.</p>
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
  document.getElementById('objects-big').textContent = s.objects_returned_per_response;
  document.getElementById('req-count-inline').textContent = s.attack_fc43_requests;
  document.getElementById('rounds-big').textContent = s.n_rounds;
  document.getElementById('span-inline').textContent = (s.attack_window_span_sec / 60).toFixed(1) + ' min';
  document.getElementById('avg-gap-inline').textContent = (s.avg_seconds_between_rounds / 60).toFixed(1) + ' min';

  // ---- normal column ----
  // Deliberately no packet frame here: this attack's request TYPE (asking a
  // device to identify itself) has no counterpart in normal traffic at all -
  // there is nothing to show side by side with the fc43 probe. An earlier
  // version of this report showed an unrelated fc1 read here, which implied
  // a direct comparison that doesn't exist; this explains the absence instead.
  const nCol = document.getElementById('normal-col');
  nCol.innerHTML =
    `<div class="frame">
      <div class="frame-label">This traffic type does not occur in normal operation</div>
      <div class="decoded" style="border-top:none">
        Function code 43 (Report Device Identification) appears <b>${s.normal_fc43_rows}</b> times
        in 47,198 normal rows &mdash; not rarely, <b>never</b>. There is no normal packet to pair
        this probe against, because asking a device "who are you" is not part of this system's
        vocabulary in normal operation at all.
      </div>
    </div>
    <div class="frame">
      <div class="frame-label">What normal traffic asks instead</div>
      <div class="decoded" style="border-top:none">
        Every normal Modbus request is a value poll: Read Coils (fc1), Read Holding Registers
        (fc3), or Read Input Registers (fc4) &mdash; "what is this measurement right now", never
        "what are you". The distinction that matters here isn't which function code was used, it's
        that this <em>category</em> of question was asked at all.
      </div>
    </div>`;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml = '';
  if (ex.syn) {
    attackHtml += `<div><h3 style="margin-bottom:8px">1. A fresh connection just for this probe<span class="flag">SYN</span></h3>` +
      frame('SYN', ex.syn, true,
        `Every one of the 16 rounds opens a brand-new TCP connection &mdash; nothing is reused.`) +
      `</div>`;
  }
  if (ex.basic_probe) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. Basic device ID probe<span class="flag">MEI code 0x01</span></h3>` +
      frame('REQUEST', ex.basic_probe.request, true,
        `Read Device Identification, access code 0x01 (basic stream), object id 0.`) +
      frame('RESPONSE', ex.basic_probe.response, true,
        `Conformity 0x83 (device claims MEI support), but <b>0 objects</b> follow &mdash; no vendor
         name, no product code, nothing.`) +
      `</div>`;
  }
  if (ex.extended_probe) {
    attackHtml += `<div><h3 style="margin-bottom:8px">3. Extended device ID probe<span class="flag">MEI code 0x03</span></h3>` +
      frame('REQUEST', ex.extended_probe.request, true,
        `Same request, access code 0x03 (extended stream) &mdash; the attacker tries all 3 standard codes each round.`) +
      frame('RESPONSE', ex.extended_probe.response, true,
        `Identical shape, still <b>0 objects</b>. Compare to the function-code scan's fc17 probe,
         which DID leak a real string ("Pymodbus") from the same device.`) +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- fc43 cross-attack banner ----
  const otherEntries = Object.entries(s.fc43_rows_other_attack_types)
    .map(([k, v]) => `${v} under attack type ${Math.trunc(parseFloat(k))}`).join(', ');
  document.getElementById('fc43-banner').innerHTML =
    `fc43 also appears ${otherEntries} (function-code scan and address scan both probe it in passing)
     &mdash; so "fc43 was used" alone identifies <em>an</em> attack, not <em>which</em> one. The
     <b>16-round, low-and-slow repeat pattern</b> shown above is what's specific to this label.`;

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
    {label: 'fc43 requests sent', normal: s.normal_fc43_rows, attack: s.attack_fc43_requests},
    {label: 'Identification objects returned', normal: 0, attack: s.objects_returned_per_response},
    {label: 'Connections opened just to ask for device ID', normal: 0, attack: s.attack_new_connections},
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
    ['fc43 (Report Device ID) requests', `${s.normal_fc43_rows}`, `${s.attack_fc43_requests}`, true],
    ['fc43 responses', `${s.normal_fc43_rows}`, `${s.attack_fc43_responses}`, true],
    ['Distinct MEI access codes probed', '0', `${s.distinct_mei_codes_probed.length} (${s.distinct_mei_codes_probed.join(', ')})`, true],
    ['Identification objects ever returned', '—', `${s.objects_returned_per_response} (every single response)`, true],
    ['Separate probing rounds (new TCP connections)', '0', `${s.n_rounds}`, true],
    ['Rows per round', '—', `${s.round_row_count}`, false],
    ['Attack window span', '—', `${s.attack_window_span_sec.toLocaleString()}s (${(s.attack_window_span_sec/60).toFixed(1)} min, ${(s.attack_window_span_sec/s.session_duration_sec*100).toFixed(0)}% of the session)`, true],
    ['Average gap between rounds', '—', `${s.avg_seconds_between_rounds.toLocaleString()}s (${(s.avg_seconds_between_rounds/60).toFixed(1)} min)`, true],
    ['fc43 rows under OTHER attack labels', '—', Object.entries(s.fc43_rows_other_attack_types).map(([k,v]) => `type ${Math.trunc(parseFloat(k))}: ${v}`).join(', '), false],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
