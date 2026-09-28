"""
Fourth in the per-attack-type series for Smart Grid (after function-code
scan, address scan, device identification attack - see the Smart Grid
normal-behavior-baseline memory), applied to "naive sensor read"
(attack_specific == 4).

Unlike device identification, this attack's request TYPE (reading a value)
DOES have a direct normal-traffic counterpart, so reusing
extract_normal_pair() from packet_compare_smartgrid_fcscan.py is the right
call here (see that memory for why it was wrong for device-ID). The
difference from normal is the QUANTITY parameter, not the function code:
normal always reads quantity=1; this attack cycles through all 4 standard
read codes (fc1/2/3/4) requesting the PROTOCOL MAXIMUM each time - 2000 for
coil-type reads (fc1/fc2), 125 for register-type reads (fc3/fc4) - a
brute-force "dump everything in one request" pattern, not a subtler evasion.

Structure, verified from the raw data: 18 bursts of exactly 10.0s each (44
reads = 11 full fc1->fc2->fc4->fc3 cycles per burst), scattered irregularly
across the whole session (t=133.3s-5357.1s). The very last burst also
contains a short, distinct write test: 15 rapid Write Single Coil (fc5)
requests to the same address in under 1 second, all ACCEPTED (echoed back) -
the same "unauthorized command" pattern as the function-code scan's fc5, but
tacked onto the end of a read campaign rather than its own attack.

Notable and verified (not assumed) methodology finding: 30.8% of this
attack's 6,221 labeled rows are NOT attack behavior at all - they are
192.168.0.1's completely ordinary background TCP/DATA/WEBSOCKET traffic with
its legitimate partner 192.168.0.111 (statistically identical in packet size
to that same traffic when labeled "normal" elsewhere in the session), simply
swept up because attack_specific appears to label ALL of a host's traffic
during its active window, not just the malicious packets. Checked: this
co-mingling is negligible for the first 3 attacks in this series (0.0-0.6%
of their rows) but substantial here - worth checking again for the remaining
attacks rather than assuming it's always small.

Also found: 3 single-packet connection attempts from 192.168.0.1 to hosts it
doesn't normally touch (192.168.0.21/22/40) around t=2426s, each immediately
RST-ACK'd (refused) - unlike address scan's SYN-ACK'd (accepted) probes to
192.168.0.31. A minor, failed side-activity, not the main signature.

Output goes into data_visualisation/smartgrid_naive_sensor_read/ (all
filenames get the optional --tag suffix so earlier results are not
overwritten): packets.json, stats.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_naivesensorread.py 2>&1 | tee data_visualisation/smartgrid_naive_sensor_read/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv
from packet_compare_smartgrid_fcscan import extract_normal_pair, pack

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_naive_sensor_read"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
TARGET_IP = "192.168.0.31"
PARTNER_IP = "192.168.0.111"
BURST_GAP_SEC = 5.0


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
    a4 = df[df.attack_specific == 4].sort_values("row")
    mb = a4[a4.protocol == "MODBUS"].sort_values("frame_time_relative").reset_index(drop=True)

    def find_pair(fc, before_row_limit=None):
        sub = mb if before_row_limit is None else mb[mb.row < before_row_limit]
        for i in range(len(sub) - 1):
            if sub.iloc[i].ip_src == ATTACKER_IP and sub.iloc[i].modbus_func_code == fc \
               and sub.iloc[i + 1].ip_src == TARGET_IP:
                return {"request": pack(sub.iloc[i]), "response": pack(sub.iloc[i + 1])}
        return None

    max_read = find_pair(1)   # fc1, quantity 2000
    write_burst = find_pair(5)  # fc5, tacked onto the final burst

    rejected = a4[(a4.ip_src == ATTACKER_IP) & (a4.protocol == "TCP") & (a4.tcp_flags == "0x0002")
                  & a4.ip_dst.isin(["192.168.0.21", "192.168.0.22", "192.168.0.40"])].sort_values("row")
    rejected_probe = None
    if len(rejected):
        syn_row = rejected.iloc[0]
        rst_candidates = df[(df.row > syn_row.row) & (df.row <= syn_row.row + 3)
                            & (df.ip_src == syn_row.ip_dst) & (df.ip_dst == ATTACKER_IP)]
        rejected_probe = {"syn": pack(syn_row),
                          "rst": pack(rst_candidates.iloc[0]) if len(rst_candidates) else None}

    return {"max_quantity_read": max_read, "write_burst": write_burst, "rejected_probe": rejected_probe}


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a4 = df[df.attack_specific == 4]
    mb = a4[a4.protocol == "MODBUS"]
    req = mb[mb.ip_src == ATTACKER_IP]
    read_req = req[req.modbus_func_code.isin([1, 2, 3, 4])]
    write_req = req[req.modbus_func_code == 5]

    # background noise: this attacker's traffic with its own legitimate
    # partner, statistically identical to that same traffic when labeled
    # normal elsewhere in the session - not attack behavior, just co-labeled.
    noise = a4[a4.protocol.isin(["DATA", "WEBSOCKET"])
              & a4.ip_src.isin([ATTACKER_IP, PARTNER_IP]) & a4.ip_dst.isin([ATTACKER_IP, PARTNER_IP])]
    normal_partner_traffic = n[n.protocol.isin(["DATA", "WEBSOCKET"])
                               & n.ip_src.isin([ATTACKER_IP, PARTNER_IP]) & n.ip_dst.isin([ATTACKER_IP, PARTNER_IP])]

    # burst structure
    req_sorted = req.sort_values("frame_time_relative")
    t = req_sorted.frame_time_relative.to_numpy()
    gaps = np.diff(t)
    cuts = np.where(gaps > BURST_GAP_SEC)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(t) - 1]
    burst_sizes = (ends - starts + 1)

    normal_read_qtys = []
    n_req = n[(n.protocol == "MODBUS") & (n.tcp_len == 12)]
    for d in n_req.modbus_data:
        _, q = decode_qty(d)
        if q is not None:
            normal_read_qtys.append(q)

    return {
        "session_duration_sec": round(df.frame_time_relative.max(), 1),
        "attack_total_rows": int(len(a4)),
        "attack_noise_rows": int(len(noise)),
        "attack_noise_pct": round(len(noise) / len(a4) * 100, 1),
        "noise_avg_len": round(float(noise[noise.protocol == "DATA"].tcp_len.mean()), 1) if len(noise[noise.protocol == "DATA"]) else None,
        "normal_partner_avg_len": round(float(normal_partner_traffic[normal_partner_traffic.protocol == "DATA"].tcp_len.mean()), 1) if len(normal_partner_traffic[normal_partner_traffic.protocol == "DATA"]) else None,
        "normal_max_qty": max(normal_read_qtys) if normal_read_qtys else None,
        "attack_max_qty_fc1_fc2": 2000,
        "attack_max_qty_fc3_fc4": 125,
        "n_bursts": int(len(starts)),
        "reads_per_burst_median": int(np.median(burst_sizes)),
        "burst_duration_sec": 10.0,
        "attack_window_start_sec": round(float(a4.frame_time_relative.min()), 1),
        "attack_window_end_sec": round(float(a4.frame_time_relative.max()), 1),
        "n_read_requests": int(len(read_req)),
        "n_write_requests": int(len(write_req)),
        "normal_write_count": 0,
        "n_rejected_probe_targets": int(a4[(a4.ip_src == ATTACKER_IP) & (a4.protocol == "TCP")
                                            & (a4.tcp_flags == "0x0002")
                                            & a4.ip_dst.isin(["192.168.0.21", "192.168.0.22", "192.168.0.40"])].ip_dst.nunique()),
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
    parser = argparse.ArgumentParser(description="Compare a normal exchange vs. the naive-sensor-read attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting normal packet pair (reused from packet_compare_smartgrid_fcscan - apt here, same request type)...")
    normal_pair = extract_normal_pair(df)
    print("Extracting naive-sensor-read examples (max-quantity read, write burst, rejected probe)...")
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

    print(f"Naive-sensor-read summary: {stats['n_bursts']} bursts of ~{stats['reads_per_burst_median']} reads "
          f"({stats['burst_duration_sec']}s each), {stats['n_read_requests']} total read requests at max "
          f"quantity, {stats['n_write_requests']} write requests tacked onto the last burst, "
          f"{stats['attack_noise_pct']}% of labeled rows are unrelated background traffic")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Naive Sensor Read Diff</title>
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

  footer { border-top: 1px solid var(--border); padding-top: 18px; }
  footer p { font-size: 12px; }
</style>

<div class="viz-root">
<div class="wrap">

  <header>
    <span class="eyebrow">Smart Grid &middot; ICS-SimLab capture &middot; real packets, not synthetic</span>
    <h1 style="margin-top:8px">Reading everything, all at once, over and over</h1>
    <p style="margin-top:10px">"Naive sensor read" cycles through all 4 standard read function codes
      (fc1, fc2, fc3, fc4) but always requests the Modbus <b>protocol maximum</b> quantity &mdash;
      2000 for coil-type reads, 125 for register-type reads &mdash; instead of the single value a real
      poll ever asks for. It runs as <b id="bursts-inline"></b> separate 10-second bursts scattered
      across the whole session, and the very last burst also tacks on a short, distinct write test.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>Not everything labeled "attack" here is attack behavior</h2>
      <p style="margin-top:6px">Checked directly against the raw data, not assumed: a large share of
        this attack's labeled rows are ordinary background chatter that happens to be swept up by the
        label, not actual malicious traffic.</p>
    </div>
    <div class="shared-target-banner">
      Of the <b id="noise-total-inline"></b> rows labeled <code class="mono">attack_specific&nbsp;==&nbsp;4</code>,
      <b id="noise-pct-inline"></b> (<b id="noise-rows-inline"></b> rows) are
      192.168.0.1's completely ordinary DATA/WEBSOCKET traffic with its regular partner 192.168.0.111 &mdash;
      average packet length <b id="noise-len-inline"></b> bytes, statistically indistinguishable from that
      same host pair's traffic when it's labeled normal elsewhere in the session
      (<b id="normal-len-inline"></b> bytes average). The dataset's <code class="mono">attack_specific</code>
      label appears to tag <em>all</em> of a host's traffic during its active window, not just the
      malicious packets. This co-mingling was negligible for the first 3 attacks in this series
      (0.0&ndash;0.6% of their rows) but is substantial here &mdash; a reminder to check this per attack
      rather than assume it away. Every finding below is drawn only from the genuinely malicious rows.
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; a real value poll</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; naive sensor read</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Statistics: this attack type vs. the session's normal baseline</h2>
      <p style="margin-top:6px">Read quantities are the protocol's own documented maximums, not
        arbitrary numbers &mdash; this is a brute-force "dump everything in one request" pattern, not a
        subtle one.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal baseline</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Naive sensor read</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Naive sensor read</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows labeled
      <code class="mono">attack_specific == 4</code>. PDU <code class="mono">0x0100000007d0</code>
      decodes as address&nbsp;0, quantity&nbsp;<b>2000</b> (fc1's protocol maximum for coil-type reads);
      <code class="mono">0x030000007d</code> decodes as address&nbsp;0, quantity&nbsp;<b>125</b> (fc3's
      protocol maximum for register-type reads). Companion page to
      <code class="mono">packet_compare_smartgrid_fcscan.py</code>,
      <code class="mono">packet_compare_smartgrid_addressscan.py</code> and
      <code class="mono">packet_compare_smartgrid_deviceid.py</code>.</p>
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
  document.getElementById('bursts-inline').textContent = s.n_bursts;
  document.getElementById('noise-total-inline').textContent = s.attack_total_rows.toLocaleString();
  document.getElementById('noise-pct-inline').textContent = s.attack_noise_pct + '%';
  document.getElementById('noise-rows-inline').textContent = s.attack_noise_rows.toLocaleString();
  document.getElementById('noise-len-inline').textContent = s.noise_avg_len != null ? s.noise_avg_len : 'n/a';
  document.getElementById('normal-len-inline').textContent = s.normal_partner_avg_len != null ? s.normal_partner_avg_len : 'n/a';

  // ---- normal column ----
  const nCol = document.getElementById('normal-col');
  const np = DATA.normal_pair;
  let normalHtml = '';
  if (np) {
    normalHtml += `<div><h3 style="margin-bottom:8px">A routine value poll<span class="flag" style="background:var(--normal-bg); color:var(--normal)">quantity = 1</span></h3>` +
      frame('REQUEST', np.request, false,
        `Read Coils (fc1), address 0, quantity <b>1</b> &mdash; every normal read in this session asks for exactly one value.`) +
      frame('RESPONSE', np.response, false,
        `A single coil's status, nothing more.`) +
      `</div>`;
  }
  nCol.innerHTML = normalHtml;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml = '';
  if (ex.max_quantity_read) {
    attackHtml += `<div><h3 style="margin-bottom:8px">1. Maximum-quantity read<span class="flag">quantity = 2000</span></h3>` +
      frame('REQUEST', ex.max_quantity_read.request, true,
        `Read Coils (fc1), address 0, quantity <b>2000</b> &mdash; the Modbus protocol's own documented
         maximum for coil-type reads. fc2 is probed the same way; fc3/fc4 (register-type reads) request
         the protocol maximum of 125 instead. Repeated in 18 bursts of ~10s each across the whole session.`) +
      frame('RESPONSE', ex.max_quantity_read.response, true,
        `Same fixed 259-byte response shape every time, regardless of which of the 4 function codes asked.`) +
      `</div>`;
  }
  if (ex.write_burst) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. Write burst tacked onto the last cycle<span class="flag">fc5, accepted</span></h3>` +
      frame('REQUEST', ex.write_burst.request, true,
        `Write Single Coil (fc5), address 9, value 0xff00 (ON) &mdash; 15 identical requests in under 1
         second, right at the end of the final read burst. Same "unauthorized command" pattern as the
         function-code scan's fc5, but a short coda here rather than the main signature.`) +
      frame('RESPONSE', ex.write_burst.response, true,
        `Echoed back &mdash; ACCEPTED. 0 writes occur in normal traffic.`) +
      `</div>`;
  }
  if (ex.rejected_probe && ex.rejected_probe.syn) {
    attackHtml += `<div><h3 style="margin-bottom:8px">3. A side-probe that got refused<span class="flag">SYN &rarr; RST-ACK</span></h3>` +
      frame('SYN', ex.rejected_probe.syn, true,
        `A brief attempt to open a connection to a host (${ex.rejected_probe.syn.ip_dst}) this attacker
         never otherwise talks to &mdash; ${s.n_rejected_probe_targets} such hosts touched around the
         same moment.`) +
      (ex.rejected_probe.rst ? frame('RST-ACK', ex.rejected_probe.rst, true,
        `Connection actively refused. Unlike the address scan's targets, which answered with SYN-ACK,
         this host rejects the attempt outright &mdash; a minor, failed side-activity, not the main
         signature of this attack.`) : '') +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- bar charts ----
  function fmtNum(v) { return v >= 1000 ? Math.round(v).toLocaleString() : (Number.isInteger(v) ? v : v.toFixed(1)); }

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
    {label: 'Read quantity requested (fc1/fc2)', normal: s.normal_max_qty || 1, attack: s.attack_max_qty_fc1_fc2, logScale: true},
    {label: 'Read quantity requested (fc3/fc4)', normal: s.normal_max_qty || 1, attack: s.attack_max_qty_fc3_fc4, logScale: true},
    {label: 'Write requests (fc5) sent', normal: s.normal_write_count, attack: s.n_write_requests},
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
    ['Read requests, quantity per request', '1', `2000 (fc1/fc2), 125 (fc3/fc4)`, true],
    ['Total read requests', '—', `${s.n_read_requests}`, true],
    ['Write requests (fc5)', '0', `${s.n_write_requests}`, true],
    ['Bursts (10s each, >5s gap between)', '—', `${s.n_bursts}`, true],
    ['Reads per burst (median)', '—', `${s.reads_per_burst_median}`, false],
    ['Attack window span', '—', `${(s.attack_window_end_sec - s.attack_window_start_sec).toFixed(1)}s of a ${s.session_duration_sec}s session`, false],
    ['Rows labeled attack_specific=4', '—', `${s.attack_total_rows.toLocaleString()}`, false],
    ['...of which unrelated background traffic', '—', `${s.attack_noise_rows.toLocaleString()} (${s.attack_noise_pct}%)`, true],
    ['Side-probe targets refused (SYN &rarr; RST-ACK)', '—', `${s.n_rejected_probe_targets}`, false],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
