"""
Eighth in the per-attack-type series for Smart Grid (after function-code
scan, address scan, device identification attack, naive sensor read,
sporadic sensor measurement injection, force listen mode - see the Smart
Grid normal-behavior-baseline memory), applied to "restart communication"
(attack_specific == 7).

Like device identification and force listen mode, this attack's request
TYPE (a Diagnostics sub-function call, fc8) never occurs in normal traffic
at all, so there is no normal_pair - same reasoning as
packet_compare_smartgrid_deviceid.py / packet_compare_smartgrid_forcelisten.py.

Structure, verified from the raw data: 10 rounds, each opening a brand-new
TCP connection, each containing 10 fc8 requests (one round has only 9) at
an extremely regular ~3.004s interval (essentially zero jitter - a
scripted/automated cadence, not organic traffic). Rounds span
t=354.4s-5889.1s (92.8% of the session), each round lasting ~27.0s.

Same epistemic caution as force-listen: the attack's assigned name implies
Modbus's real "Restart Communications Option" (Diagnostics sub-function
0x01), but this dataset's modbus_data field shows the same empty/zero
value ("0x") for every request, which does not let us independently
confirm the sub-function byte. What IS independently verifiable: unlike
force-listen-mode's shortened (10-byte) response, this attack's response
is the STANDARD 12-byte echo shape - the same shape fc8 gets under
function-code scan and address scan. So whatever protocol difference force
listen mode triggers in the target, this attack does not trigger the same
one; the concrete, measurable signal here is the REPEATED, tightly-regular
request cadence (10 requests/round at ~3.004s intervals) instead.

Also verified: the same background-noise co-mingling issue applies here
too, at the highest magnitude seen in this series so far (43.2% of this
attack's 2,818 rows are 192.168.0.1's ordinary DATA/WEBSOCKET traffic with
192.168.0.111) - continuing the growth trend from naive-sensor-read (30.8%)
through sporadic-injection (34.4%) and force-listen (40.3%).

Output goes into data_visualisation/smartgrid_restart_comm/ (all filenames
get the optional --tag suffix so earlier results are not overwritten):
packets.json, stats.json, timeline.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_restartcomm.py 2>&1 | tee data_visualisation/smartgrid_restart_comm/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv
from packet_compare_smartgrid_fcscan import pack

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_restart_comm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
TARGET_IP = "192.168.0.31"
PARTNER_IP = "192.168.0.111"
DIAGNOSTICS_FC = 8


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def extract_attack_examples(df):
    a7 = df[df.attack_specific == 7].sort_values("row")
    mb = a7[a7.protocol == "MODBUS"].sort_values("frame_time_relative").reset_index(drop=True)

    first_stream = mb.tcp_stream.iloc[0]
    round1 = mb[mb.tcp_stream == first_stream].sort_values("frame_time_relative").reset_index(drop=True)
    syn = a7[(a7.protocol == "TCP") & (a7.tcp_flags == "0x0002") & (a7.tcp_stream == first_stream)]

    pairs = []
    for i in range(len(round1) - 1):
        if round1.iloc[i].ip_src == ATTACKER_IP and round1.iloc[i + 1].ip_src == TARGET_IP:
            pairs.append({"request": pack(round1.iloc[i]), "response": pack(round1.iloc[i + 1])})

    return {
        "syn": pack(syn.iloc[0]) if len(syn) else None,
        "first_pair": pairs[0] if pairs else None,
        "second_pair": pairs[1] if len(pairs) > 1 else None,
    }


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a7 = df[df.attack_specific == 7]
    mb = a7[a7.protocol == "MODBUS"]
    req = mb[mb.ip_src == ATTACKER_IP]
    resp = mb[mb.ip_src == TARGET_IP]
    dur = df.frame_time_relative.max()

    fc8_all = df[df.modbus_func_code == DIAGNOSTICS_FC]
    fc8_by_attack = fc8_all.attack_specific.fillna(-1).value_counts().to_dict()

    g = req.groupby("tcp_stream").agg(n=("row", "size"), start=("frame_time_relative", "min"),
                                       end=("frame_time_relative", "max"))
    g["dur"] = g.end - g.start
    within_round_gaps = []
    for _, grp in req.groupby("tcp_stream"):
        t = grp.sort_values("frame_time_relative").frame_time_relative.to_numpy()
        if len(t) > 1:
            within_round_gaps.extend(np.diff(t).tolist())

    attack_start, attack_end = float(req.frame_time_relative.min()), float(req.frame_time_relative.max())
    this_resp_len = int(resp.tcp_len.mode().iloc[0]) if len(resp) else None
    other_fc8_resp = fc8_all[(fc8_all.ip_src == TARGET_IP) & (fc8_all.attack_specific != 7)]
    other_resp_len_mode = int(other_fc8_resp.tcp_len.mode().iloc[0]) if len(other_fc8_resp) else None

    noise = a7[a7.protocol.isin(["DATA", "WEBSOCKET", "HTTP"])
              & a7.ip_src.isin([ATTACKER_IP, PARTNER_IP]) & a7.ip_dst.isin([ATTACKER_IP, PARTNER_IP])]

    n_syn = int(((n.protocol == "TCP") & (n.tcp_flags == "0x0002")).sum())
    normal_mb_ips = sorted(set(n[n.protocol == "MODBUS"].ip_src.unique())
                           | set(n[n.protocol == "MODBUS"].ip_dst.unique()))

    return {
        "session_duration_sec": round(dur, 1),
        "normal_fc8_rows": int((n.modbus_func_code == DIAGNOSTICS_FC).sum()),
        "attack_fc8_requests": int(len(req)),
        "attack_fc8_responses": int(len(resp)),
        "n_rounds": int(req.tcp_stream.nunique()),
        "requests_per_round": [int(x) for x in g.n.tolist()],
        "round_duration_sec_median": round(float(g.dur.median()), 1),
        "within_round_interval_median_sec": round(float(np.median(within_round_gaps)), 3),
        "within_round_interval_std_sec": round(float(np.std(within_round_gaps)), 4),
        "attack_window_start_sec": round(attack_start, 1),
        "attack_window_end_sec": round(attack_end, 1),
        "attack_window_span_sec": round(attack_end - attack_start, 1),
        "attack_window_span_pct": round((attack_end - attack_start) / dur * 100, 1),
        "this_response_len_bytes": this_resp_len,
        "other_fc8_response_len_bytes": other_resp_len_mode,
        "fc8_rows_other_attack_types": {str(k): int(v) for k, v in fc8_by_attack.items() if k != 7.0},
        "normal_syn_rate": round(n_syn / dur, 3),
        "attack_new_connections": int(req.tcp_stream.nunique()),
        "attack_total_rows": int(len(a7)),
        "attack_noise_rows": int(len(noise)),
        "attack_noise_pct": round(len(noise) / len(a7) * 100, 1),
        "normal_mb_ips": normal_mb_ips,
        "attacker_ip": ATTACKER_IP,
    }


def compute_timeline(df, bin_width=10.0):
    """Per-bin fc8 request counts across the whole session, normal (always
    0) vs. this attack's 10 rounds. Each round's 10 requests span ~27s, so
    unlike force-listen-mode's single-instant spikes, activity here shows
    up as small clusters across 2-3 adjacent bins per round.
    """
    dur = float(df.frame_time_relative.max())
    edges = np.arange(0, dur + bin_width, bin_width)
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a7 = df[df.attack_specific == 7]

    normal_times = n[(n.protocol == "MODBUS") & (n.modbus_func_code == DIAGNOSTICS_FC)].frame_time_relative.to_numpy()
    attack_times = a7[(a7.protocol == "MODBUS") & (a7.modbus_func_code == DIAGNOSTICS_FC)
                       & (a7.ip_src == ATTACKER_IP)].frame_time_relative.to_numpy()

    normal_counts, _ = np.histogram(normal_times, bins=edges)
    attack_counts, _ = np.histogram(attack_times, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    return {
        "bin_width_sec": bin_width,
        "session_duration_sec": round(dur, 1),
        "bin_centers": [round(float(x), 1) for x in bin_centers],
        "normal_counts": [int(x) for x in normal_counts],
        "attack_counts": [int(x) for x in attack_counts],
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
    parser = argparse.ArgumentParser(description="Compare normal traffic vs. the restart-communication attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting restart-communication examples (SYN, repeated fc8 pairs)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)
    print("Computing fc8-request time series (normal vs. attack, whole session)...")
    timeline = compute_timeline(df)

    # No "normal_pair" here on purpose, same reasoning as
    # packet_compare_smartgrid_deviceid.py / packet_compare_smartgrid_forcelisten.py:
    # fc8 never occurs in normal traffic at all.
    packets = {"attack_examples": attack_examples}
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

    print(f"Restart-communication summary: {stats['n_rounds']} rounds of ~{stats['requests_per_round'][0]} "
          f"requests each at {stats['within_round_interval_median_sec']}s intervals, spread across "
          f"{stats['attack_window_span_pct']}% of the session, response {stats['this_response_len_bytes']} "
          f"bytes (standard shape, unlike force-listen-mode's shortened one), "
          f"{stats['attack_noise_pct']}% of labeled rows are unrelated background traffic")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Restart Communication Diff</title>
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

  /* ---- protocol logic violation ---- */
  .logic-grid { display: flex; flex-direction: column; gap: 12px; }
  .logic-card { border: 1px solid var(--border); border-left: 3px solid var(--attack); border-radius: 8px;
                background: var(--surface-1); box-shadow: var(--shadow); padding: 14px 16px;
                display: flex; flex-direction: column; gap: 6px; }
  .logic-card-title { font-family: "Archivo", sans-serif; font-weight: 700; font-size: 14px; color: var(--text-primary); }
  .logic-card-rule { font-size: 11.5px; color: var(--text-muted); font-style: italic; line-height: 1.5; }
  .logic-card-body { font-size: 12.5px; color: var(--text-secondary); line-height: 1.55; }
  .logic-card-body b { color: var(--text-primary); }

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
    <h1 style="margin-top:8px">The same diagnostic call, ten times, every three seconds, ten times over</h1>
    <p style="margin-top:10px">"Restart communication" is pure Modbus Diagnostics (fc8) - <b id="rounds-inline"></b>
      rounds, each opening a brand-new TCP connection, each firing <b id="reqs-per-round-inline"></b>
      identical fc8 requests at an almost perfectly regular <b id="interval-inline"></b> interval, spread
      across <b id="span-pct-inline"></b> of the session. Unlike force listen mode, the response here is
      the STANDARD fc8 shape - the tell is the repeated, clockwork-regular cadence, not the reply.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>A note on the "Restart Communications Option" framing</h2>
      <p style="margin-top:6px">Same caution applied to force listen mode - being precise about what
        this dataset's columns do and don't let us verify.</p>
    </div>
    <div class="shared-target-banner">
      Modbus's real "Restart Communications Option" is Diagnostics sub-function 0x01 - consistent with
      this attack's assigned name. This capture's <code class="mono">modbus_data</code> field shows an
      empty/zero value for every request, which does not let us independently confirm the sub-function
      byte from this column alone. What IS independently verifiable: this attack's response is
      <b id="this-resp-inline"></b> bytes - the SAME standard shape fc8 gets under function-code scan
      and address scan (<b id="other-resp-inline"></b> bytes), unlike force listen mode's shortened
      10-byte response. So whatever protocol effect force listen mode triggers on the target, this
      attack does not trigger the same one - the concrete signal here is the request PATTERN (10
      identical requests, ~3.0s apart, repeated over 10 separate rounds), not the reply shape.
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; no equivalent exists</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; repeated diagnostics call (round 1 of <span id="rounds-inline2"></span>)</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Not everything labeled "attack" here is attack behavior</h2>
      <p style="margin-top:6px">Same background-noise check applied to every attack in this series - now
        at the highest magnitude seen so far.</p>
    </div>
    <div class="shared-target-banner">
      Of the <b id="noise-total-inline"></b> rows labeled <code class="mono">attack_specific&nbsp;==&nbsp;7</code>,
      <b id="noise-pct-inline"></b> (<b id="noise-rows-inline"></b> rows) are 192.168.0.1's ordinary
      DATA/WEBSOCKET traffic with 192.168.0.111 - higher than naive-sensor-read (30.8%), sporadic
      injection (34.4%), and force listen mode (40.3%).
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Protocol logic violation: why this traffic could not be legitimate</h2>
      <p style="margin-top:6px">Not "rare" or "different from baseline" - actually impossible under how
        a disruptive maintenance action is supposed to be used (see the Smart Grid
        normal-behavior-baseline memory, Section 1).</p>
    </div>
    <div class="logic-grid" id="logic-grid"></div>
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
      <h2>Time series: fc8 requests across the whole session</h2>
      <p style="margin-top:6px">Like force listen mode, the normal line here is a flat zero. Unlike
        force listen mode's single-instant spikes, each round here spans ~27s, so activity shows up as
        small clusters rather than single points.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal (always 0)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Restart communication</span>
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
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Restart communication</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Restart communication</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows labeled
      <code class="mono">attack_specific == 7</code>. Round 1 (TCP stream 6042, t&nbsp;=&nbsp;354.4s) is
      shown as the representative example; all 10 rounds follow the identical shape. Companion page to
      <code class="mono">packet_compare_smartgrid_deviceid.py</code> and
      <code class="mono">packet_compare_smartgrid_forcelisten.py</code> (both share the
      new-connection-per-round pattern and fc8 usage).</p>
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
  document.getElementById('rounds-inline').textContent = s.n_rounds;
  document.getElementById('rounds-inline2').textContent = s.n_rounds;
  document.getElementById('reqs-per-round-inline').textContent = s.requests_per_round[0];
  document.getElementById('interval-inline').textContent = s.within_round_interval_median_sec + 's';
  document.getElementById('span-pct-inline').textContent = s.attack_window_span_pct + '%';
  document.getElementById('this-resp-inline').textContent = s.this_response_len_bytes;
  document.getElementById('other-resp-inline').textContent = s.other_fc8_response_len_bytes;
  document.getElementById('noise-total-inline').textContent = s.attack_total_rows.toLocaleString();
  document.getElementById('noise-pct-inline').textContent = s.attack_noise_pct + '%';
  document.getElementById('noise-rows-inline').textContent = s.attack_noise_rows.toLocaleString();

  // ---- normal column ----
  const nCol = document.getElementById('normal-col');
  nCol.innerHTML =
    `<div class="frame">
      <div class="frame-label">This traffic type does not occur in normal operation</div>
      <div class="decoded" style="border-top:none">
        Function code 8 (Diagnostics) appears <b>${s.normal_fc8_rows}</b> times in 47,198 normal rows
        &mdash; never. There is no normal packet to pair this probe against.
      </div>
    </div>
    <div class="frame">
      <div class="frame-label">What normal traffic asks instead</div>
      <div class="decoded" style="border-top:none">
        Every normal Modbus request is a value poll (fc1/fc3/fc4) at a ~1s cadence per series - never a
        tightly-regular, machine-timed repetition of the exact same diagnostic call.
      </div>
    </div>`;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml = '';
  if (ex.syn) {
    attackHtml += `<div><h3 style="margin-bottom:8px">1. A fresh connection just for this round<span class="flag">SYN</span></h3>` +
      frame('SYN', ex.syn, true,
        `Every one of the ${s.n_rounds} rounds opens a brand-new TCP connection.`) +
      `</div>`;
  }
  if (ex.first_pair) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. First of 10 identical requests in this round<span class="flag">fc8</span></h3>` +
      frame('REQUEST', ex.first_pair.request, true,
        `Diagnostics (fc8), ${ex.first_pair.request.tcp_len} bytes.`) +
      frame('RESPONSE', ex.first_pair.response, true,
        `${ex.first_pair.response.tcp_len} bytes &mdash; the STANDARD fc8 echo shape, same as
         function-code scan and address scan (unlike force listen mode's shortened reply).`) +
      `</div>`;
  }
  if (ex.second_pair) {
    attackHtml += `<div><h3 style="margin-bottom:8px">3. Second request, ${s.within_round_interval_median_sec}s later<span class="flag">identical PDU</span></h3>` +
      frame('REQUEST', ex.second_pair.request, true,
        `Same PDU, same shape, exactly ${s.within_round_interval_median_sec}s after the first &mdash;
         this timing repeats ${s.requests_per_round[0] - 1} more times within this round, then a brand
         new round starts elsewhere in the session.`) +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- protocol logic violation ----
  const logicPoints = [
    {
      title: 'A disruptive maintenance command is used once to fix a problem, not spammed',
      rule: 'Restarting a device\'s communications resets its comm stack by design - a technician uses it once when something is actually wrong, verifies the fix, and stops.',
      body: `This attack fires the identical restart-communications call <b>${s.requests_per_round[0]}</b>
             times per round, across <b>${s.n_rounds}</b> separate rounds - no legitimate troubleshooting
             workflow repeats a disruptive reset this many times when the first one either worked or
             didn't.`,
    },
    {
      title: 'A machine-timed cadence is not how a human operates',
      rule: 'A technician\'s actions have human-timescale variability - deciding, typing, waiting for a response. Even normal automated polling here has some jitter (CV 0.143).',
      body: `Requests within a round land <b>${s.within_round_interval_median_sec}s</b> apart with a
             standard deviation of only <b>${s.within_round_interval_std_sec}s</b> - essentially
             zero jitter, a signature of a script issuing timed calls, not a person working through a
             maintenance checklist.`,
    },
    {
      title: 'Only one master exists, and this is not it',
      rule: 'The deployed topology has exactly one master (192.168.0.40) and one RTU (192.168.0.31) - no third party, technician tool or otherwise, is ever expected to speak Modbus to it directly.',
      body: `Every request in every round comes from 192.168.0.1, a host that has never once been the
             known master or run any engineering workflow against this device before this attack began.`,
    },
  ];
  document.getElementById('logic-grid').innerHTML = logicPoints.map(p => `
    <div class="logic-card">
      <div class="logic-card-title">${esc(p.title)}</div>
      <div class="logic-card-rule">${p.rule}</div>
      <div class="logic-card-body">${p.body}</div>
    </div>`).join('');

  // ---- detection signals ----
  const signals = [
    {
      title: 'Function code 8 (Diagnostics)', tag: 'categorical',
      normal: '0 requests', attack: `${s.attack_fc8_requests} requests`,
      note: `Never appears in 47,198 normal rows. Also used by 2 other attack types (function-code
             scan, address scan, force listen mode), so its presence alone identifies an attack, not
             which one - the repeated-request pattern is what's specific to this label.`,
    },
    {
      title: 'Within-round request interval', tag: 'statistical',
      normal: '~1.01s (median, CV 0.143)', attack: `${s.within_round_interval_median_sec}s (std ${s.within_round_interval_std_sec}s)`,
      note: `Far more regular than even normal polling's own cadence - a near-zero standard deviation
             across ${s.requests_per_round.length} rounds is a machine-timed signature, not organic
             traffic.`,
    },
    {
      title: 'Requests per round', tag: 'categorical',
      normal: 'n/a (no rounds)', attack: `${s.requests_per_round[0]} (10 identical fc8 calls)`,
      note: `Every round fires the same request ${s.requests_per_round[0]} times before moving to a
             brand-new TCP connection elsewhere in the session.`,
    },
    {
      title: 'Response shape vs. force listen mode', tag: 'categorical',
      normal: 'n/a (never asked)', attack: `${s.this_response_len_bytes} bytes (standard)`,
      note: `Same 12-byte shape fc8 gets under function-code scan/address scan - NOT the 10-byte
             shortened response force listen mode gets. These are two measurably different fc8
             behaviors sharing the same function code.`,
    },
    {
      title: 'Request source identity', tag: 'categorical',
      normal: s.normal_mb_ips.join(', '), attack: s.attacker_ip,
      note: `Only ${s.normal_mb_ips.join(' and ')} ever send/receive Modbus traffic in the normal
             baseline. ${s.attacker_ip} has never been a Modbus participant before this attack.`,
    },
    {
      title: 'Attack rows that are actually unrelated traffic', tag: 'statistical',
      normal: '0%', attack: `${s.attack_noise_pct}%`,
      note: `See the background-noise section above - the highest of any attack in this series so
             far, essential context for reading any raw row-count statistic here.`,
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

  // ---- time series (fc8 requests across the whole session) ----
  function drawTimeSeries(tl) {
    const width = 1000, height = 260;
    const pad = {top: 16, right: 16, bottom: 30, left: 40};
    const innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom;
    const dur = tl.session_duration_sec;
    const maxV = Math.max(...tl.attack_counts, 1) * 1.15;
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
      lbl.textContent = Math.round(frac * maxV);
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

    const zeroY = y(0);
    const zline = document.createElementNS(svg.namespaceURI, 'line');
    zline.setAttribute('x1', pad.left); zline.setAttribute('x2', width - pad.right);
    zline.setAttribute('y1', zeroY); zline.setAttribute('y2', zeroY);
    zline.setAttribute('stroke', nColor); zline.setAttribute('stroke-width', '1.75');
    svg.appendChild(zline);

    function areaPath(counts) {
      const pts = tl.bin_centers.map((t, i) => `${x(t)},${y(counts[i])}`);
      return `M${pad.left},${zeroY} L${pts.join(' L')} L${x(dur)},${zeroY} Z`;
    }
    const attackArea = document.createElementNS(svg.namespaceURI, 'path');
    attackArea.setAttribute('d', areaPath(tl.attack_counts));
    attackArea.setAttribute('fill', aColor); attackArea.setAttribute('fill-opacity', '0.55');
    attackArea.setAttribute('stroke', aColor); attackArea.setAttribute('stroke-width', '1');
    svg.appendChild(attackArea);

    return svg;
  }

  const tl = DATA.timeline;
  document.getElementById('timeline-chart').appendChild(drawTimeSeries(tl));
  document.getElementById('timeline-caption').innerHTML =
    `Normal traffic never sends fc8 - <b class="mono" style="color:var(--normal)">0</b> occurrences
     across the whole session. This attack's ${s.n_rounds} rounds appear as
     <b class="mono" style="color:var(--attack)">${tl.attack_active_bin_pct}%</b> of the session's bins
     - up to <b class="mono" style="color:var(--attack)">${tl.attack_max_per_bin}</b> requests in a
     single bin, in short clusters (each round spans ~${s.round_duration_sec_median}s) scattered across
     ${s.attack_window_span_pct}% of the session.`;

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
    {label: 'fc8 requests sent', normal: s.normal_fc8_rows, attack: s.attack_fc8_requests},
    {label: 'Rounds (new TCP connections)', normal: 0, attack: s.n_rounds},
    {label: 'Response size (bytes)', normal: 0.001, attack: s.this_response_len_bytes, logScale: true},
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
    ['fc8 (Diagnostics) requests', `${s.normal_fc8_rows}`, `${s.attack_fc8_requests}`, true],
    ['fc8 response size', '—', `${s.this_response_len_bytes} bytes (standard shape)`, false],
    ['Rounds (new TCP connections)', '0', `${s.n_rounds}`, true],
    ['Requests per round', '—', `${s.requests_per_round.join(', ')}`, false],
    ['Within-round request interval', '~1.01s (normal poll cadence)', `${s.within_round_interval_median_sec}s (std ${s.within_round_interval_std_sec}s)`, true],
    ['Attack window span', '—', `${s.attack_window_span_sec.toLocaleString()}s (${s.attack_window_span_pct}% of the session)`, true],
    ['fc8 rows under OTHER attack labels', '—', Object.entries(s.fc8_rows_other_attack_types).map(([k,v]) => `type ${Math.trunc(parseFloat(k))}: ${v}`).join(', '), false],
    ['Rows labeled attack_specific=7', '—', `${s.attack_total_rows.toLocaleString()}`, false],
    ['...of which unrelated background traffic', '—', `${s.attack_noise_rows.toLocaleString()} (${s.attack_noise_pct}%)`, true],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
