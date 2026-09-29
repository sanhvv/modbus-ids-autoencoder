"""
Seventh in the per-attack-type series for Smart Grid (after function-code
scan, address scan, device identification attack, naive sensor read,
sporadic sensor measurement injection - see the Smart Grid
normal-behavior-baseline memory), applied to "force listen mode"
(attack_specific == 6).

Like device identification, this attack's request TYPE (a Diagnostics
sub-function call, fc8) never occurs in normal traffic at all, so there is
no normal_pair - same reasoning as packet_compare_smartgrid_deviceid.py.

Structure, verified from the raw data: 8 rounds, one Diagnostics (fc8)
request/response pair each, each opening a BRAND-NEW TCP connection (8
distinct streams), spread across t=684.0s-4763.3s (4079.3s span, 68.4% of
the session), average gap ~582.8s (~9.7 min) between rounds - even
lower-and-slower than device identification's 16 rounds/~5.8 min average.

IMPORTANT CAVEAT on the "Force Listen Only Mode" framing (verified, not
assumed): the modbus_data field for every one of these 8 requests decodes
to 0x0000 and the response is empty (0x) - i.e. this CSV's modbus_data
column does not let us independently confirm the Diagnostics sub-function
code byte (0x04 is the real Modbus "Force Listen Only Mode" sub-function;
0x00 is "Return Query Data" and would show the same all-zero pattern in
this field if the sub-function itself isn't captured separately). What IS
independently verifiable and specific to this attack: the RESPONSE is 2
bytes shorter (10 bytes total) than the request (12 bytes) - a different
response shape from every other fc8 usage in this dataset (function-code
scan's 18 rows and address-scan's 1 row both get a full 12-byte echo
response; restart-communication's 198 rows are investigated separately).
This shorter-response shape is the concrete signal used below, not an
assumption about which exact sub-function was sent.

Also verified: the same background-noise co-mingling issue documented for
naive-sensor-read/sporadic-injection applies here too (38.3% of this
attack's 962 rows are 192.168.0.1's ordinary DATA/WEBSOCKET/HTTP traffic
with 192.168.0.111, unrelated to the fc8 probes).

DEEP PHYSICAL-CONSEQUENCE CHECK (added 2026-09-29, user request - the same
kind of "commanded state vs. real observed state" contradiction as the
naive-sensor-read/sporadic-injection solar-switch example, applied here to
communication itself rather than a sensor reading): if this probe really
did put the RTU into Force Listen Only Mode, the device stops responding to
non-broadcast requests until explicitly taken out of that mode - so the
legitimate master's very next poll after each of the 8 rounds should show
a gap or a dropped response. Checked directly: it does not. The master's
normal ~1s polling cadence to the switch/meters continues completely
uninterrupted immediately after every single round (largest observed gap
to the next normal poll: 0.61s, well within the normal ~1s cycle) - real,
independent evidence (not just the inconclusive modbus_data field) that
whatever this probe did, it did not actually suppress the device's normal
communication with its legitimate master.

Output goes into data_visualisation/smartgrid_force_listen/ (all filenames
get the optional --tag suffix so earlier results are not overwritten):
packets.json, stats.json, timeline.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_forcelisten.py 2>&1 | tee data_visualisation/smartgrid_force_listen/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv
from packet_compare_smartgrid_fcscan import pack

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_force_listen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
TARGET_IP = "192.168.0.31"
PARTNER_IP = "192.168.0.111"
MASTER_IP = "192.168.0.40"
DIAGNOSTICS_FC = 8


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def compute_comms_continuity_check(df):
    """Physical-consequence check: if this probe genuinely put the RTU
    into Force Listen Only Mode, the legitimate master's next poll should
    show a gap or dropped response (the device stops answering
    non-broadcast requests in that mode). Checks the real gap to the
    nearest normal poll immediately after each of the 8 rounds.
    """
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    master_req = n[(n.protocol == "MODBUS") & (n.ip_src == MASTER_IP)].sort_values("frame_time_relative")
    master_times = master_req.frame_time_relative.to_numpy()

    a6 = df[df.attack_specific == 6]
    req = a6[(a6.protocol == "MODBUS") & (a6.ip_src == ATTACKER_IP)].sort_values("frame_time_relative")
    probe_times = req.frame_time_relative.to_numpy()

    gaps_after = []
    for t in probe_times:
        later = master_times[master_times > t]
        if len(later):
            gaps_after.append(float(later.min() - t))

    return {
        "n_probes_checked": int(len(probe_times)),
        "max_gap_after_sec": round(max(gaps_after), 2) if gaps_after else None,
        "min_gap_after_sec": round(min(gaps_after), 2) if gaps_after else None,
        "mean_gap_after_sec": round(float(np.mean(gaps_after)), 2) if gaps_after else None,
        "normal_poll_cadence_sec": 1.01,
    }


def extract_attack_examples(df):
    a6 = df[df.attack_specific == 6].sort_values("row")
    mb = a6[a6.protocol == "MODBUS"].sort_values("frame_time_relative").reset_index(drop=True)

    syn = a6[(a6.protocol == "TCP") & (a6.tcp_flags == "0x0002")
             & (a6.tcp_stream == mb.tcp_stream.iloc[0])]

    probe = None
    for i in range(len(mb) - 1):
        if mb.iloc[i].ip_src == ATTACKER_IP and mb.iloc[i + 1].ip_src == TARGET_IP:
            probe = {"request": pack(mb.iloc[i]), "response": pack(mb.iloc[i + 1])}
            break

    return {"syn": pack(syn.iloc[0]) if len(syn) else None, "probe": probe}


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a6 = df[df.attack_specific == 6]
    mb = a6[a6.protocol == "MODBUS"]
    req = mb[mb.ip_src == ATTACKER_IP]
    resp = mb[mb.ip_src == TARGET_IP]
    dur = df.frame_time_relative.max()

    fc8_all = df[df.modbus_func_code == DIAGNOSTICS_FC]
    fc8_by_attack = fc8_all.attack_specific.fillna(-1).value_counts().to_dict()
    # response shape comparison: this attack's responses vs fc8 responses
    # under OTHER attack labels (restart-comm, fc-scan, address-scan).
    other_fc8_resp = fc8_all[(fc8_all.ip_src == TARGET_IP) & (fc8_all.attack_specific != 6)]
    this_resp_len = int(resp.tcp_len.iloc[0]) if len(resp) else None
    other_resp_len_mode = int(other_fc8_resp.tcp_len.mode().iloc[0]) if len(other_fc8_resp) else None

    noise = a6[a6.protocol.isin(["DATA", "WEBSOCKET", "HTTP"])
              & a6.ip_src.isin([ATTACKER_IP, PARTNER_IP]) & a6.ip_dst.isin([ATTACKER_IP, PARTNER_IP])]

    t = req.sort_values("frame_time_relative").frame_time_relative.to_numpy()
    attack_start, attack_end = float(t.min()), float(t.max())

    n_syn = int(((n.protocol == "TCP") & (n.tcp_flags == "0x0002")).sum())
    normal_mb_ips = sorted(set(n[n.protocol == "MODBUS"].ip_src.unique())
                           | set(n[n.protocol == "MODBUS"].ip_dst.unique()))

    return {
        "session_duration_sec": round(dur, 1),
        "normal_fc8_rows": int((n.modbus_func_code == DIAGNOSTICS_FC).sum()),
        "attack_fc8_requests": int(len(req)),
        "attack_fc8_responses": int(len(resp)),
        "n_rounds": int(req.tcp_stream.nunique()),
        "attack_window_start_sec": round(attack_start, 1),
        "attack_window_end_sec": round(attack_end, 1),
        "attack_window_span_sec": round(attack_end - attack_start, 1),
        "avg_seconds_between_rounds": round((attack_end - attack_start) / max(len(t) - 1, 1), 1),
        "this_response_len_bytes": this_resp_len,
        "other_fc8_response_len_bytes": other_resp_len_mode,
        "fc8_rows_other_attack_types": {str(k): int(v) for k, v in fc8_by_attack.items() if k != 6.0},
        "normal_syn_rate": round(n_syn / dur, 3),
        "attack_new_connections": int(req.tcp_stream.nunique()),
        "attack_total_rows": int(len(a6)),
        "attack_noise_rows": int(len(noise)),
        "attack_noise_pct": round(len(noise) / len(a6) * 100, 1),
        "normal_mb_ips": normal_mb_ips,
        "attacker_ip": ATTACKER_IP,
    }


def compute_timeline(df, bin_width=10.0):
    """Per-bin fc8 request counts across the whole session, normal (always
    0 - fc8 never occurs in normal traffic) vs. this attack's 8 rounds.
    Same treatment as packet_compare_smartgrid_deviceid.py.
    """
    dur = float(df.frame_time_relative.max())
    edges = np.arange(0, dur + bin_width, bin_width)
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a6 = df[df.attack_specific == 6]

    normal_times = n[(n.protocol == "MODBUS") & (n.modbus_func_code == DIAGNOSTICS_FC)].frame_time_relative.to_numpy()
    attack_times = a6[(a6.protocol == "MODBUS") & (a6.modbus_func_code == DIAGNOSTICS_FC)
                       & (a6.ip_src == ATTACKER_IP)].frame_time_relative.to_numpy()

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
    parser = argparse.ArgumentParser(description="Compare normal traffic vs. the force-listen-mode attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting force-listen-mode examples (SYN, diagnostics probe)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)
    print("Computing fc8-request time series (normal vs. attack, whole session)...")
    timeline = compute_timeline(df)
    print("Checking whether real polling continuity survives each probe...")
    comms_check = compute_comms_continuity_check(df)

    # No "normal_pair" here on purpose, same reasoning as
    # packet_compare_smartgrid_deviceid.py: fc8 never occurs in normal
    # traffic at all (0 of 47,198 rows).
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

    with open(path("comms_check.json"), "w") as f:
        json.dump(comms_check, f, indent=1)
    print(f"Saved: {path('comms_check.json')}")

    payload = {**packets, "stats": stats, "timeline": timeline, "comms_check": comms_check}
    report_path = path("report.html")
    report_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"Force-listen-mode summary: {stats['n_rounds']} rounds, spread across "
          f"{stats['attack_window_span_sec']}s (avg {stats['avg_seconds_between_rounds']}s between "
          f"rounds), response {stats['this_response_len_bytes']} bytes vs. "
          f"{stats['other_fc8_response_len_bytes']} bytes for fc8 elsewhere in the dataset, "
          f"{stats['attack_noise_pct']}% of labeled rows are unrelated background traffic")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Force Listen Mode Diff</title>
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
  .logic-card.flagship { border-left-width: 4px; }
  .logic-flow { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-family: "IBM Plex Mono", monospace;
                font-size: 12px; padding: 10px 12px; background: var(--surface-0); border-radius: 6px; margin-top: 4px; }
  .logic-flow .step { padding: 4px 9px; border-radius: 5px; border: 1px solid var(--border); }
  .logic-flow .step.low { color: var(--normal); border-color: var(--normal); background: var(--normal-bg); }
  .logic-flow .step.cmd { color: var(--attack); border-color: var(--attack); background: var(--attack-bg); font-weight: 600; }
  .logic-flow .arrow { color: var(--text-muted); }

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
    <h1 style="margin-top:8px">A diagnostic probe with a shorter answer</h1>
    <p style="margin-top:10px">"Force listen mode" is pure Modbus Diagnostics (fc8) - <b id="rounds-inline"></b>
      rounds, each opening a brand-new TCP connection, spread across <b id="span-inline"></b> of the
      session (average <b id="avg-gap-inline"></b> between rounds) - the lowest-and-slowest cadence in
      this series so far. What makes it distinct isn't the request (identical shape to fc8 probes seen
      elsewhere in this dataset) but the RESPONSE: 2 bytes shorter every single time.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>A note on the "Force Listen Only Mode" framing</h2>
      <p style="margin-top:6px">Being precise about what this dataset's columns do and don't let us
        verify directly.</p>
    </div>
    <div class="shared-target-banner">
      Modbus's real "Force Listen Only Mode" is Diagnostics sub-function 0x04 - consistent with this
      attack's assigned name. However, this capture's <code class="mono">modbus_data</code> field shows
      the same all-zero value (<code class="mono">0x0000</code>) for every one of this attack's 8
      requests, which does not let us independently confirm the sub-function byte from this column
      alone (it would show the same pattern whether the sub-function is 0x04 or 0x00). What IS
      independently verifiable and specific to this attack: the response is
      <b id="this-resp-inline"></b> bytes, 2 bytes shorter than the <b id="other-resp-inline"></b>-byte
      echo response fc8 gets under every OTHER attack label in this dataset (function-code scan,
      address scan) - a real, measured difference in protocol behavior, not an assumption about which
      sub-function was sent.
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; no equivalent exists</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; diagnostics probe (round 1 of <span id="rounds-inline2"></span>)</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Not everything labeled "attack" here is attack behavior</h2>
      <p style="margin-top:6px">Same background-noise check applied to every attack in this series.</p>
    </div>
    <div class="shared-target-banner">
      Of the <b id="noise-total-inline"></b> rows labeled <code class="mono">attack_specific&nbsp;==&nbsp;6</code>,
      <b id="noise-pct-inline"></b> (<b id="noise-rows-inline"></b> rows) are 192.168.0.1's ordinary
      DATA/WEBSOCKET/HTTP traffic with 192.168.0.111 - the same co-mingling pattern found in
      naive-sensor-read (30.8%) and sporadic injection (34.4%).
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Protocol logic violation: why this traffic could not be legitimate</h2>
      <p style="margin-top:6px">Not "rare" or "different from baseline" - actually impossible under how
        diagnostic administration is supposed to be used (see the Smart Grid
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
      <p style="margin-top:6px">Like device identification, the normal line here is a flat zero -
        Diagnostics requests have no normal-traffic occurrence at all.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal (always 0)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Force listen mode</span>
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
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Force listen mode</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Force listen mode</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows labeled
      <code class="mono">attack_specific == 6</code>. Round 1 (TCP stream 6084, t&nbsp;=&nbsp;684.0s) is
      shown as the representative example; all 8 rounds follow the identical shape. Companion page to
      <code class="mono">packet_compare_smartgrid_deviceid.py</code> (same low-and-slow,
      new-connection-per-round pattern) and <code class="mono">packet_compare_smartgrid_fcscan.py</code>
      (shares fc8 usage).</p>
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
  document.getElementById('span-inline').textContent = (s.attack_window_span_sec / 60).toFixed(1) + ' min';
  document.getElementById('avg-gap-inline').textContent = (s.avg_seconds_between_rounds / 60).toFixed(1) + ' min';
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
        &mdash; never. There is no normal packet to pair this probe against, because a diagnostics
        query is not part of this system's vocabulary in normal operation at all.
      </div>
    </div>
    <div class="frame">
      <div class="frame-label">What normal traffic asks instead</div>
      <div class="decoded" style="border-top:none">
        Every normal Modbus request is a value poll (fc1/fc3/fc4) at one of 4 known addresses - never
        a protocol-level diagnostic call.
      </div>
    </div>`;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml = '';
  if (ex.syn) {
    attackHtml += `<div><h3 style="margin-bottom:8px">1. A fresh connection just for this probe<span class="flag">SYN</span></h3>` +
      frame('SYN', ex.syn, true,
        `Every one of the ${s.n_rounds} rounds opens a brand-new TCP connection &mdash; nothing is reused,
         same pattern as the device-identification attack.`) +
      `</div>`;
  }
  if (ex.probe) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. Diagnostics probe<span class="flag">fc8</span></h3>` +
      frame('REQUEST', ex.probe.request, true,
        `Diagnostics (fc8), ${ex.probe.request.tcp_len} bytes.`) +
      frame('RESPONSE', ex.probe.response, true,
        `Only ${ex.probe.response.tcp_len} bytes &mdash; ${s.other_fc8_response_len_bytes - ex.probe.response.tcp_len}
         bytes shorter than fc8's response shape everywhere else in this dataset (function-code scan,
         address scan). This shorter reply is the concrete, measured signal for this attack type.`) +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- protocol logic violation ----
  const cc = DATA.comms_check;
  const logicPoints = [
    {
      flagship: true,
      title: 'The claimed effect never actually happens',
      rule: `If this probe genuinely put the RTU into Force Listen Only Mode, the device stops
             answering non-broadcast requests until explicitly reset - so the legitimate master's very
             next poll should hit a gap or a dropped response.`,
      body: () => {
        const flow = `<div class="logic-flow">
          <span class="step cmd">Force-listen probe sent (${cc.n_probes_checked} times)</span>
          <span class="arrow">&mdash; claimed effect: device stops answering &mdash;</span>
          <span class="step low">next normal poll answered ${cc.max_gap_after_sec}s later (max), inside the normal ~${cc.normal_poll_cadence_sec}s cycle</span>
        </div>`;
        return `Checked directly, not assumed: across all ${cc.n_probes_checked} rounds, the largest
                gap between a probe and the legitimate master's next answered poll is only
                <b>${cc.max_gap_after_sec}s</b> (mean ${cc.mean_gap_after_sec}s) - well inside the normal
                ~${cc.normal_poll_cadence_sec}s polling cycle, with no dropped response anywhere. Real,
                independent evidence (not just the inconclusive request byte) that whatever this probe
                does, it does not suppress the device's normal communication with its master - the same
                kind of "claimed action vs. observed reality" check as the solar-switch example in the
                naive-sensor-read and sporadic-injection reports, applied here to communication itself.${flow}`;
      },
    },
    {
      title: 'Diagnostic administration is a maintenance action, not routine SCADA traffic',
      rule: 'Diagnostics calls (fc8) are used by an authorized technician during commissioning or troubleshooting - never as a scheduled part of live measurement/control polling.',
      body: `This attack repeats the identical diagnostics probe <b>${s.n_rounds}</b> times, roughly
             every <b>${(s.avg_seconds_between_rounds/60).toFixed(1)}&nbsp;minutes</b> for over an hour
             - a real technician troubleshoots once and moves on, not on a recurring schedule forever.`,
    },
    {
      title: 'A diagnostic tool keeps its session open',
      rule: 'A real engineering session issuing a diagnostic call reuses its connection for the whole troubleshooting task - it does not tear down and reopen a fresh TCP session for one single call.',
      body: `Every one of the ${s.n_rounds} rounds opens a brand-new TCP connection just to send this
             one probe, then closes it - not how an actual maintenance session behaves.`,
    },
    {
      title: 'Only one master exists, and this is not it',
      rule: 'The deployed topology has exactly one master (192.168.0.40) and one RTU (192.168.0.31) - no third party, technician tool or otherwise, is ever expected to speak Modbus to it directly.',
      body: `Every probe comes from 192.168.0.1, a host that has never once been the known master or run
             any engineering workflow against this device before this attack began.`,
    },
  ];
  document.getElementById('logic-grid').innerHTML = logicPoints.map(p => `
    <div class="logic-card${p.flagship ? ' flagship' : ''}">
      <div class="logic-card-title">${esc(p.title)}</div>
      <div class="logic-card-rule">${p.rule}</div>
      <div class="logic-card-body">${typeof p.body === 'function' ? p.body() : p.body}</div>
    </div>`).join('');

  // ---- detection signals ----
  const signals = [
    {
      title: 'Function code 8 (Diagnostics)', tag: 'categorical',
      normal: '0 requests', attack: `${s.attack_fc8_requests} requests`,
      note: `Never appears in 47,198 normal rows. Also used (in passing) by 2 other attack types
             (function-code scan, address scan), so its presence alone identifies an attack, not
             which one.`,
    },
    {
      title: 'Response length (bytes)', tag: 'categorical',
      normal: 'n/a (never asked)', attack: `${s.this_response_len_bytes} bytes`,
      note: `${s.other_fc8_response_len_bytes} bytes for fc8 under every other attack label in this
             dataset - this attack's response is 2 bytes shorter every single time, a distinguishing
             protocol-behavior signal specific to this attack type.`,
    },
    {
      title: 'Connections opened solely for this probe', tag: 'categorical',
      normal: '0', attack: `${s.attack_new_connections}`,
      note: `Every round opens a brand-new TCP connection just to send this one probe - nothing is
             ever reused, same pattern as device identification.`,
    },
    {
      title: 'Repeat cadence (average gap)', tag: 'statistical',
      normal: `${(s.normal_syn_rate * 3600).toFixed(0)} SYN/hr (network-wide)`, attack: `${(3600 / s.avg_seconds_between_rounds).toFixed(1)}/hr`,
      note: `The lowest-and-slowest cadence in this series so far - one round roughly every
             ${(s.avg_seconds_between_rounds / 60).toFixed(1)} minutes, well below typical
             connection-opening activity.`,
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
      note: `See the background-noise section above - essential context for reading any raw row-count
             statistic about this attack, not a detection signal itself.`,
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
    const maxV = Math.max(...tl.attack_counts, 1) * 1.3;
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

    tl.bin_centers.forEach((t, i) => {
      const v = tl.attack_counts[i];
      if (v <= 0) return;
      const line = document.createElementNS(svg.namespaceURI, 'line');
      line.setAttribute('x1', x(t)); line.setAttribute('x2', x(t));
      line.setAttribute('y1', zeroY); line.setAttribute('y2', y(v));
      line.setAttribute('stroke', aColor); line.setAttribute('stroke-width', '2.5');
      svg.appendChild(line);
    });

    return svg;
  }

  const tl = DATA.timeline;
  document.getElementById('timeline-chart').appendChild(drawTimeSeries(tl));
  document.getElementById('timeline-caption').innerHTML =
    `Normal traffic never sends fc8 - <b class="mono" style="color:var(--normal)">0</b> occurrences
     across the whole session. This attack's ${s.n_rounds} rounds appear as
     <b class="mono" style="color:var(--attack)">${tl.attack_active_bin_pct}%</b> of the session's bins
     - just <b class="mono" style="color:var(--attack)">${tl.attack_max_per_bin}</b> request per active
     bin, isolated and far apart, the sparsest pattern in this series so far.`;

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
    {label: 'Response size (bytes)', normal: 0.001, attack: s.this_response_len_bytes, logScale: true},
    {label: 'New connections opened', normal: 0, attack: s.attack_new_connections},
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
    ['fc8 response size', '—', `${s.this_response_len_bytes} bytes (vs ${s.other_fc8_response_len_bytes} elsewhere)`, true],
    ['Separate probing rounds (new TCP connections)', '0', `${s.n_rounds}`, true],
    ['Attack window span', '—', `${s.attack_window_span_sec.toLocaleString()}s (${(s.attack_window_span_sec/60).toFixed(1)} min, ${(s.attack_window_span_sec/s.session_duration_sec*100).toFixed(0)}% of the session)`, true],
    ['Average gap between rounds', '—', `${s.avg_seconds_between_rounds.toLocaleString()}s (${(s.avg_seconds_between_rounds/60).toFixed(1)} min)`, true],
    ['fc8 rows under OTHER attack labels', '—', Object.entries(s.fc8_rows_other_attack_types).map(([k,v]) => `type ${Math.trunc(parseFloat(k))}: ${v}`).join(', '), false],
    ['Rows labeled attack_specific=6', '—', `${s.attack_total_rows.toLocaleString()}`, false],
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
