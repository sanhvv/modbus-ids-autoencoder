"""
Same treatment as packet_compare_smartgrid_fcscan.py, applied to Smart Grid's
"address scan" attack (attack_specific == 1) - part of the per-attack-type
series requested after the function-code-scan comparison (see the Smart Grid
normal-behavior-baseline memory).

This attack tells a fundamentally different story than the function-code
scan: 192.168.0.1 is itself a LEGITIMATE host on this network (it has 1,252
normal-labeled rows, all TCP/websocket traffic with 192.168.0.111, and it
never sends a single SYN packet in the entire normal baseline). During the
attack, the SAME host suddenly ARP-resolves and TCP-SYN-scans 5 hosts it has
NEVER contacted before (192.168.0.21/22/23/31/40), spread across nearly the
whole session (not one short burst like the function-code scan) - and only
in the last ~0.075% of its footprint does it touch Modbus at all (a handful
of Report Device ID probes and one large-quantity read near the end).

The standout finding: 35,183 of this attack's 37,306 rows (94.3%) are ARP -
which is 98.0% of ALL the ARP traffic in the entire session. The current
AE/LLM pipeline only ever looks at protocol == "MODBUS" rows, so it is
structurally blind to nearly this entire attack; it can only ever see the 28
Modbus rows at the very tail end.

Output goes into data_visualisation/smartgrid_address_scan/ (all filenames
get the optional --tag suffix so earlier results are not overwritten):
packets.json, stats.json, report.html. Run with a log, e.g.:
    python packet_compare_smartgrid_addressscan.py 2>&1 | tee data_visualisation/smartgrid_address_scan/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_address_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
KNOWN_PARTNER_IP = "192.168.0.111"   # the only host .1 legitimately talks to
STANDARD_FC = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 20, 21, 22, 23, 24, 43}

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


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def extract_normal_example(df):
    """192.168.0.1's normal traffic: an ACK continuing its one established
    session with 192.168.0.111 - never a SYN, it never opens a new connection."""
    n1 = df[((df.ip_src == ATTACKER_IP) | (df.ip_dst == ATTACKER_IP))
            & (df.attack_specific.isna() | (df.attack_specific == 0))
            & (df.protocol.isin(["DATA", "TCP"]))].sort_values("row")
    push = n1[(n1.protocol == "DATA") & (n1.ip_src == KNOWN_PARTNER_IP)].iloc[0]
    ack = df.iloc[int(push.row) + 1]
    return {"push": pack(push), "ack": pack(ack)}


def extract_attack_examples(df):
    a1 = df[df.attack_specific == 1].sort_values("row")

    # 1) a SYN scan probe against a host .1 has never talked to before
    syn = a1[(a1.ip_src == ATTACKER_IP) & (a1.protocol == "TCP")
             & (a1.tcp_flags == "0x0002") & (a1.ip_dst == "192.168.0.31")].iloc[0]

    # 2) fc43 Report Device Identification probe/response (first occurrence)
    mb = a1[a1.protocol == "MODBUS"].sort_values("frame_time_relative")
    fc43 = mb[mb.modbus_func_code == 43]
    device_id = {"request": pack(fc43.iloc[0]), "response": pack(fc43.iloc[1])} if len(fc43) >= 2 else None

    # 3) the late full-quantity read near the end of the scan window
    fc1 = mb[mb.modbus_func_code == 1]
    late_read = {"request": pack(fc1.iloc[0]), "response": pack(fc1.iloc[1])} if len(fc1) >= 2 else None

    return {"syn_probe": pack(syn), "device_id_probe": device_id, "late_read": late_read}


def compute_timeline(df, bin_width=10.0):
    """Per-bin ARP frame counts across the whole session, normal vs. this
    attack - ARP, not Modbus, is where this attack's real signal lives
    (94.3% of its rows), so this plots a different metric than the other
    reports in this series. Added 2026-09-29 per user request, same
    treatment as packet_compare_smartgrid_naivesensorread.py.
    """
    dur = float(df.frame_time_relative.max())
    edges = np.arange(0, dur + bin_width, bin_width)
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a1 = df[df.attack_specific == 1]

    normal_times = n[n.protocol == "ARP"].frame_time_relative.to_numpy()
    attack_times = a1[a1.protocol == "ARP"].frame_time_relative.to_numpy()

    normal_counts, _ = np.histogram(normal_times, bins=edges)
    attack_counts, _ = np.histogram(attack_times, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # This is not a sustained flood - it's a repeating cycle of short ARP
    # sweeps. Cluster attack ARP frames by >5s gaps (same method used
    # across this series) to measure that cycle precisely.
    sorted_t = np.sort(attack_times)
    gaps = np.diff(sorted_t)
    cuts = np.where(gaps > 5.0)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(sorted_t) - 1]
    n_bursts = len(starts)
    burst_durs = sorted_t[ends] - sorted_t[starts]
    gap_between = sorted_t[starts[1:]] - sorted_t[ends[:-1]] if n_bursts > 1 else np.array([0.0])

    return {
        "bin_width_sec": bin_width,
        "session_duration_sec": round(dur, 1),
        "bin_centers": [round(float(x), 1) for x in bin_centers],
        "normal_counts": [int(x) for x in normal_counts],
        "attack_counts": [int(x) for x in attack_counts],
        "normal_avg_per_bin": round(float(normal_counts.mean()), 2),
        "attack_avg_per_bin": round(float(attack_counts.mean()), 2),
        "attack_max_per_bin": int(attack_counts.max()),
        "attack_active_bin_pct": round(float((attack_counts > 0).mean() * 100), 1),
        "n_arp_bursts": int(n_bursts),
        "arp_burst_avg_dur_sec": round(float(burst_durs.mean()), 1),
        "arp_burst_avg_gap_sec": round(float(gap_between.mean()), 1),
    }


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a1 = df[df.attack_specific == 1]
    dur = df.frame_time_relative.max()
    dur1 = a1.frame_time_relative.max() - a1.frame_time_relative.min()

    n_arp = int((n.protocol == "ARP").sum())
    a_arp = int((a1.protocol == "ARP").sum())
    total_arp = int((df.protocol == "ARP").sum())

    n1_normal = n[(n.ip_src == ATTACKER_IP) | (n.ip_dst == ATTACKER_IP)]
    normal_partners = sorted((set(n1_normal.ip_src.dropna().unique())
                              | set(n1_normal.ip_dst.dropna().unique())) - {ATTACKER_IP, "224.0.0.251"})

    a1_hosts = a1[(a1.ip_src == ATTACKER_IP) & a1.ip_dst.notna()]
    attack_partners = sorted(set(a1_hosts.ip_dst.unique()) - {ATTACKER_IP, "224.0.0.251"})
    new_partners = [h for h in attack_partners if h not in normal_partners]

    n_syn_from_attacker = int(((n.ip_src == ATTACKER_IP) & (n.protocol == "TCP")
                               & (n.tcp_flags == "0x0002")).sum())
    a_syn_from_attacker = int(((a1.ip_src == ATTACKER_IP) & (a1.protocol == "TCP")
                               & (a1.tcp_flags == "0x0002")).sum())

    mb = a1[a1.protocol == "MODBUS"]
    mb_fcs = sorted(int(x) for x in mb.modbus_func_code.dropna().unique())
    normal_fc = sorted(int(x) for x in n[n.protocol == "MODBUS"].modbus_func_code.dropna().unique())

    return {
        "session_duration_sec": round(dur, 1),
        "attack_duration_sec": round(dur1, 1),
        "normal_n_rows": int(len(n)),
        "attack_n_rows": int(len(a1)),
        "attack_n_modbus_rows": int(len(mb)),
        "attack_modbus_pct": round(len(mb) / len(a1) * 100, 3),
        "normal_arp_rows": n_arp,
        "attack_arp_rows": a_arp,
        "total_arp_rows": total_arp,
        "attack_arp_share_of_all_arp_pct": round(a_arp / total_arp * 100, 1),
        "normal_arp_rate": round(n_arp / dur, 2),
        "attack_arp_rate": round(a_arp / dur1, 2),
        "normal_partners": normal_partners,
        "attack_partners": attack_partners,
        "new_partners": new_partners,
        "normal_syn_from_attacker": n_syn_from_attacker,
        "attack_syn_from_attacker": a_syn_from_attacker,
        "attack_mb_function_codes": mb_fcs,
        "normal_fc": normal_fc,
        "attack_mb_nonstd_fc": [f for f in mb_fcs if f not in STANDARD_FC],
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
    parser = argparse.ArgumentParser(description="Compare a normal exchange vs. the address-scan attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting normal traffic example (192.168.0.1's only legitimate session)...")
    normal_example = extract_normal_example(df)
    print("Extracting address-scan examples (SYN probe, device ID probe, late read)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)
    print("Computing ARP-frame-rate time series (normal vs. attack, whole session)...")
    timeline = compute_timeline(df)

    packets = {"normal_example": normal_example, "attack_examples": attack_examples}
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

    print(f"Address scan summary: {stats['attack_n_rows']} rows, {stats['attack_modbus_pct']}% Modbus "
          f"({stats['attack_n_modbus_rows']} rows), {stats['attack_arp_share_of_all_arp_pct']}% of ALL "
          f"ARP traffic in the session, {len(stats['new_partners'])} new hosts contacted "
          f"(never seen in {ATTACKER_IP}'s normal traffic)")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Address Scan Diff</title>
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
    <h1 style="margin-top:8px">A known host that suddenly scans the rest of the subnet</h1>
    <p style="margin-top:10px">192.168.0.1 is not an outsider &mdash; it is a <b>legitimate host</b> on this
      network, with 1,252 normal-labeled rows of its own (all TCP/websocket traffic with
      <code class="mono">192.168.0.111</code>). During "address scan" the SAME host ARP-resolves and
      TCP-SYN-probes 5 hosts it has never contacted before, spread across almost the entire 99.4-minute
      session &mdash; not one short burst. Only in the last 0.075% of its footprint does it ever touch
      Modbus.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>Pipeline blind spot</h2>
      <p style="margin-top:6px">The AE model and the LLM prompt both only ever look at
        <code class="mono">protocol == "MODBUS"</code> rows. Here is what fraction of this specific
        attack that filter actually sees.</p>
    </div>
    <div class="ip-grid">
      <div class="ip-card attack">
        <div class="col-head attack">What the attack actually sends</div>
        <div class="ip-card-body">
          <div class="ip-note" style="font-size:13px">
            <b id="modbus-pct-big" style="font-size:28px; font-family:'Archivo',sans-serif; color:var(--attack);"></b>
            of this attack's 37,306 rows are Modbus &mdash; the rest is
            <b id="arp-count-inline"></b> ARP frames and TCP SYN/handshake noise.
          </div>
        </div>
      </div>
      <div class="ip-card attack">
        <div class="col-head attack">Share of ALL ARP traffic in the session</div>
        <div class="ip-card-body">
          <div class="ip-note" style="font-size:13px">
            <b id="arp-share-big" style="font-size:28px; font-family:'Archivo',sans-serif; color:var(--attack);"></b>
            of every ARP frame captured in the whole 99.4-minute session belongs to this one attack.
          </div>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; 192.168.0.1's own baseline</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; address scan</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Source IP: who does 192.168.0.1 talk to</h2>
      <p style="margin-top:6px">Not a new attacker IP this time &mdash; the same host, suddenly reaching
        far more of the subnet than it ever does normally.</p>
    </div>
    <div class="ip-grid" id="ip-grid"></div>
    <div class="shared-target-banner" id="shared-target-banner"></div>
  </section>

  <section>
    <div class="section-head">
      <h2>Protocol logic violation: why this traffic could not be legitimate</h2>
      <p style="margin-top:6px">Not "rare" or "different from baseline" - actually impossible under how
        this deployment's fixed, small network topology works (see the Smart Grid
        normal-behavior-baseline memory, Section 1). Checked but doesn't apply here: the
        "commanded-state-contradicts-reality" check used elsewhere in this series (naive-sensor-read's
        solar switch, force-listen's communication check) needs a WRITE or a claimed behavioral effect
        to falsify - this attack's Modbus tail is read-only (fc43, fc1), so there is no false command to
        check against real sensor readings.</p>
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
      <h2>Time series: ARP frame rate across the whole session</h2>
      <p style="margin-top:6px">Modbus is nearly silent in this attack (0.075% of its rows), so ARP -
        not Modbus request rate - is the metric that actually shows the difference. Not a sustained
        flood either: <span id="bursts-inline"></span> short, periodic ARP sweeps repeat across almost
        the entire session, roughly every 2 minutes.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal ARP baseline</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Address scan</span>
    </div>
    <div class="timeline-card">
      <div class="chart-svg-box" id="timeline-chart"></div>
      <p style="font-size:12px" id="timeline-caption"></p>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Statistics: address-scan window vs. the session's normal baseline</h2>
      <p style="margin-top:6px">192.168.0.1's behavior while attack_specific==1 is active (spans almost
        the whole session, t&nbsp;=&nbsp;29.5s&ndash;5860.5s) compared against its own normal traffic and
        the network's normal ARP/SYN baseline.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal baseline</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Address scan</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Address scan</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, all protocols (ARP/TCP/MODBUS), rows
      labeled <code class="mono">attack_specific == 1</code>. "Normal baseline" for 192.168.0.1
      specifically is its own attack_specific-NaN/0 rows (1,252 of them, all with 192.168.0.111);
      network-wide normal ARP/SYN rates are computed across all normal rows. This companion page follows
      the same treatment as the function-code-scan comparison (see <code class="mono">packet_compare_smartgrid_fcscan.py</code>).</p>
  </footer>

</div>
</div>

<script>
(function () {
  const DATA = __PACKET_DATA__;

  function esc(s) {
    return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  }

  function truncHex(hex) {
    if (!hex) return 'n/a';
    if (hex.length <= 74) return hex;
    return `${hex.slice(0, 34)} ... ${hex.slice(-16)} (${(hex.length - 2) / 2} bytes)`;
  }

  function frame(label, pkt, isAttack, note) {
    const rows = [
      ['time', pkt.time],
      ['src -> dst', `${pkt.ip_src} -> ${pkt.ip_dst}`],
      ['protocol/flags', pkt.modbus_func_code != null ? `Modbus fc${Math.trunc(pkt.modbus_func_code)}` : `TCP flags ${pkt.tcp_flags}`],
      ['pdu (hex)', truncHex(pkt.modbus_data), true],
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

  // ---- pipeline blind-spot numbers ----
  document.getElementById('modbus-pct-big').textContent = s.attack_modbus_pct + '%';
  document.getElementById('arp-count-inline').textContent = s.attack_arp_rows.toLocaleString();
  document.getElementById('arp-share-big').textContent = s.attack_arp_share_of_all_arp_pct + '%';

  // ---- normal column: 192.168.0.1's own baseline (ACK-only, no SYN, ever) ----
  const nCol = document.getElementById('normal-col');
  const ne = DATA.normal_example;
  nCol.innerHTML =
    frame('DATA (from 192.168.0.111)', ne.push, false,
      `192.168.0.111 pushes application data over an already-established connection.`) +
    frame('ACK (from 192.168.0.1)', ne.ack, false,
      `192.168.0.1 only ever ACKs &mdash; it never initiates. Across 1,252 normal rows and
       5,963 seconds, this host sends <b>zero</b> SYN packets and never contacts any IP besides
       192.168.0.111 (and one mDNS multicast).`) +
    `<div class="frame"><div class="frame-label">CONTEXT &mdash; what "normal" means for this specific host</div>
      <div class="decoded" style="border-top:none">
        This host's entire normal footprint is a passive role: it receives pushed data from
        192.168.0.111 and ACKs it, ${s.normal_syn_from_attacker} times a SYN, ${s.normal_arp_rows}
        ARP frames network-wide (0.08/s). It never touches Modbus, never contacts the RTU, and never
        appears in the "Known-good communication pairs" list from the function-code-scan baseline
        &mdash; it simply isn't part of that system at all, normally.
      </div>
    </div>`;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml =
    `<div><h3 style="margin-bottom:8px">1. SYN scan of a host it has never contacted<span class="flag">port scan, new host</span></h3>` +
    frame('SYN', ex.syn_probe, true,
      `192.168.0.1 opens a connection to 192.168.0.31 (the RTU) &mdash; a host that appears
       <b>nowhere</b> in this IP's normal traffic. Repeated against 5 different hosts, 614 times, across
       the whole session.`) +
    `</div>`;
  if (ex.device_id_probe) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. Once inside Modbus: device identification<span class="flag">fc43</span></h3>` +
      frame('REQUEST', ex.device_id_probe.request, true,
        `Report Device Identification (fc43) &mdash; one of only 28 Modbus rows in this entire 37,306-row attack.`) +
      frame('RESPONSE', ex.device_id_probe.response, true,
        `Target answers with its device ID object &mdash; the scan has moved from network-layer
         discovery to protocol-layer fingerprinting.`) +
      `</div>`;
  }
  if (ex.late_read) {
    attackHtml += `<div><h3 style="margin-bottom:8px">3. A late, maximal read<span class="flag">fc1, near session end</span></h3>` +
      frame('REQUEST', ex.late_read.request, true,
        `Read Coils, requested near the end of the scan window &mdash; this is the same
         "ask for far more than normal" pattern seen in the function-code scan.`) +
      frame('RESPONSE', ex.late_read.response, true,
        `259-byte reply vs. a 63-byte normal maximum &mdash; a bulk dump, not a routine poll.`) +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- protocol logic violation ----
  const logicPoints = [
    {
      title: 'A fixed, small network does not "discover" new peers at runtime',
      rule: 'This deployment has a small, fixed topology - one master, one RTU, and 192.168.0.1\'s own established session with 192.168.0.111. Legitimate devices don\'t probe for new neighbors during operation.',
      body: `192.168.0.1 goes from <b>1</b> normal partner (192.168.0.111, passive ACK-only) to
             <b>${s.attack_partners.length}</b> contacted hosts during this attack, <b>${s.new_partners.length}
             of them never touched before</b> - including the RTU itself. No legitimate role for this
             host involves finding new devices on the network.`,
    },
    {
      title: 'Self-identification mid-scan is an engineering action, not reconnaissance follow-through',
      rule: 'Asking a device to report its own identity (fc43) belongs to a commissioning/maintenance workflow performed by a known engineering workstation - never triggered automatically by a network scan.',
      body: `Once the scan does reach Modbus, one of its first actions is a Report Device Identification
             (fc43) probe - a request type that has no place in a routine value-monitoring pipeline
             and no place following automatically from a host-discovery sweep either.`,
    },
    {
      title: 'A bulk read has no monitoring purpose',
      rule: 'Every real measurement point in this deployment is read one value at a time (quantity always 1) - a monitoring task never needs more than the single current reading.',
      body: `The late read near the end of this scan's window returns a 259-byte reply vs. a 63-byte
             normal maximum - reading far more data than any legitimate monitoring purpose requires,
             the same "ask for far more than normal" pattern seen in naive-sensor-read.`,
    },
  ];
  document.getElementById('logic-grid').innerHTML = logicPoints.map(p => `
    <div class="logic-card">
      <div class="logic-card-title">${esc(p.title)}</div>
      <div class="logic-card-rule">${p.rule}</div>
      <div class="logic-card-body">${p.body}</div>
    </div>`).join('');

  // ---- IP highlight cards ----
  const ipGrid = document.getElementById('ip-grid');
  function chip(ip, cls) { return `<span class="ip-chip${cls}">${ip}</span>`; }
  ipGrid.innerHTML = `
    <div class="ip-card normal">
      <div class="col-head normal">192.168.0.1's normal partners</div>
      <div class="ip-card-body">
        <div class="ip-chip-row">${s.normal_partners.map(ip => chip(ip, ' role-active')).join('')}</div>
        <div class="ip-note">Exactly <b>${s.normal_partners.length} host</b> in 5,963 seconds of normal
          traffic. No SYN packets, no ARP sweeps &mdash; just steady ACKs on one existing session.</div>
      </div>
    </div>
    <div class="ip-card attack">
      <div class="col-head attack">Contacted during address scan</div>
      <div class="ip-card-body">
        <div class="ip-chip-row">${s.attack_partners.map(ip => chip(ip, s.new_partners.includes(ip) ? ' role-active' : '')).join('')}</div>
        <div class="ip-note"><b>${s.new_partners.length} of ${s.attack_partners.length}</b> hosts
          (highlighted) have <b>never</b> been contacted by 192.168.0.1 before &mdash; including the RTU
          itself (192.168.0.31). Only 192.168.0.111 (not highlighted) is a repeat, legitimate partner.</div>
      </div>
    </div>`;
  document.getElementById('shared-target-banner').innerHTML =
    `The degree of this host jumps from <b>1</b> normal partner to <b>${s.attack_partners.length}</b>
     during the scan &mdash; a fan-out, not a rate change. This is exactly what MITRE ATT&amp;CK for ICS
     <a href="https://attack.mitre.org/techniques/T0846/" target="_blank" rel="noopener" style="color:inherit">T0846 Remote System Discovery</a> describes.`;

  // ---- detection signals ----
  const signals = [
    {
      title: 'Host fan-out (distinct partners)', tag: 'categorical',
      normal: `${s.normal_partners.length}`, attack: `${s.attack_partners.length}`,
      note: `192.168.0.1 contacts exactly 1 host (192.168.0.111) in its entire normal baseline. During
             this attack it reaches ${s.attack_partners.length}, ${s.new_partners.length} of them never
             contacted before - a degree change, not a rate change.`,
    },
    {
      title: 'SYN packets sent (this host)', tag: 'categorical',
      normal: '0', attack: `${s.attack_syn_from_attacker}`,
      note: `192.168.0.1 never initiates a connection in its normal traffic (purely passive - only
             ACKs). This attack sends ${s.attack_syn_from_attacker} SYN packets across 5 targets.`,
    },
    {
      title: 'ARP frame rate', tag: 'statistical',
      normal: `${s.normal_arp_rate}/s`, attack: `${s.attack_arp_rate}/s`,
      note: `${Math.round(s.attack_arp_rate / s.normal_arp_rate)}&times; the normal network-wide ARP
             rate - this attack alone accounts for ${s.attack_arp_share_of_all_arp_pct}% of every ARP
             frame captured in the whole session.`,
    },
    {
      title: 'Modbus function codes touched', tag: 'categorical',
      normal: `fc${s.normal_fc.join(', fc')}`, attack: `fc${s.attack_mb_function_codes.join(', fc')}`,
      note: `fc2, fc8 and fc43 (Report Device Identification) never appear in 47,198 normal Modbus rows
             - though this only covers the 0.075% of this attack that is Modbus at all (see the
             pipeline blind-spot section above).`,
    },
    {
      title: 'Modbus visibility to the pipeline', tag: 'statistical',
      normal: '100%', attack: `${s.attack_modbus_pct}%`,
      note: `The AE model and LLM prompt only ever see protocol=="MODBUS" rows. For normal traffic
             that's everything; for this attack it's ${s.attack_n_modbus_rows} of ${s.attack_n_rows.toLocaleString()}
             rows - the other 94.3% (ARP) is structurally invisible to the current pipeline.`,
    },
    {
      title: 'Late read reply size', tag: 'statistical',
      normal: 'max 63 bytes', attack: '259 bytes',
      note: `Once the scan does touch Modbus, the same "ask for far more than normal" pattern from the
             function-code-scan and naive-sensor-read attacks shows up again: a bulk read near the end
             of the scan window, not a routine single-value poll.`,
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

  // ---- time series (ARP frame rate across the whole session) ----
  function drawTimeSeries(tl) {
    const width = 1000, height = 260;
    const pad = {top: 16, right: 16, bottom: 30, left: 40};
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

    function areaPath(counts) {
      const pts = tl.bin_centers.map((t, i) => `${x(t)},${y(counts[i])}`);
      return `M${pad.left},${y(0)} L${pts.join(' L')} L${x(dur)},${y(0)} Z`;
    }

    const attackArea = document.createElementNS(svg.namespaceURI, 'path');
    attackArea.setAttribute('d', areaPath(tl.attack_counts));
    attackArea.setAttribute('fill', aColor); attackArea.setAttribute('fill-opacity', '0.55');
    attackArea.setAttribute('stroke', aColor); attackArea.setAttribute('stroke-width', '1');
    svg.appendChild(attackArea);

    const normalArea = document.createElementNS(svg.namespaceURI, 'path');
    normalArea.setAttribute('d', areaPath(tl.normal_counts));
    normalArea.setAttribute('fill', nColor); normalArea.setAttribute('fill-opacity', '0.55');
    normalArea.setAttribute('stroke', nColor); normalArea.setAttribute('stroke-width', '1.5');
    svg.appendChild(normalArea);

    return svg;
  }

  const tl = DATA.timeline;
  document.getElementById('bursts-inline').innerHTML = `<b>${tl.n_arp_bursts}</b>`;
  document.getElementById('timeline-chart').appendChild(drawTimeSeries(tl));
  document.getElementById('timeline-caption').innerHTML =
    `Normal ARP traffic averages <b class="mono" style="color:var(--normal)">${tl.normal_avg_per_bin}</b>
     frames per 10s bin, network-wide - flat, essentially silent. This attack forms
     <b class="mono" style="color:var(--attack)">${tl.n_arp_bursts}</b> distinct sweeps, each
     ~<b class="mono" style="color:var(--attack)">${tl.arp_burst_avg_dur_sec}s</b> long, spaced
     ~<b class="mono" style="color:var(--attack)">${tl.arp_burst_avg_gap_sec}s</b> apart - active in
     <b class="mono" style="color:var(--attack)">${tl.attack_active_bin_pct}%</b> of the session's bins,
     a repeating cycle, not a single burst (like the function-code scan) or a constant flood.`;

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
    {label: 'Distinct hosts contacted', normal: s.normal_partners.length, attack: s.attack_partners.length},
    {label: 'SYN packets sent (ever, whole session)', normal: s.normal_syn_from_attacker, attack: s.attack_syn_from_attacker},
    {label: 'ARP frames sent', normal: s.normal_arp_rows, attack: s.attack_arp_rows, logScale: true},
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
    ['Distinct hosts contacted by 192.168.0.1', `${s.normal_partners.length} (${s.normal_partners.join(', ')})`, `${s.attack_partners.length} (${s.attack_partners.join(', ')})`, true],
    ['New hosts (never contacted before)', '0', `${s.new_partners.length} (${s.new_partners.join(', ')})`, true],
    ['SYN packets from 192.168.0.1 (whole session)', `${s.normal_syn_from_attacker}`, `${s.attack_syn_from_attacker}`, true],
    ['ARP frames sent', `${s.normal_arp_rows} (${s.normal_arp_rate}/s network-wide)`, `${s.attack_arp_rows} (${s.attack_arp_rate}/s)`, true],
    ['Share of ALL session ARP traffic', '—', `${s.attack_arp_share_of_all_arp_pct}% (${s.attack_arp_rows} of ${s.total_arp_rows})`, true],
    ['Rows that are Modbus (visible to the pipeline)', '100% (47,198 of 47,198)', `${s.attack_modbus_pct}% (${s.attack_n_modbus_rows} of ${s.attack_n_rows.toLocaleString()})`, true],
    ['Modbus function codes touched', `${s.normal_fc.length} (fc ${s.normal_fc.join(', ')})`, `${s.attack_mb_function_codes.length} (fc ${s.attack_mb_function_codes.join(', ')})`, true],
    ['Attack window span', '—', `${s.attack_duration_sec.toLocaleString()}s (${(s.attack_duration_sec/60).toFixed(1)} min, ${(s.attack_duration_sec/s.session_duration_sec*100).toFixed(0)}% of the session)`, true],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
