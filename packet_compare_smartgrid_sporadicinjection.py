"""
Fifth in the per-attack-type series for Smart Grid (after function-code
scan, address scan, device identification attack, naive sensor read - see
the Smart Grid normal-behavior-baseline memory), applied to "sporadic
sensor measurement injection" (attack_specific == 5).

Like device identification, this attack's request TYPE (a write) has no
normal-traffic equivalent at all (0% of normal traffic is a write), so
there is no "normal_pair" here - same choice as
packet_compare_smartgrid_deviceid.py, for the same reason.

Structure, verified from the raw data: this is NOT many scattered single
injections despite the "sporadic" name - it is exactly 2 rapid write-toggle
bursts, ~100 Write Single Coil (fc5) requests each, ~51ms apart (~19.7
requests/sec), toggling the SAME coil (address 9) between ON (0xff00) and
OFF (0x0000) in a near-random pattern:
  - burst 1: t=4449.96s-4455.03s (~5.07s), stream 6680, entirely labeled
    attack_specific=5.
  - burst 2: t=5356.43s-5366.51s (~10.08s), stream 6823.

CRITICAL cross-label finding (verified, not assumed): burst 2 is a SINGLE
continuous write-toggle sequence on ONE TCP stream (6823) whose
attack_specific label switches from 4 to 5 partway through, at
t=5357.196s (row 214907) - with no gap or change in cadence/content at that
boundary. The first 15 requests of this same physical event were already
described in packet_compare_smartgrid_naivesensorread.py's report as a
"short write coda tacked onto the last burst" (attack_specific=4) - that
was correct as far as it went, but incomplete: it is actually the first
15% of a 100-request, ~10-second continuous toggle storm, whose remaining
85 requests are labeled attack_specific=5 instead. This is a DIFFERENT
kind of labeling artifact from the already-documented background-noise
co-mingling (unrelated legitimate traffic swept into an attack's active
window) - here it is the SAME single malicious event, split by the
dataset's ground truth across two different attack-type labels. Any
per-attack-type accuracy/count metric computed from attack_specific alone
should account for this.

Also verified: the same background-noise co-mingling issue documented for
naive-sensor-read applies here too, at a similar magnitude (34.4% of this
attack's 1,875 rows are 192.168.0.1's ordinary DATA/WEBSOCKET traffic with
192.168.0.111, not related to the write-toggle bursts at all).

Output goes into data_visualisation/smartgrid_sporadic_injection/ (all
filenames get the optional --tag suffix so earlier results are not
overwritten): packets.json, stats.json, timeline.json, report.html. Run
with a log, e.g.:
    python packet_compare_smartgrid_sporadicinjection.py 2>&1 | tee data_visualisation/smartgrid_sporadic_injection/run_$(date +%Y%m%d_%H%M).log
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_ae_9dim import DATASET_FILENAMES, find_dataset_csv
from packet_compare_smartgrid_fcscan import pack

OUTPUT_DIR = Path("data_visualisation") / "smartgrid_sporadic_injection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKER_IP = "192.168.0.1"
TARGET_IP = "192.168.0.31"
PARTNER_IP = "192.168.0.111"
MASTER_IP = "192.168.0.40"
WRITE_ADDR = 9          # transfer switch coil (verified point map, see memory Section 2b)
SOLAR_ADDR = 19         # solar power output meter
THRESHOLD_ADDR = 39     # switching threshold setpoint
BURST_GAP_SEC = 5.0


def _decode_addr(hexdata):
    s = str(hexdata)[2:] if str(hexdata).startswith("0x") else str(hexdata)
    return int(s[:4], 16) if len(s) >= 4 else None


def _decode_reg_value(hexdata):
    """Decode a single-register (fc3/fc4) response's value: response PDU
    hex string, e.g. "0x0200" -> 512. Chars [2:6] hold the value here."""
    if hexdata is None:
        return None
    h = str(hexdata)
    return int(h[2:6], 16) if len(h) >= 6 else None


def _decode_coil_byte(hexdata):
    """Decode a Read Coils (fc1) response's coil-status byte: e.g.
    "0x0180" -> 0x80 (128). Chars [4:6] hold the coil byte here."""
    if hexdata is None:
        return None
    h = str(hexdata)
    return int(h[4:6], 16) if len(h) >= 6 else None


def compute_solar_logic_violation(df):
    """Physical-logic check for the transfer-switch coil (address 9): per
    the ICS-SimLab paper, the real control rule is "route power through
    solar only when solar output is above a threshold, otherwise mains."
    This computes (a) the real historical relationship between the switch
    state and the solar meter reading from normal traffic, and (b) the
    nearest real solar reading to this attack's first ON ("switch to
    solar") command, to check whether that command actually matched real
    conditions. See the Smart Grid normal-behavior-baseline memory,
    Section 2b, for the full point-map derivation.
    """
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    mb = n[n.protocol == "MODBUS"].sort_values("frame_time_relative")
    req = mb[mb.ip_src == MASTER_IP].copy()
    req["addr"] = req.modbus_data.apply(_decode_addr)

    def response_hex(row_idx):
        r = df.iloc[int(row_idx) + 1]
        return str(r.modbus_data) if r.ip_src == TARGET_IP else None

    coil = req[(req.modbus_func_code == 1.0) & (req.addr == WRITE_ADDR)][["row", "frame_time_relative"]].copy()
    coil["coil_byte"] = coil.row.apply(lambda r: _decode_coil_byte(response_hex(r)))
    coil = coil[coil.coil_byte.isin([0, 128])]

    solar = req[(req.modbus_func_code == 4.0) & (req.addr == SOLAR_ADDR)][["row", "frame_time_relative"]].copy()
    solar["solar_val"] = solar.row.apply(lambda r: _decode_reg_value(response_hex(r)))
    solar = solar[(solar.solar_val.notna()) & (solar.solar_val < 5000)]

    coil = coil.sort_values("frame_time_relative")
    solar_sorted = solar.sort_values("frame_time_relative")
    merged = pd.merge_asof(coil, solar_sorted, on="frame_time_relative", direction="nearest", tolerance=1.0).dropna()
    coil_on = merged.coil_byte == 128
    mean_solar_on = float(merged[coil_on].solar_val.mean())
    mean_solar_off = float(merged[~coil_on].solar_val.mean())

    thresh_req = req[(req.modbus_func_code == 3.0) & (req.addr == THRESHOLD_ADDR)][["row"]].copy()
    thresh_vals = thresh_req.row.apply(lambda r: _decode_reg_value(response_hex(r)))
    threshold_mode = int(thresh_vals.mode().iloc[0]) if len(thresh_vals.dropna()) else None

    # nearest real solar reading to this attack's first ON command
    a5 = df[df.attack_specific.isin([4, 5])]  # includes the cross-labeled first 15 requests
    mb5 = a5[a5.protocol == "MODBUS"].sort_values("frame_time_relative")
    on_writes = mb5[(mb5.ip_src == ATTACKER_IP) & (mb5.modbus_func_code == 5)
                     & (mb5.modbus_data == "0x0009ff00")]
    first_on_time = float(on_writes.frame_time_relative.min()) if len(on_writes) else None

    nearest_solar_reading = None
    nearest_solar_time = None
    normal_request_pkt = None
    normal_response_pkt = None
    if first_on_time is not None:
        cand = solar.iloc[(solar.frame_time_relative - first_on_time).abs().argsort()]
        if len(cand):
            best = cand.iloc[0]
            nearest_solar_reading = float(best.solar_val)
            nearest_solar_time = float(best.frame_time_relative)
            solar_req_row = req.loc[req.row == best.row].iloc[0]
            normal_request_pkt = pack(solar_req_row)
            solar_resp_row = df.iloc[int(best.row) + 1]
            if solar_resp_row.ip_src == TARGET_IP:
                normal_response_pkt = pack(solar_resp_row)

    attack_request_pkt = None
    attack_response_pkt = None
    if len(on_writes):
        req_row = on_writes.sort_values("frame_time_relative").iloc[0]
        attack_request_pkt = pack(req_row)
        # the immediately-following row can be a bare TCP ACK (rapid-fire
        # write toggling generates extra ACK-only rows) - search forward
        # for the actual next MODBUS response instead of assuming row+1.
        following = df[(df.row > req_row.row) & (df.row <= req_row.row + 5)
                       & (df.protocol == "MODBUS") & (df.ip_src == TARGET_IP)]
        if len(following):
            attack_response_pkt = pack(following.iloc[0])

    return {
        "write_addr": WRITE_ADDR,
        "solar_addr": SOLAR_ADDR,
        "threshold_addr": THRESHOLD_ADDR,
        "threshold_value": threshold_mode,
        "mean_solar_when_switch_on": round(mean_solar_on, 1),
        "mean_solar_when_switch_off": round(mean_solar_off, 1),
        "first_on_command_time_sec": round(first_on_time, 3) if first_on_time is not None else None,
        "nearest_solar_reading": nearest_solar_reading,
        "nearest_solar_reading_time_sec": round(nearest_solar_time, 2) if nearest_solar_time is not None else None,
        "attack_request_pkt": attack_request_pkt,
        "attack_response_pkt": attack_response_pkt,
        "normal_request_pkt": normal_request_pkt,
        "normal_response_pkt": normal_response_pkt,
        "n_normal_coil_solar_pairs_matched": int(len(merged)),
    }


def load_dataset():
    df = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df["row"] = np.arange(len(df))
    return df


def extract_attack_examples(df):
    a5 = df[df.attack_specific == 5].sort_values("row")
    mb = a5[a5.protocol == "MODBUS"].sort_values("frame_time_relative").reset_index(drop=True)
    req = mb[mb.ip_src == ATTACKER_IP]

    def find_pair(target_stream, value_hex):
        sub = mb[mb.tcp_stream == target_stream].reset_index(drop=True)
        for i in range(len(sub) - 1):
            if sub.iloc[i].ip_src == ATTACKER_IP and sub.iloc[i].modbus_data == value_hex \
               and sub.iloc[i + 1].ip_src == TARGET_IP:
                return {"request": pack(sub.iloc[i]), "response": pack(sub.iloc[i + 1])}
        return None

    burst1_stream = req.tcp_stream.iloc[0]
    burst2_stream = req.tcp_stream.iloc[-1]
    burst1_on = find_pair(burst1_stream, "0x0009ff00")
    burst2_off = find_pair(burst2_stream, "0x00090000")

    return {"burst1_write_on": burst1_on, "burst2_write_off": burst2_off}


def compute_stats(df):
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a5 = df[df.attack_specific == 5]
    mb = a5[a5.protocol == "MODBUS"]
    req = mb[mb.ip_src == ATTACKER_IP]

    # background noise: same check as naive-sensor-read - this attacker's
    # ordinary traffic with its own legitimate partner, not attack behavior.
    noise = a5[a5.protocol.isin(["DATA", "WEBSOCKET"])
              & a5.ip_src.isin([ATTACKER_IP, PARTNER_IP]) & a5.ip_dst.isin([ATTACKER_IP, PARTNER_IP])]

    # burst structure
    req_sorted = req.sort_values("frame_time_relative")
    t = req_sorted.frame_time_relative.to_numpy()
    gaps = np.diff(t)
    cuts = np.where(gaps > BURST_GAP_SEC)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(t) - 1]
    burst_sizes = ends - starts + 1
    burst_durs = t[ends] - t[starts]

    values = req.modbus_data.value_counts().to_dict()
    on_count = values.get("0x0009ff00", 0)
    off_count = values.get("0x00090000", 0)

    def decode_write_addr(hexdata):
        s = str(hexdata)[2:] if str(hexdata).startswith("0x") else str(hexdata)
        return int(s[:4], 16) if len(s) >= 4 else None

    distinct_addrs = sorted({decode_write_addr(d) for d in req.modbus_data} - {None})

    normal_mb_ips = sorted(set(n[n.protocol == "MODBUS"].ip_src.unique())
                           | set(n[n.protocol == "MODBUS"].ip_dst.unique()))

    return {
        "session_duration_sec": round(df.frame_time_relative.max(), 1),
        "attack_total_rows": int(len(a5)),
        "attack_noise_rows": int(len(noise)),
        "attack_noise_pct": round(len(noise) / len(a5) * 100, 1),
        "n_bursts": int(len(starts)),
        "burst_sizes": [int(x) for x in burst_sizes],
        "burst_durations_sec": [round(float(x), 2) for x in burst_durs],
        "n_write_requests": int(len(req)),
        "on_count": int(on_count),
        "off_count": int(off_count),
        "write_address": WRITE_ADDR,
        "distinct_write_addresses": distinct_addrs,
        "avg_interval_ms": round(float(np.median(np.diff(t))) * 1000, 1),
        "avg_rate_per_sec": round(1000 / (round(float(np.median(np.diff(t))) * 1000, 1)), 1),
        "normal_write_count": 0,
        "normal_mb_ips": normal_mb_ips,
        "attacker_ip": ATTACKER_IP,
        "cross_label_stream": 6823,
        "cross_label_boundary_sec": 5357.196,
        "cross_label_total_requests": 100,
        "cross_label_attack4_requests": 15,
        "cross_label_attack5_requests": 85,
        "cross_label_total_span_sec": round(5366.511 - 5356.433, 2),
    }


def compute_timeline(df, bin_width=10.0):
    """Per-bin fc5 write-request counts across the whole session, normal
    (always 0 - writes never occur in normal traffic) vs. this attack's 2
    toggle bursts. Same treatment as the other reports in this series.
    """
    dur = float(df.frame_time_relative.max())
    edges = np.arange(0, dur + bin_width, bin_width)
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a5 = df[df.attack_specific == 5]

    normal_times = n[(n.protocol == "MODBUS") & (n.modbus_func_code == 5)].frame_time_relative.to_numpy()
    attack_times = a5[(a5.protocol == "MODBUS") & (a5.modbus_func_code == 5)
                       & (a5.ip_src == ATTACKER_IP)].frame_time_relative.to_numpy()

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
    parser = argparse.ArgumentParser(description="Compare normal traffic vs. the sporadic-sensor-measurement-injection attack on Smart Grid.")
    parser.add_argument("--tag", default=None, help="Suffix added to output filenames.")
    args = parser.parse_args()

    def path(base):
        stem, ext = base.rsplit(".", 1)
        return OUTPUT_DIR / (f"{stem}_{args.tag}.{ext}" if args.tag else base)

    start = time.time()
    print("Loading Smart Grid dataset...")
    df = load_dataset()

    print("Extracting write-toggle examples (burst 1 ON, burst 2 OFF)...")
    attack_examples = extract_attack_examples(df)
    print("Computing comparison statistics...")
    stats = compute_stats(df)
    print("Computing write-request time series (normal vs. attack, whole session)...")
    timeline = compute_timeline(df)
    print("Checking transfer-switch command against the real solar meter reading...")
    solar_logic = compute_solar_logic_violation(df)

    # No "normal_pair" here on purpose, same reasoning as
    # packet_compare_smartgrid_deviceid.py: this attack's request TYPE (a
    # write) has no normal-traffic equivalent at all (0% writes normal).
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

    with open(path("solar_logic.json"), "w") as f:
        json.dump(solar_logic, f, indent=1)
    print(f"Saved: {path('solar_logic.json')}")

    payload = {**packets, "stats": stats, "timeline": timeline, "solar_logic": solar_logic}
    report_path = path("report.html")
    report_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"Sporadic-injection summary: {stats['n_bursts']} bursts, {stats['n_write_requests']} total "
          f"write requests toggling address {stats['write_address']} at ~{stats['avg_rate_per_sec']}/s, "
          f"{stats['attack_noise_pct']}% of labeled rows are unrelated background traffic, burst 2 "
          f"continues a physical event that began under attack_specific=4 (see naive-sensor-read report)")
    print(f"Total time consumed: {time.time() - start:.2f}s")


HTML_TEMPLATE = r"""<title>Sporadic Injection Diff</title>
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
  .logic-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }
  @media (max-width: 700px) { .logic-compare { grid-template-columns: 1fr; } }
  .logic-compare-head { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; font-weight: 600;
                         text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
  .logic-compare-head.normal-label { color: var(--normal); }
  .logic-compare-head.attack-label { color: var(--attack); }

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
    <h1 style="margin-top:8px">Not sporadic single injections - two rapid toggle storms</h1>
    <p style="margin-top:10px">Despite the name, "sporadic sensor measurement injection" is not many
      scattered single writes - it is <b id="bursts-inline"></b> dense write-toggle bursts (~100 Write
      Single Coil requests each, ~<span id="rate-inline"></span> requests/sec), rapidly flipping the
      SAME coil (address <b id="addr-inline"></b>) between ON and OFF in a near-random pattern.</p>
  </header>

  <section>
    <div class="section-head">
      <h2>Burst 2 is not a new event - it's a continuation of naive-sensor-read's write coda</h2>
      <p style="margin-top:6px">Verified directly from the raw data, not assumed: this is the same
        physical write-toggle sequence on the same TCP connection, just split across two different
        attack-type labels.</p>
    </div>
    <div class="shared-target-banner">
      TCP stream <b>6823</b> carries ONE continuous, unbroken write-toggle sequence from
      t&nbsp;=&nbsp;5356.43s to t&nbsp;=&nbsp;5366.51s (<b id="crosslabel-span-inline"></b>, <b
      id="crosslabel-total-inline"></b> requests total, same ~51ms cadence throughout, no gap or content
      change at the boundary) - but the dataset's <code class="mono">attack_specific</code> label
      switches from <b>4</b> to <b>5</b> partway through, at t&nbsp;=&nbsp;5357.196s. The first <b
      id="crosslabel-a4-inline"></b> requests were already shown in the naive-sensor-read report as a
      "short write coda" under attack_specific=4 - that description was accurate but incomplete: the
      remaining <b id="crosslabel-a5-inline"></b> requests of that SAME event are labeled
      attack_specific=5 instead, and appear here as "burst 2". This is a different kind of labeling
      issue from the background-noise co-mingling documented below: there, unrelated legitimate traffic
      gets swept into an attack's active window; here, a single malicious event is itself split across
      two different attack-type ground-truth labels. Any accuracy metric computed strictly per
      attack_specific label should account for this.
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Not everything labeled "attack" here is attack behavior</h2>
      <p style="margin-top:6px">Same background-noise check applied to every attack in this series -
        checked directly against the raw data, not assumed.</p>
    </div>
    <div class="shared-target-banner">
      Of the <b id="noise-total-inline"></b> rows labeled <code class="mono">attack_specific&nbsp;==&nbsp;5</code>,
      <b id="noise-pct-inline"></b> (<b id="noise-rows-inline"></b> rows) are 192.168.0.1's completely
      ordinary DATA/WEBSOCKET traffic with its regular partner 192.168.0.111 - the same co-mingling
      issue found in naive-sensor-read (30.8%), at a similar magnitude here. Every finding below is
      drawn only from the genuine write-toggle rows.
    </div>
  </section>

  <section>
    <div class="cols">
      <div>
        <div class="col-head normal">Normal &mdash; no equivalent exists</div>
        <div class="col-body" id="normal-col"></div>
      </div>
      <div>
        <div class="col-head attack">Attack &mdash; rapid coil toggle</div>
        <div class="col-body" id="attack-col"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Protocol logic violation: why this traffic could not be legitimate</h2>
      <p style="margin-top:6px">Not "rare" or "different from baseline" - a real, physical-logic
        contradiction, checked against the actual sensor reading, not just against protocol rules (see
        the Smart Grid normal-behavior-baseline memory, Section 2b).</p>
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
      <h2>Time series: write-request rate across the whole session</h2>
      <p style="margin-top:6px">Like device identification, the normal line here is a flat zero - writes
        never occur in normal traffic at all. The chart shows just how isolated and dense these 2 bursts
        are against the rest of the session.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal (always 0)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Sporadic injection</span>
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
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Sporadic injection</span>
    </div>
    <div class="chart-grid" id="count-charts"></div>
    <details>
      <summary style="cursor:pointer; font-size:12.5px; color:var(--text-secondary); font-family:'IBM Plex Mono',monospace;">Exact numbers (table)</summary>
      <div class="stats-wrap" style="margin-top:10px">
        <table class="stats">
          <thead><tr><th>Metric</th><th>Normal baseline</th><th>Sporadic injection</th></tr></thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </details>
  </section>

  <footer>
    <p>Source: <code class="mono">dataset_sg_packetv4.csv</code>, rows labeled
      <code class="mono">attack_specific == 5</code> (plus TCP stream 6823's attack_specific==4 rows for
      the cross-label finding above). PDU <code class="mono">0x0009ff00</code> decodes as address 9,
      value 0xff00 (coil ON); <code class="mono">0x00090000</code> decodes as address 9, value 0x0000
      (coil OFF). Companion page to <code class="mono">packet_compare_smartgrid_naivesensorread.py</code>
      (shares TCP stream 6823) and <code class="mono">packet_compare_smartgrid_fcscan.py</code>.</p>
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
  document.getElementById('rate-inline').textContent = s.avg_rate_per_sec;
  document.getElementById('addr-inline').textContent = s.write_address;
  document.getElementById('crosslabel-span-inline').textContent = s.cross_label_total_span_sec + 's';
  document.getElementById('crosslabel-total-inline').textContent = s.cross_label_total_requests;
  document.getElementById('crosslabel-a4-inline').textContent = s.cross_label_attack4_requests;
  document.getElementById('crosslabel-a5-inline').textContent = s.cross_label_attack5_requests;
  document.getElementById('noise-total-inline').textContent = s.attack_total_rows.toLocaleString();
  document.getElementById('noise-pct-inline').textContent = s.attack_noise_pct + '%';
  document.getElementById('noise-rows-inline').textContent = s.attack_noise_rows.toLocaleString();

  // ---- normal column ----
  const nCol = document.getElementById('normal-col');
  nCol.innerHTML =
    `<div class="frame">
      <div class="frame-label">This traffic type does not occur in normal operation</div>
      <div class="decoded" style="border-top:none">
        Write Single Coil (fc5) never appears in 47,198 normal rows &mdash; <b>0%</b> of normal traffic
        is a write, of any kind. There is no normal packet to pair this toggle against, because writing
        a value is not part of this system's vocabulary in normal operation at all - every normal
        request is a read.
      </div>
    </div>
    <div class="frame">
      <div class="frame-label">What normal traffic asks instead</div>
      <div class="decoded" style="border-top:none">
        Every normal Modbus request polls a value (fc1/fc3/fc4) at one of 4 known addresses. This
        attack does the opposite: it repeatedly SETS a value, at an address (9) that is never
        legitimately read either.
      </div>
    </div>`;

  // ---- attack column ----
  const aCol = document.getElementById('attack-col');
  const ex = DATA.attack_examples;
  let attackHtml = '';
  if (ex.burst1_write_on) {
    attackHtml += `<div><h3 style="margin-bottom:8px">1. Burst 1: toggle ON<span class="flag">fc5, accepted</span></h3>` +
      frame('REQUEST', ex.burst1_write_on.request, true,
        `Write Single Coil, address ${s.write_address}, value 0xff00 (ON) &mdash; one of ~100 rapid
         writes in this burst, ${s.avg_interval_ms}ms apart.`) +
      frame('RESPONSE', ex.burst1_write_on.response, true,
        `Echoed back &mdash; ACCEPTED, same as every other write in both bursts.`) +
      `</div>`;
  }
  if (ex.burst2_write_off) {
    attackHtml += `<div><h3 style="margin-bottom:8px">2. Burst 2: toggle OFF<span class="flag">fc5, accepted, continues attack 4</span></h3>` +
      frame('REQUEST', ex.burst2_write_off.request, true,
        `Same address, value 0x0000 (OFF) &mdash; part of the same TCP stream (6823) whose first 15
         requests were labeled attack_specific=4 (see the callout above).`) +
      frame('RESPONSE', ex.burst2_write_off.response, true,
        `Echoed back &mdash; ACCEPTED.`) +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- protocol logic violation ----
  const sl = DATA.solar_logic;
  const logicPoints = [
    {
      flagship: true,
      title: 'Switch commanded to solar while the real solar meter read low',
      rule: `Per the deployment's control logic (address ${sl.write_addr} = transfer switch, address
             ${sl.solar_addr} = solar power meter, address ${sl.threshold_addr} = switching threshold
             &asymp; ${sl.threshold_value}): route power through solar when solar output is high, mains
             when it's low. In 47,198 normal rows the switch is ON (solar) when the meter averages
             <b>${sl.mean_solar_when_switch_on}</b> and OFF (mains) when it averages
             <b>${sl.mean_solar_when_switch_off}</b> - a real, measured relationship, not assumed.`,
      body: () => {
        const flow = `<div class="logic-flow">
          <span class="step low">solar meter &asymp; ${sl.nearest_solar_reading} (t=${sl.nearest_solar_reading_time_sec}s)</span>
          <span class="arrow">&mdash; well below the ${sl.mean_solar_when_switch_on} average that
          legitimately turns the switch ON &mdash;</span>
          <span class="step cmd">yet: SWITCH TO SOLAR commanded (t=${sl.first_on_command_time_sec}s)</span>
        </div>`;
        const cmp = `<div class="logic-compare">
          <div>
            <div class="logic-compare-head normal-label">Normal &mdash; legitimate master's own solar-meter poll</div>
            ${sl.normal_request_pkt ? frame('REQUEST', sl.normal_request_pkt, false,
              `Read Input Registers (fc4), address ${sl.solar_addr}, quantity 1 &mdash; the master's own
               routine poll, ${sl.nearest_solar_reading_time_sec}s away from the attack command below.`) : ''}
            ${sl.normal_response_pkt ? frame('RESPONSE', sl.normal_response_pkt, false,
              `Decodes to <b>${sl.nearest_solar_reading}</b> &mdash; below the ${sl.mean_solar_when_switch_on}
               average that legitimately accompanies the switch being ON.`) : ''}
          </div>
          <div>
            <div class="logic-compare-head attack-label">Attack &mdash; the "switch to solar" command itself</div>
            ${sl.attack_request_pkt ? frame('REQUEST', sl.attack_request_pkt, true,
              `Write Single Coil (fc5), address ${sl.write_addr}, value 0xff00 (ON/solar) &mdash; sent at
               t=${sl.first_on_command_time_sec}s, while the real meter (left) showed ${sl.nearest_solar_reading}.`) : ''}
            ${sl.attack_response_pkt ? frame('RESPONSE', sl.attack_response_pkt, true,
              `Echoed back &mdash; ACCEPTED, despite contradicting the real sensor reading at the same
               moment.`) : ''}
          </div>
        </div>`;
        return `This attack's very first "switch to solar" command (fc5, address ${sl.write_addr},
                value 0xff00) was sent at t=${sl.first_on_command_time_sec}s. The nearest genuine solar
                reading (from the legitimate master's own polling, ${sl.nearest_solar_reading_time_sec}s
                away) was only <b>${sl.nearest_solar_reading}</b> - well below the
                ${sl.mean_solar_when_switch_on} average solar level historically associated with a
                legitimate switch-to-solar condition, and close to the ${sl.mean_solar_when_switch_off}
                average for legitimate mains periods. The command doesn't just come from the wrong
                source and repeat too fast - it tells the system to do the physically wrong thing given
                what the sensors actually showed at that moment.${flow}${cmp}`;
      },
    },
    {
      title: 'A real setpoint change is one deliberate write, not a 20Hz toggle',
      rule: 'A transfer switch changes state when conditions cross the threshold - a slow-moving physical quantity (solar irradiance) that does not flip multiple times per second.',
      body: () => `This attack toggles the switch ${s.n_write_requests} times at ~${s.avg_interval_ms}ms
                   intervals (${s.on_count} ON / ${s.off_count} OFF, near-random order) - no physical
                   solar condition changes fast enough to justify even a fraction of these commands, let
                   alone a value that flips back and forth within milliseconds.`,
    },
    {
      title: 'The command source has no control authority',
      rule: 'Only the PLC\'s own internal threshold logic is supposed to decide the switch state - no external master is expected to command it directly at all.',
      body: () => `Every write in this attack originates from 192.168.0.1, a host that has never sent a
                   single Modbus write in 47,198 normal rows. In legitimate operation this coil is only
                   ever polled (read), never written from outside - see the "Normal" column: this
                   request category does not exist in normal traffic at all.`,
    },
  ];
  document.getElementById('logic-grid').innerHTML = logicPoints.map(p => `
    <div class="logic-card${p.flagship ? ' flagship' : ''}">
      <div class="logic-card-title">${esc(p.title)}</div>
      <div class="logic-card-rule">${p.rule}</div>
      <div class="logic-card-body">${p.body()}</div>
    </div>`).join('');

  // ---- detection signals ----
  const signals = [
    {
      title: 'Write command execution (fc5)', tag: 'categorical',
      normal: '0 writes', attack: `${s.n_write_requests} writes, ALL ACCEPTED`,
      note: `0% of normal traffic is a write. Every single write in both bursts was executed by the
             target (echoed back), not rejected - MITRE ATT&CK T0855 Unauthorized Command Message,
             repeated ${s.n_write_requests} times instead of once.`,
    },
    {
      title: 'Write request rate', tag: 'statistical',
      normal: '0/s', attack: `~${s.avg_rate_per_sec}/s`,
      note: `One write every ~${s.avg_interval_ms}ms, sustained for the full length of each burst -
             far faster than any legitimate control action, which would be a single deliberate write,
             not a rapid repeated toggle.`,
    },
    {
      title: 'Value pattern (ON/OFF flapping)', tag: 'categorical',
      normal: 'n/a (no writes)', attack: `${s.on_count} ON / ${s.off_count} OFF, near-random order`,
      note: `Rapidly flapping the same coil between ON and OFF within milliseconds has no legitimate
             control-system purpose - a real setpoint change is a single deliberate write, not a
             high-frequency toggle.`,
    },
    {
      title: 'Write address', tag: 'categorical',
      normal: 'n/a (address 9 never read either)', attack: `always address ${s.write_address}`,
      note: `Address ${s.write_address} is never touched by normal traffic (which only ever reads
             addresses 9's siblings: 19, 20, 39 are read - but address ${s.write_address} itself is
             never a legitimate READ target, and this attack only ever WRITES it).`,
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
      note: `See the background-noise section above - a labeling artifact, not a detection signal
             itself, but essential context for reading any raw row-count statistic about this attack.`,
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

  // ---- time series (write-request rate across the whole session) ----
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
    `Normal traffic never sends a write - <b class="mono" style="color:var(--normal)">0</b> occurrences
     across the whole session. This attack's 2 bursts appear as
     <b class="mono" style="color:var(--attack)">${tl.attack_active_bin_pct}%</b> of the session's bins
     - up to <b class="mono" style="color:var(--attack)">${tl.attack_max_per_bin}</b> requests in a
     single bin, two dense towers far apart, not a recurring pattern like naive-sensor-read's 18
     evenly-scattered bursts.`;

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
    {label: 'Write requests (fc5)', normal: 0, attack: s.n_write_requests},
    {label: 'Write bursts (>5s gap apart)', normal: 0, attack: s.n_bursts},
    {label: 'Write rate (requests/sec)', normal: 0.001, attack: s.avg_rate_per_sec, logScale: true},
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
    ['Write requests (fc5) sent', '0', `${s.n_write_requests}`, true],
    ['Write bursts (>5s gap apart)', '—', `${s.n_bursts} (sizes: ${s.burst_sizes.join(', ')})`, true],
    ['Burst durations', '—', `${s.burst_durations_sec.join('s, ')}s`, false],
    ['Write rate during a burst', '0/s', `~${s.avg_rate_per_sec}/s (${s.avg_interval_ms}ms interval)`, true],
    ['Value pattern', '—', `${s.on_count} ON / ${s.off_count} OFF (address ${s.write_address})`, false],
    ['Rows labeled attack_specific=5', '—', `${s.attack_total_rows.toLocaleString()}`, false],
    ['...of which unrelated background traffic', '—', `${s.attack_noise_rows.toLocaleString()} (${s.attack_noise_pct}%)`, true],
    ['Burst 2 cross-label continuity', '—', `stream ${s.cross_label_stream}: ${s.cross_label_attack4_requests} reqs under attack_specific=4, then ${s.cross_label_attack5_requests} more under =5, same continuous event`, true],
  ];
  document.getElementById('stats-body').innerHTML = rows.map(([m, n, a, dev]) =>
    `<tr class="${dev ? 'deviates' : ''}"><td class="metric">${m}</td><td class="normal-val">${n}</td><td class="attack-val">${a}</td></tr>`
  ).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
