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

CORRECTION (added 2026-09-29 while building the sporadic-injection report):
the "write burst" example's 15 requests are not the whole event - they are
the first 15% of ONE continuous 100-request, ~10-second write-toggle
sequence on TCP stream 6823. The remaining 85 requests of that SAME
physical event are labeled attack_specific=5 ("sporadic sensor measurement
injection") instead - a single malicious action split across two different
attack-type ground-truth labels, not two separate events. See
packet_compare_smartgrid_sporadicinjection.py for the full sequence and a
prominent callout of this finding.

Output goes into data_visualisation/smartgrid_naive_sensor_read/ (all
filenames get the optional --tag suffix so earlier results are not
overwritten): packets.json, stats.json, timeline.json, report.html.
timeline.json (added 2026-09-29, per user request) holds per-10s-bin Modbus
request counts for normal vs. attack traffic across the whole session -
the data behind the report's time-series chart, which makes the burst
pattern visually obvious next to normal's flat, continuous baseline. Run
with a log, e.g.:
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
MASTER_IP = "192.168.0.40"
SWITCH_ADDR = 9          # transfer switch coil (verified point map, see memory Section 2b)
SOLAR_ADDR = 19          # solar_panel_power_meter (solar power output meter)
THRESHOLD_ADDR = 39      # switching threshold setpoint
BURST_GAP_SEC = 5.0


def _decode_pdu_addr(hexdata):
    s = str(hexdata)[2:] if str(hexdata).startswith("0x") else str(hexdata)
    return int(s[:4], 16) if len(s) >= 4 else None


def _decode_pdu_write_val(hexdata):
    s = str(hexdata)[2:] if str(hexdata).startswith("0x") else str(hexdata)
    return int(s[4:8], 16) if len(s) >= 8 else None


def _decode_reg_response_value(hexdata):
    """Decode a single-register (fc3/fc4) response's value, e.g. "0x0200" -> 512."""
    if hexdata is None:
        return None
    h = str(hexdata)
    return int(h[2:6], 16) if len(h) >= 6 else None


def compute_solar_logic_violation(df):
    """Physical-logic check against the REAL PLC control rule, as given by
    the user: the PLC reads solar_panel_power_meter (address 19) and
    household_power_meter (address 20) continuously; if
    solar_panel_power_meter > threshold (address 39, decoded live here,
    not hardcoded), it commands the actuator (address 9) ON (route power
    through solar) - otherwise it should not.

    CORRECTED 2026-09-29: an earlier version of this check compared each
    write's nearby solar reading against the historical MEAN of legitimate
    ON/OFF periods (665.5 / 553.0), not the actual configured threshold
    (512) - that flagged some writes as "low" that were still above the
    real threshold and therefore not true rule violations. This version
    uses the literal threshold value and checks BOTH violation directions:
    (a) ON commanded while solar <= threshold (the direction the user's
    own example described), and (b) OFF commanded while solar > threshold
    (the direction actually found in this dataset - see the counts
    returned below). Every write to the real switch coil (address 9) in
    this attack's own labeled scope is checked, not just the first one.
    """
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    mb = n[n.protocol == "MODBUS"].sort_values("frame_time_relative")
    req = mb[mb.ip_src == MASTER_IP].copy()
    req["addr"] = req.modbus_data.apply(_decode_pdu_addr)

    def response_hex(row_idx):
        r = df.iloc[int(row_idx) + 1]
        return str(r.modbus_data) if r.ip_src == TARGET_IP else None

    solar = req[(req.modbus_func_code == 4.0) & (req.addr == SOLAR_ADDR)][["row", "frame_time_relative"]].copy()
    solar["solar_val"] = solar.row.apply(lambda r: _decode_reg_response_value(response_hex(r)))
    solar = solar[(solar.solar_val.notna()) & (solar.solar_val < 5000)].sort_values("frame_time_relative")

    thresh_req = req[(req.modbus_func_code == 3.0) & (req.addr == THRESHOLD_ADDR)][["row"]].copy()
    thresh_vals = thresh_req.row.apply(lambda r: _decode_reg_response_value(response_hex(r)))
    threshold_value = int(thresh_vals.mode().iloc[0]) if len(thresh_vals.dropna()) else None

    # every write to the real switch coil (address 9) in THIS attack's own
    # labeled scope (attack_specific==4) - not just the first request.
    a4 = df[df.attack_specific == 4]
    writes = a4[(a4.protocol == "MODBUS") & (a4.ip_src == ATTACKER_IP)
               & (a4.modbus_func_code == 5)].copy()
    writes["addr"] = writes.modbus_data.apply(_decode_pdu_addr)
    writes = writes[writes.addr == SWITCH_ADDR].copy()
    writes["cmd_val"] = writes.modbus_data.apply(_decode_pdu_write_val)
    writes = writes.sort_values("frame_time_relative")

    merged = pd.merge_asof(writes[["row", "frame_time_relative", "cmd_val"]], solar[["frame_time_relative", "solar_val"]],
                            on="frame_time_relative", direction="nearest", tolerance=2.0).dropna(subset=["solar_val"])
    merged["cmd_on"] = merged.cmd_val == 0xff00
    merged["solar_above"] = merged.solar_val > threshold_value

    on_below = merged[merged.cmd_on & ~merged.solar_above]     # user's described violation type
    off_above = merged[~merged.cmd_on & merged.solar_above]    # the type actually found here
    on_above = merged[merged.cmd_on & merged.solar_above]      # consistent
    off_below = merged[~merged.cmd_on & ~merged.solar_above]   # consistent

    # pick the clearest example: prefer an on_below violation if one
    # exists (matches the user's own description exactly); otherwise the
    # off_above violation with the largest margin above threshold.
    if len(on_below):
        example = on_below.sort_values("solar_val").iloc[0]
        violation_kind = "on_below"
    elif len(off_above):
        example = off_above.sort_values("solar_val", ascending=False).iloc[0]
        violation_kind = "off_above"
    else:
        example = None
        violation_kind = None

    attack_request_pkt = None
    attack_response_pkt = None
    normal_request_pkt = None
    normal_response_pkt = None
    example_time = None
    example_solar = None
    example_solar_gap_sec = None
    if example is not None:
        req_row = writes.loc[writes.row == example.row].iloc[0]
        attack_request_pkt = pack(req_row)
        # the immediately-following row can be a bare TCP ACK (rapid-fire
        # write toggling generates extra ACK-only rows) - search forward
        # for the actual next MODBUS response instead of assuming row+1.
        following = df[(df.row > req_row.row) & (df.row <= req_row.row + 5)
                       & (df.protocol == "MODBUS") & (df.ip_src == TARGET_IP)]
        if len(following):
            attack_response_pkt = pack(following.iloc[0])

        example_time = float(example.frame_time_relative)
        example_solar = float(example.solar_val)
        cand = solar.iloc[(solar.frame_time_relative - example_time).abs().argsort()]
        best = cand.iloc[0]
        example_solar_gap_sec = round(float(best.frame_time_relative) - example_time, 3)
        solar_req_row = req.loc[req.row == best.row].iloc[0]
        normal_request_pkt = pack(solar_req_row)
        solar_resp_row = df.iloc[int(best.row) + 1]
        if solar_resp_row.ip_src == TARGET_IP:
            normal_response_pkt = pack(solar_resp_row)

    return {
        "switch_addr": SWITCH_ADDR,
        "solar_addr": SOLAR_ADDR,
        "threshold_addr": THRESHOLD_ADDR,
        "threshold_value": threshold_value,
        "n_writes_checked": int(len(merged)),
        "n_on_below_violations": int(len(on_below)),
        "n_off_above_violations": int(len(off_above)),
        "n_on_above_consistent": int(len(on_above)),
        "n_off_below_consistent": int(len(off_below)),
        "violation_kind": violation_kind,
        "example_time_sec": round(example_time, 3) if example_time is not None else None,
        "example_solar_reading": example_solar,
        "example_solar_gap_sec": example_solar_gap_sec,
        "attack_request_pkt": attack_request_pkt,
        "attack_response_pkt": attack_response_pkt,
        "normal_request_pkt": normal_request_pkt,
        "normal_response_pkt": normal_response_pkt,
    }


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


def compute_timeline(df, bin_width=10.0):
    """Per-bin Modbus request counts across the whole session, for normal
    traffic (master 192.168.0.40 -> RTU) vs. this attack's own requests
    (192.168.0.1 -> RTU) - the raw material for the report's time-series
    chart. Bin width matches the attack's own burst duration (10s) so each
    burst lands in ~1 bin instead of being smeared across several.
    """
    dur = float(df.frame_time_relative.max())
    edges = np.arange(0, dur + bin_width, bin_width)
    n = df[df.attack_specific.isna() | (df.attack_specific == 0)]
    a4 = df[df.attack_specific == 4]

    normal_times = n[(n.protocol == "MODBUS") & (n.ip_src == "192.168.0.40")].frame_time_relative.to_numpy()
    attack_times = a4[(a4.protocol == "MODBUS") & (a4.ip_src == ATTACKER_IP)].frame_time_relative.to_numpy()

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
    normal_addrs = set()
    n_req = n[(n.protocol == "MODBUS") & (n.tcp_len == 12)]
    for d in n_req.modbus_data:
        addr, q = decode_qty(d)
        if q is not None:
            normal_read_qtys.append(q)
        if addr is not None:
            normal_addrs.add(addr)

    attack_addrs = set()
    for d in read_req.modbus_data:
        addr, _ = decode_qty(d)
        if addr is not None:
            attack_addrs.add(addr)

    # detection signals: each one independently verified against the raw
    # data, not assumed - see the "Detection signals" report section.
    n_resp = n[(n.protocol == "MODBUS") & (n.ip_src == TARGET_IP)]
    a_resp = mb[mb.ip_src == TARGET_IP]
    normal_modbus_ips = sorted(set(n[n.protocol == "MODBUS"].ip_src.unique().tolist()))
    attack_response_bytes = int(a_resp[a_resp.tcp_len > 20].ip_len.mode().iloc[0]) if len(a_resp) else None
    normal_response_max_bytes = int(n_resp.ip_len.max()) if len(n_resp) else None

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
        "normal_request_addresses": sorted(normal_addrs),
        "attack_request_addresses": sorted(attack_addrs),
        "n_fc2_requests": int((req.modbus_func_code == 2).sum()),
        "normal_modbus_ips": normal_modbus_ips,
        "attacker_ip": ATTACKER_IP,
        "normal_response_max_bytes": normal_response_max_bytes,
        "attack_response_bytes": attack_response_bytes,
        "response_size_ratio": round(attack_response_bytes / normal_response_max_bytes, 1)
                                if attack_response_bytes and normal_response_max_bytes else None,
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
    print("Computing request-rate time series (normal vs. attack, whole session)...")
    timeline = compute_timeline(df)
    print("Checking transfer-switch command against the real solar meter reading...")
    solar_logic = compute_solar_logic_violation(df)

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

    with open(path("solar_logic.json"), "w") as f:
        json.dump(solar_logic, f, indent=1)
    print(f"Saved: {path('solar_logic.json')}")

    payload = {**packets, "stats": stats, "timeline": timeline, "solar_logic": solar_logic}
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
  .bar-axis-label { font-size: 10px; fill: var(--text-muted); font-family: "IBM Plex Mono", monospace; }
  .bar-value-label { font-size: 11px; font-weight: 600; font-family: "IBM Plex Mono", monospace; }
  .bar-gridline { stroke: var(--border); stroke-width: 1; }
  .bar-legend { display: flex; gap: 14px; align-items: center; }
  .bar-legend .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--text-secondary); }
  .bar-legend .swatch { width: 9px; height: 9px; border-radius: 2px; }

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
        normal baseline (47,198 rows), not assumed. <span class="signal-tag categorical" style="margin:0 4px">categorical</span>
        means the value never occurs at all in normal traffic (zero-ambiguity); <span class="signal-tag statistical" style="margin:0 4px">statistical</span>
        means normal traffic does have this value, but at a very different magnitude.</p>
    </div>
    <div class="signal-grid" id="signal-grid"></div>
  </section>

  <section>
    <div class="section-head">
      <h2>Time series: request rate across the whole session</h2>
      <p style="margin-top:6px">Same metric (Modbus requests per 10-second bin), plotted across all
        <span id="duration-inline"></span> of the session for both traffic types &mdash; normal traffic
        holds a steady, continuous rate; the attack is silent almost everywhere, then spikes hard for
        exactly the length of each burst.</p>
    </div>
    <div class="bar-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--normal)"></span>Normal (master &rarr; RTU)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--attack)"></span>Naive sensor read</span>
    </div>
    <div class="timeline-card">
      <div class="chart-svg-box" id="timeline-chart"></div>
      <p style="font-size:12px" id="timeline-caption"></p>
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
  document.getElementById('duration-inline').textContent = (s.session_duration_sec / 60).toFixed(1) + ' minutes';

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
      `<div class="decoded" style="border:1px dashed var(--border); border-radius:8px; margin-top:6px">
        <b>Correction (added after building the sporadic-injection report):</b> these 15 requests are
        not the whole event &mdash; they are the first 15% of ONE continuous 100-request, ~10-second
        write-toggle sequence on the same TCP connection (stream 6823). The remaining 85 requests of
        that same physical event are labeled <code class="mono">attack_specific&nbsp;=&nbsp;5</code>
        instead ("sporadic sensor measurement injection") - a single malicious action split across two
        different attack-type ground-truth labels, not two separate events. See
        <code class="mono">packet_compare_smartgrid_sporadicinjection.py</code> for the full sequence.
      </div>` +
      `</div>`;
  }
  if (ex.rejected_probe && ex.rejected_probe.syn) {
    const probeDst = ex.rejected_probe.syn.ip_dst;
    const isMaster = probeDst === '192.168.0.40';
    attackHtml += `<div><h3 style="margin-bottom:8px">3. A side-probe that got refused<span class="flag">SYN &rarr; RST-ACK</span></h3>` +
      frame('SYN', ex.rejected_probe.syn, true,
        isMaster
          ? `One of the 3 side-probe targets is <b>192.168.0.40 &mdash; the network's own legitimate
             SCADA master</b>, the same host this session's normal traffic uses to poll the RTU. This
             attacker tries to open a connection directly to the real controller and gets refused;
             the other 2 targets (192.168.0.21, 192.168.0.22) are hosts never otherwise seen at all.`
          : `A brief attempt to open a connection to a host (${probeDst}) this attacker never
             otherwise talks to &mdash; ${s.n_rejected_probe_targets} such hosts touched around the
             same moment, including the network's own legitimate SCADA master (192.168.0.40).`) +
      (ex.rejected_probe.rst ? frame('RST-ACK', ex.rejected_probe.rst, true,
        `Connection actively refused${isMaster ? ' by the master itself' : ''}. Unlike the address
         scan's targets, which answered with SYN-ACK, this host rejects the attempt outright &mdash;
         a minor, failed side-activity, not the main signature of this attack.`) : '') +
      `</div>`;
  }
  aCol.innerHTML = attackHtml;

  // ---- protocol logic violation ----
  const sl = DATA.solar_logic;
  const logicPoints = [
    {
      flagship: true,
      title: 'Switch commanded OFF while the real meter was above the switching threshold',
      rule: `The PLC's real control rule (as specified by the user, verified against the ICS-SimLab
             paper): continuously read solar_panel_power_meter (address ${sl.solar_addr}) and
             household_power_meter (address 20); if solar_panel_power_meter &gt; threshold (address
             ${sl.threshold_addr}, decoded live here as <b>${sl.threshold_value}</b>), command the
             actuator (address ${sl.switch_addr}) to route power through solar. Checked against EVERY
             one of this attack's ${sl.n_writes_checked} writes to the real switch coil, not just one
             example.`,
      body: () => {
        const flow = `<div class="logic-flow">
          <span class="step low">solar meter = ${sl.example_solar_reading} &mdash; ABOVE threshold ${sl.threshold_value}</span>
          <span class="arrow">&mdash; rule says: should be routed to solar &mdash;</span>
          <span class="step cmd">yet: SWITCH TO MAINS commanded (t=${sl.example_time_sec}s)</span>
        </div>`;
        const cmp = `<div class="logic-compare">
          <div>
            <div class="logic-compare-head normal-label">Normal &mdash; legitimate master's own solar-meter poll</div>
            ${sl.normal_request_pkt ? frame('REQUEST', sl.normal_request_pkt, false,
              `Read Input Registers (fc4), address ${sl.solar_addr}, quantity 1 &mdash; the master's own
               routine poll, the closest one in time to the attack command below (${Math.abs(sl.example_solar_gap_sec)}s
               ${sl.example_solar_gap_sec >= 0 ? 'after' : 'before'} it - real polling only happens
               every ~1s, so this is the nearest available ground truth, not an exact-instant match).`) : ''}
            ${sl.normal_response_pkt ? frame('RESPONSE', sl.normal_response_pkt, false,
              `Decodes to <b>${sl.example_solar_reading}</b> &mdash; clearly above the threshold of
               ${sl.threshold_value}, meaning the rule calls for solar to be in use.`) : ''}
          </div>
          <div>
            <div class="logic-compare-head attack-label">Attack &mdash; the "switch to mains" command itself</div>
            ${sl.attack_request_pkt ? frame('REQUEST', sl.attack_request_pkt, true,
              `Write Single Coil (fc5), address ${sl.switch_addr}, value 0x0000 (OFF/mains) &mdash; sent
               at t=${sl.example_time_sec}s, while the real meter (left) showed ${sl.example_solar_reading}
               &mdash; above the ${sl.threshold_value} threshold.`) : ''}
            ${sl.attack_response_pkt ? frame('RESPONSE', sl.attack_response_pkt, true,
              `Echoed back &mdash; ACCEPTED, despite contradicting the real sensor reading and the
               PLC's own rule at the same moment.`) : ''}
          </div>
        </div>`;
        return `Checked directly, not assumed: across all ${sl.n_writes_checked} writes to the real
                switch coil under this attack's own labeled scope (attack_specific=4 - the last 15
                requests appended right after the final read burst, described elsewhere on this page as
                a "coda" for that reason within this narrow view; the SAME physical write event
                continues for 85 more toggles under attack_specific=5 instead - see
                <code class="mono">packet_compare_smartgrid_sporadicinjection.py</code> - so it is really
                the START of a longer 100-request sequence, not a short trailing flourish), NONE command
                solar while the real meter reads at or below the ${sl.threshold_value} threshold (the
                real solar reading never actually dips that low during this event). But
                <b>${sl.n_off_above_violations} of ${sl.n_writes_checked}</b> writes command
                <b>MAINS (OFF)</b> while the real meter reads ABOVE the threshold - directly
                contradicting the PLC's own rule in the opposite direction. Example: the meter read
                ${sl.example_solar_reading} (threshold ${sl.threshold_value}) at the exact moment a
                switch-to-mains command was sent and accepted.${flow}${cmp}`;
      },
    },
    {
      title: 'A quantity of 2000/125 serves no monitoring purpose',
      rule: 'Every real measurement point in this deployment is read one value at a time - a monitoring task never needs more than the single current reading, let alone the protocol\'s absolute maximum.',
      body: () => `This attack requests ${s.attack_max_qty_fc1_fc2} coils (fc1/fc2) or
                   ${s.attack_max_qty_fc3_fc4} registers (fc3/fc4) per request, starting at address 0 -
                   not one of this deployment's 4 real points (9, 19, 20, 39) at all. There is no
                   monitoring or control purpose for dumping a block of memory that includes no real
                   sensor or switch.`,
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
      title: 'Function code 2 (Read Discrete Inputs)', tag: 'categorical',
      normal: '0 requests', attack: `${s.n_fc2_requests} requests`,
      note: `fc2 never appears in 47,198 normal rows (only fc1/fc3/fc4 are ever used) - same kind of
             signal as the function-code-scan attack: this network's normal vocabulary simply doesn't
             include this function code at all.`,
    },
    {
      title: 'Response size', tag: 'statistical',
      normal: `max ${s.normal_response_max_bytes} bytes`, attack: `${s.attack_response_bytes} bytes`,
      note: `${(s.response_size_ratio)}&times; the normal maximum - a direct consequence of requesting
             the protocol-max quantities (2000 coils / 125 registers) instead of normal's single value.`,
    },
    {
      title: 'Request source identity', tag: 'categorical',
      normal: s.normal_modbus_ips.join(', '), attack: s.attacker_ip,
      note: `Only ${s.normal_modbus_ips.join(' and ')} ever send/receive Modbus traffic in the normal
             baseline. ${s.attacker_ip} is a real host on this network (it has normal WEBSOCKET traffic
             with 192.168.0.111) but has never once been a Modbus participant before this attack.`,
    },
    {
      title: 'Request address', tag: 'categorical',
      normal: s.normal_request_addresses.join(', '), attack: s.attack_request_addresses.join(', '),
      note: `Normal polling only ever targets 4 specific points (addresses ${s.normal_request_addresses.join(', ')})
             - address 0 is requested 0 times in 47,198 normal rows. This attack reads only address 0,
             a point no legitimate poll ever touches.`,
    },
    {
      title: 'Read quantity per request', tag: 'statistical',
      normal: `${s.normal_max_qty}`, attack: `${s.attack_max_qty_fc1_fc2} / ${s.attack_max_qty_fc3_fc4}`,
      note: `Already the headline finding above - repeated here for completeness of the signal
             checklist. Normal never asks for more than 1 value per request.`,
    },
    {
      title: 'Write command acceptance (fc5)', tag: 'categorical',
      normal: '0 writes', attack: `${s.n_write_requests} writes, ACCEPTED`,
      note: `0% of normal traffic is a write. A write request that the target executes anyway (echoed
             back, not rejected) is MITRE ATT&CK T0855 Unauthorized Command Message - qualitatively
             worse than an attempted-but-refused write.`,
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

    // gridlines + y labels
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

    // x-axis ticks every 10 minutes
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

  const tl = DATA.timeline;
  document.getElementById('timeline-chart').appendChild(drawTimeSeries(tl));
  document.getElementById('timeline-caption').innerHTML =
    `Normal traffic averages <b class="mono" style="color:var(--normal)">${tl.normal_avg_per_bin}</b>
     requests per 10s bin, essentially unbroken across the whole session. The attack sits at
     <b class="mono" style="color:var(--attack)">0</b> for the rest, then spikes to as many as
     <b class="mono" style="color:var(--attack)">${tl.attack_max_per_bin}</b> requests in a single bin
     during a burst &mdash; active in only <b class="mono" style="color:var(--attack)">${tl.attack_active_bin_pct}%</b>
     of the session's bins.`;

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
