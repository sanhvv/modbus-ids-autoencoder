"""
Builds data_visualisation/index.html - a single hub page linking to every
report under data_visualisation/, so opening one URL (and refreshing it) is
enough to see whichever attack-type analysis was completed most recently.
Named index.html on purpose: Python's http.server serves it automatically
for a bare directory request, so http://<host>:<port>/data_visualisation/
shows this page with no extra path to remember.

This script has no dataset logic of its own - it only reads the ENTRIES list
below (updated by hand each time a new packet_compare_smartgrid_*.py script
is completed) and the session-wide report already produced by
data_visualisation.py. Re-run it after finishing each new attack-type report:
    python build_data_visualisation_index.py
"""

from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("data_visualisation")
OUTPUT_DIR.mkdir(exist_ok=True)

# One entry per attack type in the Smart Grid per-attack-type series (see the
# Smart Grid normal-behavior-baseline memory). Update this list by hand each
# time a new packet_compare_smartgrid_*.py script is completed, then re-run
# this script. "folder" is relative to data_visualisation/; leave it None
# for an attack not started yet so it still shows on the roadmap, greyed out.
SMARTGRID_ATTACKS = [
    {
        "attack_specific": 1,
        "title": "Address Scan",
        "folder": "smartgrid_address_scan",
        "script": "packet_compare_smartgrid_addressscan.py",
        "summary": "192.168.0.1 is a legitimate host that suddenly ARP/SYN-scans 5 new hosts. "
                   "94.3% of the attack is ARP (98% of ALL session ARP traffic) - only 0.075% is "
                   "Modbus, so the AE/LLM pipeline is structurally blind to most of it.",
        "tag": "pipeline blind spot",
    },
    {
        "attack_specific": 2,
        "title": "Function Code Scan",
        "folder": "smartgrid_function_code_scan",
        "script": "packet_compare_smartgrid_fcscan.py",
        "summary": "One 22ms burst probing 45 function codes (~6,000 pkt/s). A write command "
                   "(fc5) was ACCEPTED by the target despite 0% writes in normal traffic, and a "
                   "Report Slave ID probe (fc17) leaked the string \"Pymodbus\".",
        "tag": "info leak + accepted write",
    },
    {
        "attack_specific": 3,
        "title": "Device Identification Attack",
        "folder": "smartgrid_device_id",
        "script": "packet_compare_smartgrid_deviceid.py",
        "summary": "16 low-and-slow rounds (~1.9ms each, new TCP connection every time), "
                   "averaging one every 5.8 minutes. All 3 standard MEI access codes are probed "
                   "each round, but the device returns 0 identification objects every time.",
        "tag": "well-formed but empty",
    },
    {
        "attack_specific": 4,
        "title": "Naive Sensor Read",
        "folder": "smartgrid_naive_sensor_read",
        "script": "packet_compare_smartgrid_naivesensorread.py",
        "summary": "18 bursts of max-quantity reads (2000/125 vs normal's 1) across the whole session, "
                   "plus a 15-write burst tacked onto the end. 30.8% of labeled rows are unrelated "
                   "background traffic, not attack behavior - much higher than the first 3 attacks.",
        "tag": "brute-force + noisy label",
    },
    {
        "attack_specific": 5,
        "title": "Sporadic Sensor Measurement Injection",
        "folder": None,
        "script": "packet_compare_smartgrid_sporadicinjection.py",
        "summary": None,
        "tag": None,
    },
    {
        "attack_specific": 6,
        "title": "Force Listen Mode",
        "folder": None,
        "script": "packet_compare_smartgrid_forcelisten.py",
        "summary": None,
        "tag": None,
    },
    {
        "attack_specific": 7,
        "title": "Restart Communication",
        "folder": None,
        "script": "packet_compare_smartgrid_restartcomm.py",
        "summary": None,
        "tag": None,
    },
    {
        "attack_specific": 8,
        "title": "Data Flood Attack",
        "folder": None,
        "script": "packet_compare_smartgrid_dataflood.py",
        "summary": None,
        "tag": None,
    },
]

SESSION_WIDE = {
    "title": "Attack Timing Distribution + Detection-Window Experiment",
    "path": "attack_timing_overview/report.html",
    "script": "data_visualisation.py",
    "summary": "All 3 datasets (IED, Smart Grid, WBF): when normal vs. attack traffic falls within "
               "each session, and an AUC-ROC test of the pipeline's hardcoded k=4s rate window "
               "against candidate window sizes 0.5-60s.",
}

CSS = """
  .viz-root {
    color-scheme: light;
    --surface-0: #f9f9f7; --surface-1: #fcfcfb; --border: #e4e3de;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8980;
    --normal: #2a78d6; --attack: #eb6834; --normal-bg: #eaf2fc; --attack-bg: #fdece2;
    --shadow: 0 1px 2px rgba(20,20,15,0.06), 0 6px 20px rgba(20,20,15,0.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .viz-root {
      color-scheme: dark;
      --surface-0: #0d0d0d; --surface-1: #1a1a19; --border: #302f2b;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #7d7c74;
      --normal: #3987e5; --attack: #d95926; --normal-bg: #13253d; --attack-bg: #3a2116;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-0: #0d0d0d; --surface-1: #1a1a19; --border: #302f2b;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #7d7c74;
    --normal: #3987e5; --attack: #d95926; --normal-bg: #13253d; --attack-bg: #3a2116;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 6px 20px rgba(0,0,0,0.35);
  }
  .viz-root {
    min-height: 100vh; background: var(--surface-0); color: var(--text-primary);
    font-family: "IBM Plex Sans", system-ui, sans-serif; padding: 32px 20px 64px;
  }
  .viz-root * { box-sizing: border-box; }
  .wrap { max-width: 980px; margin: 0 auto; display: flex; flex-direction: column; gap: 34px; }
  h1, h2 { font-family: "Archivo", system-ui, sans-serif; text-wrap: balance; margin: 0; }
  h1 { font-size: clamp(26px, 4vw, 36px); font-weight: 800; letter-spacing: -0.01em; }
  h2 { font-size: 16px; font-weight: 700; }
  .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .eyebrow {
    font-family: "IBM Plex Mono", monospace; font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted);
  }
  p { color: var(--text-secondary); line-height: 1.55; margin: 0; max-width: 72ch; }
  section { display: flex; flex-direction: column; gap: 14px; }

  .card {
    display: flex; flex-direction: column; gap: 8px; padding: 16px 18px; border-radius: 10px;
    border: 1px solid var(--border); background: var(--surface-1); box-shadow: var(--shadow);
    text-decoration: none; color: inherit;
  }
  a.card:hover { border-color: var(--normal); }
  .card.pending { opacity: 0.55; box-shadow: none; border-style: dashed; }
  .card-title-row { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
  .card-title { font-family: "Archivo", sans-serif; font-weight: 700; font-size: 15px; color: var(--text-primary); }
  .card-meta { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--text-muted); }
  .card-summary { font-size: 12.5px; color: var(--text-secondary); line-height: 1.5; }
  .tag {
    display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 4px;
    background: var(--attack-bg); color: var(--attack); font-family: "IBM Plex Mono", monospace;
    width: fit-content;
  }
  .tag.session-wide { background: var(--normal-bg); color: var(--normal); }
  .status-pending { font-size: 11px; color: var(--text-muted); font-style: italic; }

  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }

  footer { border-top: 1px solid var(--border); padding-top: 16px; }
  footer p { font-size: 11.5px; }
"""


def render_html(done_count, total_count, generated_at):
    session_card = f"""
    <a class="card" href="{SESSION_WIDE['path']}">
      <div class="card-title-row">
        <span class="card-title">{SESSION_WIDE['title']}</span>
        <span class="tag session-wide">session-wide, all 3 datasets</span>
      </div>
      <div class="card-summary">{SESSION_WIDE['summary']}</div>
      <div class="card-meta">{SESSION_WIDE['script']}</div>
    </a>"""

    attack_cards = []
    for a in SMARTGRID_ATTACKS:
        if a["folder"]:
            attack_cards.append(f"""
    <a class="card" href="{a['folder']}/report.html">
      <div class="card-title-row">
        <span class="card-title">{a['title']}</span>
        <span class="card-meta">attack_specific = {a['attack_specific']}</span>
      </div>
      <div class="card-summary">{a['summary']}</div>
      <div class="card-title-row">
        <span class="tag">{a['tag']}</span>
        <span class="card-meta">{a['script']}</span>
      </div>
    </a>""")
        else:
            attack_cards.append(f"""
    <div class="card pending">
      <div class="card-title-row">
        <span class="card-title">{a['title']}</span>
        <span class="card-meta">attack_specific = {a['attack_specific']}</span>
      </div>
      <div class="status-pending">Not started yet</div>
    </div>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Visualisation Index</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>html, body {{ margin: 0; }}{CSS}</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">

  <header>
    <span class="eyebrow">modbus-ids-autoencoder &middot; data_visualisation/</span>
    <h1 style="margin-top:8px">Data Visualisation Index</h1>
    <p style="margin-top:10px">Every report under <code class="mono">data_visualisation/</code> in one
      place. Refresh this page after a new attack-type analysis is completed to see its link appear
      here. Progress: <b>{done_count} of {total_count}</b> Smart Grid attack types done.
      Regenerated by <code class="mono">build_data_visualisation_index.py</code>, last run
      {generated_at}.</p>
  </header>

  <section>
    <h2>Session-wide analysis</h2>
    {session_card}
  </section>

  <section>
    <h2>Per-attack-type packet comparison (Smart Grid)</h2>
    <p>Normal vs. attack, packet by packet and statistic by statistic, one page per attack type - see
      the Smart Grid normal-behavior-baseline memory for the methodology.</p>
    <div class="grid">{"".join(attack_cards)}
    </div>
  </section>

  <footer>
    <p>IED and Water Bottle Factory per-attack-type comparisons are not started - Smart Grid is being
      done first as the template.</p>
  </footer>

</div>
</div>
</body>
</html>
"""


def main():
    done = [a for a in SMARTGRID_ATTACKS if a["folder"]]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = render_html(len(done), len(SMARTGRID_ATTACKS), generated_at)
    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Progress: {len(done)}/{len(SMARTGRID_ATTACKS)} Smart Grid attack types linked")
    for a in SMARTGRID_ATTACKS:
        status = "DONE" if a["folder"] else "pending"
        print(f"  [{status:7s}] attack_specific={a['attack_specific']}: {a['title']}")


if __name__ == "__main__":
    main()
