"""
Fill quality vs Binance -- one chart per pair, posted to Slack every day.

1. Pulls Dune query 8869650 (pair, venue, bps_threshold, n_fills, pct_fills)
2. Draws one PNG per pair into OUT_DIR
3. Uploads each PNG to freeimage.host (Slack webhooks can't take files, only image links)
4. Posts all charts to Slack in one message via the incoming webhook

Env vars (set as GitHub secrets):
  DUNE_API_KEY                 Dune API key
  SLACK_WEBHOOK_FILL_QUALITY   Slack incoming webhook URL (if empty, charts are only saved)

pip install requests pandas numpy matplotlib
"""

import os
import time

import matplotlib
matplotlib.use("Agg")          # no screen on GitHub runners
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter

# ============================== SETTINGS ==============================
DUNE_API_KEY = os.environ.get("DUNE_API_KEY", "").strip()
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_FILL_QUALITY", "").strip()
QUERY_ID = 8869650

RUN_FRESH = True           # True  = re-run the query first so the charts are always today's data (uses credits)
                           # False = use the last saved result on Dune (only if the query is scheduled on Dune)
WINDOW_LABEL = "last 1 day"
OUT_DIR = "fill_quality_pngs"

# freeimage.host public API key (same image host as the quote exec scatter report)
FREEIMAGE_KEY = "6d207e02198a847aa98d0a2a901485a5"

# "nines"  = stretches the top of the y-axis (90% -> 99% -> 99.9% get equal space)
#            so lines that sit close to 100% split apart clearly
# "linear" = normal y-axis, like Dune
Y_AXIS = "nines"

# the two dotted lines per pair (bps). must be values that exist in the query's bps array
MARKERS = {
    "SOL/USDC":   (3, 5),
    "SOL/USDT":   (3, 5),
    "SOL/USD1":   (3, 5),
    "BONK/USDC":  (20, 30),
    "HYPE/USDC":  (5, 10),
    "PENGU/USDC": (10, 15),
    "PUMP/USDC":  (10, 15),
    "TRUMP/USDC": (10, 15),
    "ZEC/USDC":   (10, 15),
    "USD1/USDC":  (0.9, 1),
    "USDC/USDT":  (0.2, 0.3),
}

# x-axis shows 0 -> (X_ZOOM x the bigger marker), capped at 50 bps
X_ZOOM = 2.0
# ======================================================================

API = "https://api.dune.com/api/v1"
HEAD = {"X-Dune-API-Key": DUNE_API_KEY}

# strong, easy-to-tell-apart colors + a different dot shape per venue
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
           "#d63fa6", "#0097a7", "#8a8a00", "#000000", "#5f5f5f", "#003f7f",
           "#ff1f8f", "#00a36c", "#b8860b", "#6a0dad", "#008b8b", "#8b0000"]
SHAPES = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "d"]


# ------------------------------ data ----------------------------------
def fetch_rows(url):
    rows, offset, last = [], 0, {}
    while offset is not None:
        r = requests.get(url, headers=HEAD, params={"limit": 10000, "offset": offset}, timeout=120)
        r.raise_for_status()
        last = r.json()
        rows += last.get("result", {}).get("rows", [])
        offset = last.get("next_offset")
    return rows, last


def load_data():
    if RUN_FRESH:
        r = requests.post(f"{API}/query/{QUERY_ID}/execute", headers=HEAD, timeout=60)
        r.raise_for_status()
        eid = r.json()["execution_id"]
        print(f"Running query {QUERY_ID} on Dune (execution {eid}) ...")
        while True:
            s = requests.get(f"{API}/execution/{eid}/status", headers=HEAD, timeout=60)
            s.raise_for_status()
            s = s.json()
            if s["state"] == "QUERY_STATE_COMPLETED":
                break
            if s.get("is_execution_finished"):
                raise RuntimeError(f"Dune run did not complete: {s.get('state')} {s.get('error')}")
            time.sleep(5)
        url = f"{API}/execution/{eid}/results"
    else:
        url = f"{API}/query/{QUERY_ID}/results"

    rows, meta = fetch_rows(url)
    if not rows:
        raise RuntimeError("Dune returned no rows. Check the query id / API key.")

    df = pd.DataFrame(rows)
    df["bps_threshold"] = df["bps_threshold"].astype(float)
    df["pct_fills"] = df["pct_fills"].astype(float)
    df["n_fills"] = df["n_fills"].astype(int)

    ran_at = str(meta.get("execution_ended_at", ""))[:16].replace("T", " ")
    print(f"Loaded {len(df):,} rows. Data from Dune run at {ran_at} UTC")

    # guard: if pct_fills only has 2 decimals, lines near 100% cannot be told apart
    if np.allclose(df["pct_fills"] * 100, (df["pct_fills"] * 100).round(), atol=1e-9):
        print("WARNING: pct_fills looks rounded to 2 decimals (0.99, 1.00 ...). "
              "In the SQL use CAST(count_if(...) AS double) / count(*) instead of 1.00 * count_if(...) / count(*).")
    return df, ran_at


# ---------------------------- helpers ---------------------------------
def logit(p):
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def pct_text(p):
    if np.isnan(p):
        return "-"
    if p < 1 and round(p * 100, 2) >= 100:
        return ">99.99%"
    return f"{p * 100:.2f}%"


def tick_text(p):
    return f"{p * 100:.3f}".rstrip("0").rstrip(".") + "%"


def value_at(x, y, bps):
    hit = np.isclose(x, bps)
    return float(y[hit][0]) if hit.any() else np.nan


# ----------------------------- chart ----------------------------------
def plot_pair(pair, d, markers, style, ran_at):
    m1, m2 = markers
    x_max = min(50.0, X_ZOOM * max(markers))

    # one entry per venue: x, y, fills, value at each marker
    lines = []
    for v, g in d.groupby("venue"):
        g = g.sort_values("bps_threshold")
        x, y = g["bps_threshold"].to_numpy(), g["pct_fills"].to_numpy()
        lines.append(dict(venue=v, x=x, y=y, n=int(g["n_fills"].iloc[0]),
                          v1=value_at(x, y, m1), v2=value_at(x, y, m2)))
    # best first (by first marker, then second)
    lines.sort(key=lambda L: (-np.nan_to_num(L["v1"], nan=-1), -np.nan_to_num(L["v2"], nan=-1)))

    view = np.concatenate([L["y"][L["x"] <= x_max] for L in lines])
    lo = max(view.min(), 0.005)
    has_100 = bool((view >= 1).any())
    below_100 = view[view < 1]
    hi = below_100.max() if below_100.size else 0.999

    fig = plt.figure(figsize=(19, 9.5), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.25], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    tb = fig.add_subplot(gs[1])
    tb.axis("off")

    # ---- y-axis ----
    if Y_AXIS == "nines":
        top = expit(logit(max(hi, lo)) + 0.7)          # "100%" sits one small step above the best real value
        y_lo, y_hi = expit(logit(lo) - 0.35), expit(logit(top) + 0.25)
        plot_y = lambda y: np.clip(np.where(y >= 1, top, y), 1e-6, None)

        ax.set_yscale("logit")
        ax.set_ylim(y_lo, y_hi)
        cand = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99,
                0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99999]
        ceiling = expit(logit(top) - 0.3) if has_100 else y_hi
        cand = [t for t in cand if y_lo <= t <= ceiling]
        gap = (logit(y_hi) - logit(y_lo)) / 20             # keep tick labels from touching
        ticks = []
        for t in reversed(cand):
            if not ticks or logit(ticks[-1]) - logit(t) >= gap:
                ticks.append(t)
        ticks = sorted(ticks) + ([top] if has_100 else [])
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda p, _: "100%" if np.isclose(p, top) else tick_text(p)))
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        plot_y = lambda y: y
        ax.set_ylim(max(0, lo - 0.03), 1.005)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # ---- venue lines ----
    for L in reversed(lines):                           # best venue drawn last = on top
        c, mk = style[L["venue"]]
        keep = L["x"] <= x_max * 1.001
        ax.plot(L["x"][keep], plot_y(L["y"][keep]), color=c, lw=2.4, marker=mk, ms=5,
                alpha=0.95, zorder=3)

    # ---- dotted marker lines + dots where each venue crosses them ----
    for m in (m1, m2):
        ax.axvline(m, color="#222222", ls=(0, (1.5, 2.5)), lw=1.8, zorder=2)
        left = m == m1                                   # first label sits left of its line, second sits right
        ax.annotate(f"{m:g} bps", xy=(m, 1.0), xycoords=ax.get_xaxis_transform(),
                    xytext=(-5 if left else 5, 6), textcoords="offset points",
                    ha="right" if left else "left", va="bottom",
                    fontsize=14, fontweight="bold", color="#222222")
        for L in lines:
            val = L["v1"] if m == m1 else L["v2"]
            if not np.isnan(val):
                c, mk = style[L["venue"]]
                ax.scatter([m], plot_y(np.array([val])), s=110, color=c, marker=mk,
                           edgecolor="white", linewidth=1.5, zorder=5)

    ax.set_xlim(0, x_max * 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Spread vs Binance (bps)", fontsize=14)
    ax.set_ylabel("% of fills within ±X bps", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="y", color="#dddddd", lw=0.9)
    ax.grid(True, axis="x", color="#f0f0f0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    note = "  ·  y-axis stretched near 100% so small gaps are visible" if Y_AXIS == "nines" else ""
    ax.text(0, 1.14, f"{pair}  —  % of fills within ±X bps of Binance",
            transform=ax.transAxes, fontsize=19, fontweight="bold")
    ax.text(0, 1.095, f"{WINDOW_LABEL}  ·  higher and further left = better{note}",
            transform=ax.transAxes, fontsize=12.5, color="#555555")

    # ---- right panel: venue (fills) + value at each dotted line ----
    n = len(lines)
    step = min(0.075, 0.86 / (n + 1))
    y0 = 0.93
    cols = (0.79, 0.99)
    tb.text(0.02, y0, "Venue (fills)", fontsize=13, fontweight="bold", va="center", transform=tb.transAxes)
    for cx, m in zip(cols, (m1, m2)):
        tb.text(cx, y0, f"{m:g} bps", fontsize=13, fontweight="bold", ha="right", va="center",
                transform=tb.transAxes)
    tb.plot([0.0, 1.0], [y0 - step * 0.55] * 2, color="#333333", lw=1.2, transform=tb.transAxes, clip_on=False)

    best1 = np.nanmax([L["v1"] for L in lines]) if n else np.nan
    best2 = np.nanmax([L["v2"] for L in lines]) if n else np.nan
    for i, L in enumerate(lines):
        yy = y0 - step * (i + 1)
        c, mk = style[L["venue"]]
        name = L["venue"] if len(L["venue"]) <= 20 else L["venue"][:19] + "…"
        tb.scatter([0.035], [yy], s=110, color=c, marker=mk, edgecolor="white", linewidth=1.2,
                   transform=tb.transAxes, clip_on=False)
        tb.text(0.075, yy, f"{name} ({L['n']:,})", fontsize=12.5, va="center", transform=tb.transAxes)
        for cx, val, best in zip(cols, (L["v1"], L["v2"]), (best1, best2)):
            tb.text(cx, yy, pct_text(val), fontsize=12.5, ha="right", va="center",
                    fontweight="bold" if np.isclose(val, best) else "normal",
                    color=c, transform=tb.transAxes)
        tb.plot([0.0, 1.0], [yy - step / 2] * 2, color="#eeeeee", lw=0.8, transform=tb.transAxes, clip_on=False)

    tb.text(0.02, y0 - step * (n + 1.2), "sorted best → worst at the first dotted line\nbold = best in that column",
            fontsize=10.5, color="#777777", va="top", transform=tb.transAxes)

    fig.text(0.01, 0.005, f"Source: Dune query {QUERY_ID}  ·  data run at {ran_at} UTC",
             fontsize=10, color="#888888")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, pair.replace("/", "-") + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")
    return path


# ------------------------------ slack ---------------------------------
def upload_image(path):
    """Upload one PNG to freeimage.host and return its public link. Tries 3 times."""
    for attempt in range(1, 4):
        try:
            with open(path, "rb") as f:
                r = requests.post("https://freeimage.host/api/1/upload",
                                  data={"key": FREEIMAGE_KEY, "action": "upload", "format": "json"},
                                  files={"source": f}, timeout=120)
            r.raise_for_status()
            return r.json()["image"]["url"]
        except Exception as e:
            print(f"  upload try {attempt} failed for {path}: {e}")
            time.sleep(5 * attempt)
    return None


def slack_send(payload, what):
    """Send one message to the webhook. Retries on Slack server errors (5xx) and rate limits (429)."""
    err = ""
    for attempt in range(1, 4):
        try:
            r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=60)
            if r.status_code == 200:
                return True
            err = f"{r.status_code} {r.text[:200]}"
            if r.status_code < 500 and r.status_code != 429:
                break                      # bad message, retrying will not help
        except requests.RequestException as e:
            err = str(e)
        print(f"  slack try {attempt} failed for {what}: {err}")
        time.sleep(15 * attempt)
    print(f"  gave up on {what}: {err}")
    return False


def post_to_slack(charts, ran_at, skipped):
    """charts = list of (pair, markers, image_url).
    One short header message, then one message per chart. Small messages are fast for
    Slack to check, and if one chart fails the others still get posted."""
    title = f"Fill quality vs Binance ({WINDOW_LABEL})"
    header = {"text": title, "blocks": [
        {"type": "header", "text": {"type": "plain_text", "text": title}},
        {"type": "context",
         "elements": [{"type": "mrkdwn",
                       "text": "% of fills within ±X bps of Binance, per venue. Higher and further left = better. "
                               "Venue names show fill count in brackets. "
                               f"Dune query {QUERY_ID}, data run at {ran_at} UTC."}]},
    ]}
    failed = [] if slack_send(header, "header") else ["header"]

    for pair, (m1, m2), url in charts:
        time.sleep(1.2)                    # webhooks allow about 1 message per second
        label = f"{pair}  ·  dotted lines at {m1:g} and {m2:g} bps"
        msg = {"text": f"{title}: {pair}", "blocks": [
            {"type": "image", "image_url": url, "alt_text": f"{pair} fill quality curve",
             "title": {"type": "plain_text", "text": label}}]}
        if not slack_send(msg, pair):
            failed.append(pair)

    if skipped:
        time.sleep(1.2)
        slack_send({"text": "Not shown: " + ", ".join(skipped)}, "not-shown note")

    print(f"posted {len(charts) - len([f for f in failed if f != 'header'])} of {len(charts)} charts to Slack")
    if failed:
        raise RuntimeError(f"Some Slack posts failed: {failed}")


# ------------------------------ main ----------------------------------
def main():
    if not DUNE_API_KEY:
        raise SystemExit("DUNE_API_KEY is empty. Add it as a GitHub secret (or env var).")

    df, ran_at = load_data()

    # same venue = same color + shape in every chart
    venues = sorted(df["venue"].unique())
    style = {v: (PALETTE[i % len(PALETTE)], SHAPES[i % len(SHAPES)]) for i, v in enumerate(venues)}

    saved, skipped = [], []
    for pair, markers in MARKERS.items():
        d = df[df["pair"] == pair]
        if d.empty:
            print(f"skip {pair}: no rows (no venue passed the n_fills filter?)")
            skipped.append(f"{pair} (no data)")
            continue
        saved.append((pair, markers, plot_pair(pair, d, markers, style, ran_at)))

    extra = sorted(set(df["pair"]) - set(MARKERS))
    if extra:
        print(f"not charted (no dotted-line levels set in MARKERS): {extra}")

    if not SLACK_WEBHOOK_URL:
        print(f"SLACK_WEBHOOK_FILL_QUALITY is empty, so charts are only saved in {OUT_DIR}/")
        return

    charts = []
    for pair, markers, path in saved:
        url = upload_image(path)
        if url:
            charts.append((pair, markers, url))
        else:
            skipped.append(f"{pair} (image upload failed)")
    if not charts:
        raise RuntimeError("No chart could be uploaded, so nothing was posted to Slack.")

    post_to_slack(charts, ran_at, skipped)


if __name__ == "__main__":
    main()
