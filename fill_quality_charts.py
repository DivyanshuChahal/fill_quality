"""
Fill quality vs Binance -- one chart per pair, posted to Slack every day.

1. Pulls Dune query 8869650 (pair, venue, bps_threshold, n_fills, pct_fills)
2. Draws one PNG per pair into OUT_DIR
3. Pushes the PNGs to a branch of this GitHub repo (CHART_BRANCH). Each run replaces
   the branch with one fresh commit, so only today's images are kept.
4. Posts a short header + one message per chart to Slack via the incoming webhook,
   using raw.githubusercontent.com links to those images.

Slack must open the image links without logging in, so the repo that holds the
images has to be public. If this repo is private, set CHART_REPO to a public repo.

Env vars (GitHub secrets):
  DUNE_API_KEY                 Dune API key
  SLACK_WEBHOOK_FILL_QUALITY   Slack incoming webhook URL (if empty, charts are only saved)
  GITHUB_TOKEN                 given by GitHub Actions automatically, used to push the images
  CHART_REPO, CHART_REPO_TOKEN optional: push images to a different (public) repo instead

pip install requests pandas numpy matplotlib
"""

import os
import shutil
import subprocess
import tempfile
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

# where the images live: a branch of this repo (replaced every run)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
CHART_REPO = os.environ.get("CHART_REPO", "").strip() or os.environ.get("GITHUB_REPOSITORY", "").strip()
CHART_TOKEN = os.environ.get("CHART_REPO_TOKEN", "").strip() or GITHUB_TOKEN
CHART_BRANCH = "fill-quality-charts"
QUERY_ID = 8869650

RUN_FRESH = True           # True  = re-run the query first so the charts are always today's data (uses credits)
                           # False = use the last saved result on Dune (only if the query is scheduled on Dune)
WINDOW_LABEL = "last 1 day"
OUT_DIR = "fill_quality_pngs"

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

# x-axis shows -(X_ZOOM x bigger marker) -> +(X_ZOOM x bigger marker),
# cut to the bps range the query has (right now -10 to 50)
X_ZOOM = 2.0

# also draw a line at 0 bps (= Binance price) and show "% of fills <= 0 bps" in the table
ZERO_COLUMN = True
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
    if "total_vol_usd" in df.columns and "volume_usd" not in df.columns:
        df = df.rename(columns={"total_vol_usd": "volume_usd"})   # the query names it total_vol_usd
    for col in ("volume_usd", "pct_volume"):           # volume table (optional columns)
        if col in df.columns:
            df[col] = df[col].astype(float)

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


def usd_text(v):
    if np.isnan(v):
        return "-"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            return f"${v / div:.1f}{suffix}"
    return f"${v:,.0f}"


def draw_table(tb, heading, first_col, rows, levels, k1, style, key, bracket, foot):
    """One table: venue (bracket) + value at each level. Sorted best -> worst at the first dotted line."""
    rows = sorted(rows, key=lambda L: tuple(-np.nan_to_num(L[key][i], nan=-1) for i in range(k1, len(levels))))
    n = len(rows)
    step = min(0.075, 0.86 / (n + 1))
    y0 = 0.93
    cols = (0.61, 0.80, 0.99) if len(levels) == 3 else (0.79, 0.99)
    tb.text(0.02, 1.0, heading, fontsize=15, fontweight="bold", va="center", color="#222222", transform=tb.transAxes)
    tb.text(0.02, y0, first_col, fontsize=13, fontweight="bold", va="center", transform=tb.transAxes)
    for cx, m in zip(cols, levels):
        tb.text(cx, y0, f"≤ {m:g} bps", fontsize=12.5 if len(levels) == 3 else 13, fontweight="bold", ha="right", va="center",
                transform=tb.transAxes)
    tb.plot([0.0, 1.0], [y0 - step * 0.55] * 2, color="#333333", lw=1.2, transform=tb.transAxes, clip_on=False)

    best = [np.nanmax([L[key][i] for L in rows]) for i in range(len(levels))]
    name_max = 14 if len(levels) == 3 else 20
    for r, L in enumerate(rows):
        yy = y0 - step * (r + 1)
        c, mk = style[L["venue"]]
        name = L["venue"] if len(L["venue"]) <= name_max else L["venue"][:name_max - 1] + "…"
        tb.scatter([0.035], [yy], s=110, color=c, marker=mk, edgecolor="white", linewidth=1.2,
                   transform=tb.transAxes, clip_on=False)
        tb.text(0.075, yy, f"{name} ({bracket(L)})", fontsize=12.5, va="center", transform=tb.transAxes)
        for cx, val, b in zip(cols, L[key], best):
            tb.text(cx, yy, pct_text(val), fontsize=12.5, ha="right", va="center",
                    fontweight="bold" if np.isclose(val, b) else "normal",
                    color=c, transform=tb.transAxes)
        tb.plot([0.0, 1.0], [yy - step / 2] * 2, color="#eeeeee", lw=0.8, transform=tb.transAxes, clip_on=False)

    tb.text(0.02, y0 - step * (n + 1.2), foot, fontsize=10.5, color="#777777", va="top", transform=tb.transAxes)


# ----------------------------- chart ----------------------------------
def draw_chart(ax, lines, ykey, vkey, levels, k1, markers, x_min, x_max, style, ylabel):
    """One curve per venue. ykey = which curve (fills or volume), vkey = its values at each level."""
    m1, m2 = markers
    view = np.concatenate([L[ykey][(L["x"] >= x_min - 1e-9) & (L["x"] <= x_max + 1e-9)] for L in lines])
    view = view[~np.isnan(view)]
    lo = max(view.min(), 0.005)                             # below 0.5% is too few fills to matter
    has_100 = bool((view >= 1).any())
    below_100 = view[view < 1]
    hi = below_100.max() if below_100.size else 0.999

    # ---- y-axis ----
    if Y_AXIS == "nines":
        top = expit(logit(max(hi, lo)) + 0.7)          # "100%" sits one small step above the best real value
        y_lo, y_hi = expit(logit(lo) - 0.35), expit(logit(top) + 0.25)
        plot_y = lambda y: np.clip(np.where(y >= 1, top, y), 1e-6, None)   # 0% drops below the chart

        ax.set_yscale("logit")
        ax.set_ylim(y_lo, y_hi)
        cand = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99,
                0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99999]
        cand = [t for t in cand if y_lo <= t < (top if has_100 else y_hi)]
        gap = (logit(y_hi) - logit(y_lo)) / 20             # keep tick labels from touching
        ticks = [top] if has_100 else []                   # start from "100%" so nothing crowds it
        for t in reversed(cand):
            if not ticks or logit(ticks[-1]) - logit(t) >= gap:
                ticks.append(t)
        ticks = sorted(ticks)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda p, _: "100%" if np.isclose(p, top) else tick_text(p)))
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        plot_y = lambda y: y
        ax.set_ylim(max(0, lo - 0.03), 1.005)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    pad = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min - pad, x_max + pad)

    # ---- better / worse than Binance (labels at the top, where the lines are not) ----
    if x_min < 0:
        ax.axvspan(x_min - pad, 0, color="#2ca02c", alpha=0.06, lw=0, zorder=0)
        ax.axvline(0, color="#555555", lw=1.4, zorder=2)
        box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2)
        roomy = (0 - (x_min - pad)) / (x_max - x_min + 2 * pad) > 0.28     # is the green side wide enough?
        ax.annotate("← better than Binance" if roomy else "← better", xy=(0, 1), xycoords=("data", "axes fraction"),
                    xytext=(-8, -8), textcoords="offset points", ha="right", va="top",
                    fontsize=12.5, color="#1e7b1e", fontweight="bold", bbox=box, zorder=6)
        ax.annotate("worse than Binance →" if roomy else "worse →", xy=(0, 1), xycoords=("data", "axes fraction"),
                    xytext=(8, -8), textcoords="offset points", ha="left", va="top",
                    fontsize=12.5, color="#a33a3a", fontweight="bold", bbox=box, zorder=6)

    # ---- venue lines (best venue drawn last = on top) ----
    order = sorted(lines, key=lambda L: tuple(-np.nan_to_num(L[vkey][i], nan=-1) for i in range(k1, len(levels))))
    for L in reversed(order):
        c, mk = style[L["venue"]]
        ax.plot(L["x"], plot_y(L[ykey]), color=c, lw=2.4, marker=mk, ms=5, alpha=0.95, zorder=3)

    # ---- dotted lines at your bps levels (+ dots where each venue crosses every level) ----
    for m in (m1, m2):
        ax.axvline(m, color="#222222", ls=(0, (1.5, 2.5)), lw=1.8, zorder=2)
        left = m == m1                                   # first label sits left of its line, second sits right
        ax.annotate(f"{m:g} bps", xy=(m, 1.0), xycoords=ax.get_xaxis_transform(),
                    xytext=(-5 if left else 5, 6), textcoords="offset points",
                    ha="right" if left else "left", va="bottom",
                    fontsize=14, fontweight="bold", color="#222222")
    for i, m in enumerate(levels):
        for L in lines:
            val = L[vkey][i]
            if not np.isnan(val):
                c, mk = style[L["venue"]]
                ax.scatter([m], plot_y(np.array([val])), s=110, color=c, marker=mk,
                           edgecolor="white", linewidth=1.5, zorder=5)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Spread vs Binance (bps)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="y", color="#dddddd", lw=0.9)
    ax.grid(True, axis="x", color="#f0f0f0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_pair(pair, d, markers, style, ran_at):
    m1, m2 = markers
    x_max = min(50.0, X_ZOOM * max(markers))
    x_min = max(float(d["bps_threshold"].min()), -x_max)     # same distance on the negative side, if data goes that far
    levels = ([0.0] if ZERO_COLUMN and x_min < 0 else []) + [m1, m2]
    k1 = levels.index(m1)

    # one entry per venue: x, fills curve, volume curve, totals, value at each level
    lines = []
    for v, g in d.groupby("venue"):
        g = g.sort_values("bps_threshold")
        x, y = g["bps_threshold"].to_numpy(), g["pct_fills"].to_numpy()
        has_vol = "pct_volume" in g.columns and "volume_usd" in g.columns
        yv = g["pct_volume"].to_numpy() if has_vol else np.full(len(x), np.nan)
        lines.append(dict(venue=v, x=x, y=y, yv=yv, n=int(g["n_fills"].iloc[0]),
                          vol=float(g["volume_usd"].iloc[0]) if has_vol else np.nan,
                          vals=[value_at(x, y, m) for m in levels],
                          vvals=[value_at(x, yv, m) for m in levels]))
    has_vol = any(not np.isnan(L["vol"]) for L in lines)

    # layout: one row per view -> [table | chart]. Row 1 = fills, row 2 = volume.
    # wspace leaves a clear gap so the table never touches the chart's y-axis labels.
    nrows = 2 if has_vol else 1
    fig = plt.figure(figsize=(22, 9.8 * nrows), dpi=150)
    gs = fig.add_gridspec(nrows, 2, width_ratios=[1.45, 2.4], wspace=0.17, hspace=0.26, left=0.01, right=0.99,
                          top=0.92 if nrows == 2 else 0.86, bottom=0.05 if nrows == 2 else 0.09)
    zero_note = "\n≤ 0 bps = got Binance price or better" if 0.0 in levels else ""

    tf, af = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    tf.axis("off")
    draw_table(tf, "BY FILLS  ·  % of fills with spread ≤ X bps", "Venue (fills)", lines, levels, k1, style, "vals",
               lambda L: f"{L['n']:,}",
               "sorted best → worst at the first dotted line\nbold = best in that column" + zero_note)
    draw_chart(af, lines, "y", "vals", levels, k1, markers, x_min, x_max, style, "% of fills with spread ≤ X bps")

    if has_vol:
        tv, av = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        tv.axis("off")
        draw_table(tv, "BY VOLUME  ·  % of USD volume with spread ≤ X bps", "Venue (volume)", lines, levels, k1,
                   style, "vvals", lambda L: usd_text(L["vol"]),
                   "sorted best → worst at the first dotted line\nbold = best in that column\n"
                   "volume = USD size of each fill" + zero_note.replace("got", "volume at"))
        draw_chart(av, lines, "yv", "vvals", levels, k1, markers, x_min, x_max, style,
                   "% of USD volume with spread ≤ X bps")

    note = "  ·  y-axis stretched near 0% and 100% so small gaps are visible" if Y_AXIS == "nines" else ""
    what = "fills and volume" if has_vol else "fills"
    tf.text(0, 1.14, f"{pair}  —  % of {what} with spread ≤ X bps vs Binance",
            transform=tf.transAxes, fontsize=21, fontweight="bold")
    tf.text(0, 1.095, f"{WINDOW_LABEL}  ·  below 0 = better price than Binance  ·  higher line = better{note}",
            transform=tf.transAxes, fontsize=12.5, color="#555555")

    fig.text(0.01, 0.005 if has_vol else 0.0, f"Source: Dune query {QUERY_ID}  ·  data run at {ran_at} UTC",
             fontsize=10, color="#888888")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, pair.replace("/", "-") + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")
    return path


# --------------------------- image links ------------------------------
def git(args, cwd):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        if CHART_TOKEN:
            msg = msg.replace(CHART_TOKEN, "***")          # never print the token
        raise RuntimeError(f"git {args[0]} failed: {msg[:300]}")
    return r.stdout.strip()


def publish_to_github(saved, ran_at):
    """Put today's PNGs on CHART_BRANCH as its only commit (force push = yesterday's images are replaced).
    Links use the commit id, so every day gets new links and Slack never shows a cached old chart."""
    work = tempfile.mkdtemp()
    for _, _, path in saved:
        shutil.copy(path, work)
    git(["init", "-q"], work)
    git(["symbolic-ref", "HEAD", f"refs/heads/{CHART_BRANCH}"], work)
    git(["config", "user.name", "github-actions[bot]"], work)
    git(["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], work)
    git(["add", "-A"], work)
    git(["commit", "-q", "-m", f"fill quality charts, data run at {ran_at} UTC"], work)
    remote = f"https://x-access-token:{CHART_TOKEN}@github.com/{CHART_REPO}.git"
    git(["push", "-q", "--force", remote, f"HEAD:refs/heads/{CHART_BRANCH}"], work)
    sha = git(["rev-parse", "HEAD"], work)
    shutil.rmtree(work, ignore_errors=True)
    print(f"pushed {len(saved)} charts to {CHART_REPO}, branch {CHART_BRANCH} ({sha[:7]})")
    return [(pair, markers, f"https://raw.githubusercontent.com/{CHART_REPO}/{sha}/{os.path.basename(path)}")
            for pair, markers, path in saved]


def link_works(url):
    """Open the link with no login, the same way Slack will. Waits up to ~1 minute."""
    for _ in range(6):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                return True
        except requests.RequestException:
            pass
        time.sleep(10)
    return False


# ------------------------------ slack ---------------------------------
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
    """charts = list of (pair, markers, image_url). Header message first, then one message per chart."""
    title = f"Fill quality vs Binance ({WINDOW_LABEL})"
    header = {"text": title, "blocks": [
        {"type": "header", "text": {"type": "plain_text", "text": title}},
        {"type": "context",
         "elements": [{"type": "mrkdwn",
                       "text": "% of fills with spread ≤ X bps vs Binance, per venue. Below 0 = better price than Binance. "
                               "Higher = better. "
                               "Venue names show fill count in brackets. "
                               f"Dune query {QUERY_ID}, data run at {ran_at} UTC."}]},
    ]}
    if not slack_send(header, "header"):
        raise RuntimeError("Slack header post failed, so the charts were not sent.")

    failed = []
    for pair, (m1, m2), url in charts:
        time.sleep(1.2)                    # webhooks allow about 1 message per second
        msg = {"text": f"{title}: {pair}", "blocks": [
            {"type": "image", "image_url": url, "alt_text": f"{pair} fill quality curve",
             "title": {"type": "plain_text", "text": f"{pair}  ·  dotted lines at {m1:g} and {m2:g} bps"}}]}
        if slack_send(msg, pair):
            print(f"  posted {pair}")
        else:
            failed.append(pair)

    if skipped:
        time.sleep(1.2)
        slack_send({"text": "Not shown: " + ", ".join(skipped)}, "not-shown note")

    print(f"posted {len(charts) - len(failed)} of {len(charts)} charts to Slack")
    if failed:
        raise RuntimeError(f"Some charts were not posted: {failed}")


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
    if not (CHART_REPO and CHART_TOKEN):
        raise SystemExit("GITHUB_REPOSITORY / GITHUB_TOKEN missing. This part runs on GitHub Actions "
                         "(see the yml), or set CHART_REPO + CHART_REPO_TOKEN.")

    charts = publish_to_github(saved, ran_at)
    if not link_works(charts[0][2]):
        raise RuntimeError(f"Slack could not open the image link {charts[0][2]}\n"
                           f"-> This happens when {CHART_REPO} is private. Make it public, or set "
                           "CHART_REPO to a public repo (plus CHART_REPO_TOKEN that can push to it).")
    post_to_slack(charts, ran_at, skipped)


if __name__ == "__main__":
    main()
