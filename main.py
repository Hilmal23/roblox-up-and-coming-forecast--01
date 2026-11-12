#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 for roblox-up-and-coming-forecast-01.

Reads /input/history.ndjson (chart reads, oldest first) and /input/task.json,
writes /output/forecast.json: for each of the top 10 places, a probability for
each candidate game, plus a top-10 guess. Standard library only, no network,
deterministic: the same history always yields the same bytes.

Method (a blend of Gaussian place and Monte Carlo rank models plus a 25%
recency-weighted place-occupancy table, tuned on rolling backtests):

  1. Gaussian place model. Each candidate game gets a forecast rank mu (its
     last rank, pulled 30% toward the rank its player count implies: rank is
     regressed on log(playing) over the last read's top 60) and a spread
     sigma (base 1.0, growing 0.1 per forecast hour out to the window
     midpoint, plus 0.3x its recent rank range). The weight of a game at a
     place is the normal density there; weights are normalised per place.

  2. Monte Carlo rank model. Each candidate's latent strength starts at
     minus its last rank and takes a Gaussian step (sd 0.35 per hour) every
     3 hours over the 72-hour window; games are re-ranked by strength each
     step and the fraction of steps each game spends at each place is its
     probability. Seeded, so it is reproducible.

  3. Recency-weighted place-occupancy table. Recent top-10 occupants contribute
     decaying counts per place, smoothed over candidates from recent reads.

The Gaussian / Monte Carlo 0.70 / 0.30 blend contributes 75%; occupancy
contributes 25%. Unnamed mass is zero because the measured new-entrant rate
is only about 3.5%. Games that age out of the chart (published more than
28 days before a window hour) lose weight in proportion to the window they
miss; the age-out date comes from the details' `created` field when the
history carries it, and is ignored when it does not.

Why this shape: the chart's top-10 set is fairly stable but the order
inside it shuffles hard (a median of 6 of 10 places change between reads),
so persistence (0.5 on the current occupant of each place) scores -43.5 on
the practice window, while an honest spread over the plausible occupants
scores about -34. Rank velocity is deliberately not extrapolated: backtests
show climbing games revert more often than they hold their climb.
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261111
HOURS = 72
STEP_HOURS = 3
N_SIM = 400
GAUSS_WEIGHT = 0.70
GAUSS_UNNAMED = 0.0
MC_UNNAMED = 0.0
MC_SIGMA_HOUR = 0.35
SIG0 = 1.0
SIG_GROWTH = 0.10
PLAY_PULL = 0.30
VEL_POINTS = 5
CAND_RANK = 60
CAND_RECENT_TOP = 30
CAND_RECENT_READS = 8
MIN_P = 0.001
MAX_ROWS_PER_PLACE = 100
AGE_OUT_DAYS = 28


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    return task, reads


def chart_rows(read):
    rows = read.get("up_and_coming") or []
    return sorted(rows, key=lambda r: r["rank"])


def build_series(reads):
    """universe_id -> list of (time, rank, playing), oldest first."""
    series = defaultdict(list)
    for read in reads:
        at = parse_time(read["at"])
        for row in chart_rows(read):
            series[row["universe_id"]].append((at, row["rank"], row.get("playing")))
    return series


def age_out_fraction(details, window_from, window_hours):
    """Fraction of the window a game survives before ageing out (1 = all)."""
    if not details:
        return 1.0
    created = details.get("created")
    if not created:
        return 1.0
    try:
        born = parse_time(created)
    except (ValueError, TypeError):
        return 1.0
    out = born + timedelta(days=AGE_OUT_DAYS)
    if out <= window_from:
        return 0.0
    frac = (out - window_from).total_seconds() / 3600.0 / window_hours
    return min(1.0, max(0.0, frac))


def candidates(series, reads):
    last = chart_rows(reads[-1])
    cand = {row["universe_id"] for row in last if row["rank"] <= CAND_RANK}
    for read in reads[-CAND_RECENT_READS:]:
        for row in chart_rows(read)[:CAND_RECENT_TOP]:
            cand.add(row["universe_id"])
    return cand


def playing_implied_rank(last_read):
    """Linear fit rank ~ a + b*ln(playing) over the last read's top 60."""
    pts = []
    for row in chart_rows(last_read):
        if row["rank"] <= CAND_RANK and row.get("playing"):
            pts.append((math.log(row["playing"]), row["rank"]))
    if len(pts) < 5:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    if den == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / den
    return (my - b * mx, b)


def gaussian_probs(series, cand, reads, window_from, window_hours, details):
    fit = playing_implied_rank(reads[-1])
    horizon = window_hours / 2.0
    mu = {}
    sig = {}
    for gid in cand:
        pts = series[gid][-VEL_POINTS:]
        r0 = pts[-1][1]
        m = float(r0)
        if fit is not None and pts[-1][2]:
            implied = fit[0] + fit[1] * math.log(pts[-1][2])
            m = m - PLAY_PULL * (m - implied)
        ranks = [rk for _, rk, _ in pts]
        vol = (max(ranks) - min(ranks)) / 2.0 if len(ranks) > 1 else 2.0
        mu[gid] = m
        sig[gid] = max(1.0, SIG0 + SIG_GROWTH * horizon + 0.3 * vol)
    probs = {}
    for place in range(1, 11):
        weights = {}
        for gid in cand:
            z = (place - mu[gid]) / sig[gid]
            w = math.exp(-0.5 * z * z) / sig[gid]
            w *= age_out_fraction(details.get(gid), window_from, window_hours)
            weights[gid] = w
        total = sum(weights.values())
        if total <= 0:
            continue
        for gid, w in weights.items():
            p = (w / total) * (1.0 - GAUSS_UNNAMED)
            if p > MIN_P:
                probs[(gid, place)] = p
    return probs


def mc_probs(series, cand, reads, window_from, window_hours, details):
    rng = random.Random(SEED)
    gids = sorted(cand)
    r0 = [series[g][ -1][1] for g in gids]
    # last step (0-based) at which each game is still on the chart
    nsteps = window_hours // STEP_HOURS
    last_step = []
    for g in gids:
        frac = age_out_fraction(details.get(g), window_from, window_hours)
        last_step.append(nsteps if frac >= 1.0 else int(frac * nsteps))
    counts = defaultdict(Counter)
    top_counts = Counter()
    sig = MC_SIGMA_HOUR * math.sqrt(STEP_HOURS)
    gauss = rng.gauss
    n = len(gids)
    for _ in range(N_SIM):
        s = [-r + gauss(0.0, 0.5) for r in r0]
        for step in range(nsteps):
            for i in range(n):
                if step <= last_step[i]:
                    s[i] += gauss(0.0, sig)
                else:
                    s[i] = -1e18
            order = sorted(range(n), key=lambda i: (-s[i], i))
            for k in range(10):
                g = gids[order[k]]
                counts[g][k + 1] += 1
                top_counts[g] += 1
    probs = {}
    denom = N_SIM * nsteps
    for g, counter in counts.items():
        for place, cnt in counter.items():
            probs[(g, place)] = (cnt / denom) * (1.0 - MC_UNNAMED)
    return probs, top_counts

def occupancy_probs(reads, window_from):
    last_at = parse_time(reads[-1]["at"])
    cand = {row["universe_id"] for row in chart_rows(reads[-1])[:60]}
    for read in reads[-8:]:
        cand.update(row["universe_id"] for row in chart_rows(read)[:30])

    counts = defaultdict(Counter)
    for read in reads:
        age_hours = (last_at - parse_time(read["at"])).total_seconds() / 3600.0
        weight = 0.5 ** (age_hours / 24.0)
        for row in chart_rows(read)[:10]:
            place = row["rank"]
            if 1 <= place <= 10:
                counts[place][row["universe_id"]] += weight
    probs = {}
    for place in range(1, 11):
        total = sum(counts[place][gid] + 0.05 for gid in cand)
        for gid in sorted(cand):
            p = (counts[place][gid] + 0.05) / total
            if p > 0.001:
                probs[(gid, place)] = p
    return probs


def persistence_fallback(reads, places):
    last = chart_rows(reads[-1])
    chart = [row["universe_id"] for row in last]
    probs = {}
    for place in range(1, places + 1):
        if place - 1 < len(chart):
            probs[(chart[place - 1], place)] = 0.5
        if place == 1 and len(chart) > 1:
            probs[(chart[1], place)] = 0.3
        elif place > 1:
            if place - 2 < len(chart):
                probs[(chart[place - 2], place)] = 0.15
            if place < len(chart):
                probs[(chart[place], place)] = 0.15
    return probs, chart[:places]


def main():
    task, reads = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {})
    window_from = parse_time(window["from"]) if window.get("from") else parse_time(reads[-1]["at"])
    window_hours = int(window.get("slots", HOURS) * window.get("slot_minutes", 60) / 60) or HOURS

    details = {}
    for read in reads:
        for gid, det in (read.get("details") or {}).items():
            if det:
                try:
                    details[int(gid)] = det
                except (TypeError, ValueError):
                    pass

    if not reads or len(reads) < 2:
        probs, top10 = persistence_fallback(reads, places)
    else:
        series = build_series(reads)
        cand = candidates(series, reads)
        g_probs = gaussian_probs(series, cand, reads, window_from, window_hours, details)
        m_probs, _ = mc_probs(series, cand, reads, window_from, window_hours, details)
        o_probs = occupancy_probs(reads, window_from)
        probs = {}
        for key in sorted(set(g_probs) | set(m_probs) | set(o_probs)):
            probs[key] = 0.75 * (g_probs.get(key, 0.0) * GAUSS_WEIGHT + m_probs.get(key, 0.0) * (1.0 - GAUSS_WEIGHT)) + 0.25 * o_probs.get(key, 0.0)
        mass = Counter()
        for (gid, _), p in probs.items():
            mass[gid] += p
        top10 = [gid for gid, _ in sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]

    # Emit deterministically: rows sorted by place, then p desc, then id.
    rows = []
    by_place = defaultdict(list)
    for (gid, place), p in probs.items():
        if 1 <= place <= places and p > MIN_P:
            by_place[place].append((gid, p))
    for place in range(1, places + 1):
        cell = sorted(by_place.get(place, []), key=lambda gp: (-gp[1], gp[0]))
        cell = cell[:MAX_ROWS_PER_PLACE]
        total = sum(p for _, p in cell)
        if total > 1.0:
            cell = [(gid, p / total) for gid, p in cell]
            total = 1.0
        # Round to 6 decimals; if rounding pushes the sum over 1, trim the largest row.
        rounded = [(gid, round(p, 6)) for gid, p in cell]
        excess = sum(p for _, p in rounded) - 1.0
        if excess > 0:
            gid0, p0 = rounded[0]
            rounded[0] = (gid0, round(p0 - excess, 6))
        for gid, p in rounded:
            if p > 0:
                rows.append({"universe_id": int(gid), "place": place, "p": p})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10 must hold exactly `places` distinct ids; pad from the ranked rows if short.
    seen = set(top10)
    if len(top10) < places:
        ranked = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0])) if 'mass' in dir() else []
        for gid, _ in ranked:
            if gid not in seen:
                top10.append(gid)
                seen.add(gid)
            if len(top10) == places:
                break
    if len(top10) < places:
        for (gid, _), _ in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0][0])):
            if gid not in seen:
                top10.append(gid)
                seen.add(gid)
            if len(top10) == places:
                break
    top10 = [int(g) for g in top10[:places]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"probabilities": rows, "top10": top10}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
