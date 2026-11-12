# Roblox Up-and-Coming forecast

Run `main.py` with Python 3. It reads `task.json` and `history.ndjson` from
`MAKEFUN_INPUT_DIR` (default `/input`) and writes `forecast.json` under
`MAKEFUN_OUTPUT_DIR` (default `/output`). The JSON contains probabilities for
chart places 1 through 10 and a list of ten distinct predicted universe IDs.

The program uses only the Python standard library, makes no network requests,
and produces the same output bytes for the same input.

## How the forecast works

Three probability estimates are blended. A Gaussian place model contributes
70% of the rank-model portion: it centers each candidate near its latest rank,
pulls that rank toward a player-count estimate, and widens the distribution
according to recent rank variation and forecast horizon. A seeded Monte Carlo
rank simulation contributes the other 30% of that portion, stepping candidate
strengths forward every three hours. Together these rank models supply 75% of
the final probability.

The remaining 25% comes from a place-occupancy table. It counts historical
top-ten occupants by place, with a 24-hour half-life, and smooths the counts
with a 0.05 prior for each candidate and place. Candidates come from the latest
read's top 60 and the top 30 of each of the last eight reads. When a game's
creation date is available, the rank models reduce its weight for the portion
of the forecast window after it ages out of the chart. No mass is reserved for
unnamed games.

## Run locally

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
