# Roblox Up-and-Coming forecast

`main.py` reads chart history from `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output/forecast.json`).
The output contains probabilities for places 1–10 and a ten-game selection.
The program uses only the Python standard library and produces deterministic
output for the same input history.

## Forecast method

The forecast blends a Gaussian place model and a Monte Carlo rank model,
then assigns 25% of the final probability to a recency-weighted
place-occupancy table. The Gaussian and Monte Carlo components contribute
70% and 30% of the remaining 75%, respectively.

The Gaussian component uses a candidate's latest rank, its player count,
and recent rank variation to distribute probability over places. The
Monte Carlo component simulates rank changes in three-hour steps using a
fixed random seed. Games that age out of the chart lose weight over the
forecast window when their creation date is available.

The occupancy component draws candidates from the latest chart's top 60
and the top 30 of each of the last eight reads. For each place it counts
historical top-ten occupants, halving a read's weight for every 24 hours
of age. A small pseudocount smooths each candidate's place probability.
No probability is reserved for unnamed games: held-out windows showed an
approximately 3.5% rate of previously unseen top-ten occupants.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
