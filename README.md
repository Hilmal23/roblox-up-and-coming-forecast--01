# Roblox Up-and-Coming forecast

`main.py` (Python 3.13, standard library only) reads the Up-and-Coming chart
history in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one forecast to
`$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/forecast.json`): for each
of the top 10 places, a probability for each candidate game, plus a top-10
guess. Deterministic: the same history always yields the same bytes.

## How it forecasts

The chart's top-10 set is fairly stable but the order inside it shuffles
hard, so betting everything on the chart standing still loses. The program
blends two rank-distribution models:

- **Gaussian place model (70%).** Each candidate game gets a forecast rank
  (its last rank, pulled 30% toward the rank its player count implies) and
  a spread that grows with the forecast horizon and its recent rank range.
  Place weights are the normal densities, normalised per place, leaving
  30% unnamed for games that enter the chart during the window.
- **Monte Carlo rank model (30%).** Each candidate's latent strength starts
  at minus its last rank and takes Gaussian steps every 3 hours over the
  72-hour window; the fraction of steps each game spends at each place is
  its probability. Seeded, so it is reproducible.

Games that age out of the chart (published more than 28 days before a
window hour) lose weight in proportion to the window they miss, using the
`created` date from the history's details when present.

## Practice result

On the kit's practice instance (history to 2026-11-07T23:00Z, 10 reads in
the window after it) the forecast scores **-34.20** log score (typical
0.033 per place, 6.4 top-10 hits) against the house persistence baseline's
**-43.51** — about 9.3 nats better. Rolling backtests on earlier 72-hour
windows of the same archive average about -39.2 against the baseline's
-50.1.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
