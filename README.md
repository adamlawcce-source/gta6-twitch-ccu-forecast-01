# GTA VI launch-day Twitch viewers, hour by hour

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adamlawcce-source/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adamlawcce-source/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [make.fun's `gta6-twitch-ccu-forecast-01` bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): given the first hour's
Twitch viewer count for the "Grand Theft Auto VI" category on release day
(2026-11-19 UTC), forecast hours 2-24 as five quantiles each (5%, 25%,
50%, 75%, 95%).

## Method

Anchor on hour 1, borrow the shape from past big launches. The case's
`comparables` hold 38-42 past launches' hourly Twitch curves (SullyGnome).
Each launch's release day is expressed as log ratios to its own hour-1
count; the forecast for hour *h* is hour 1 times exp(mu_h), where mu_h is
the kernel-weighted mean of those log ratios, with weights falling off in
log hour-1 size (bandwidth 1.5). The quantile spread is the kernel-weighted
standard deviation of the same ratios per hour (floored 0.25, capped 1.1
log units).

Why condition on hour-1 size: hour 1 alone is a misleading anchor. Small
launches barely exist as a category at 00:00 UTC and explode 10-100x once
the Americas unlock; huge launches are already near peak at hour 1 and
stay flat. The kernel separates the two regimes instead of averaging them.

Leave-one-out cross-validation on the kit's 37 usable comparables:
pinball loss 0.394 (all launches) / 0.256 (hour-1 >= 20k, the regime the
practice cases and GTA VI sit in), vs the house persistence baseline's
0.930 / 0.321.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.
`main.py` reads every `<id>.in` in the input directory and writes
`<id>.out` (`{"hours": [{"hour": 2, "q": [...]}, ...]}`) to the output
directory. If a case ships no usable comparables, it falls back to the
house baseline (flat at hour 1, spread widening as 0.15*sqrt(h-1)).

## License

MIT, see `LICENSE`.
