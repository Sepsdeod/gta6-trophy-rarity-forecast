# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Sepsdeod/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Sepsdeod/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, one week after launch. It reads one case file (`makefun/gta6-trophies-case/v1`: every valid Exophase read before the checkpoint, oldest first) and writes five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

The model is written from scratch in TypeScript (Node standard library only, no network): per trophy it fits the earn rate on the logit scale against log-time since release over the case's distinct reads, recency-weighted, with slopes shrunk hierarchically by trophy tier (platinum/gold/silver/bronze behave differently), and an honest spread that widens with horizon, fit residual and page staleness. When the history is a single flat page (launch night, before Exophase's first refresh), it applies a refresh prior instead of trusting the stale rates: the first refresh diluted every rate as the tracked-player count jumped.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `<id>.out` for each into `$MAKEFUN_OUTPUT_DIR` (else `/output/`). Deterministic: two runs on the same case write byte-identical output. Standard library only; no installs, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7nvvsk82&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
