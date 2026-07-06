---
name: football-score-model
description: Run a raw-results-only opponent-adjusted national-team football score model from a fixture name. Use when Codex is asked to calculate any supported international fixture, including win/draw/loss probabilities, expected goals, top scorelines, opponent-strength diagnostics, and historical-prior audits without betting odds or external predictions.
---

# Football Score Model

Use the bundled model; never substitute informal estimates or raw-goals-only rankings.

## Run

```powershell
python scripts/run_model.py --fixture "Argentina vs Egypt" --output-dir output
```

Only the fixture is required. Normalize translated team names to dataset labels when needed.

## Rules

- Use completed raw national-team results only; never use odds, tips, forecasts, or prediction articles.
- Resolve target date, competition, knockout stage, neutral status, and home/away order from the raw fixture row.
- Set cutoff to the earliest of match day minus one, run date, and latest completed source date.
- Apply opponent-adjusted ELO, capped historical prior, weighted Poisson, elite-opponent correction, Dixon-Coles, and at least 50,000 simulations.
- Require `hard_audit.json` to report `all_passed: true` before reporting results.
- Report lambdas, win/draw/loss with 95% intervals, top scores, opponent strength, prior impact, and warnings from `prediction.json`.

Read [references/model-spec.md](references/model-spec.md) only when formulas or audit details are needed. Invoke `scripts/model_engine.py` only through `scripts/run_model.py`.
