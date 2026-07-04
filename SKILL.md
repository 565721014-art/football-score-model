---
name: model-football-scorelines
description: Compute neutral 2026 World Cup knockout scoreline probabilities from only a fixture name using the user's opponent-adjusted ELO, capped historical prior, weighted Poisson regression, elite-opponent correction, Dixon-Coles adjustment, and Monte Carlo model. Use when Codex is given a matchup such as "Argentina vs Egypt" and asked for national-team win/draw/loss probabilities, expected goals, top scorelines, opponent-strength diagnostics, or historical-prior audits without betting odds, prediction sites, expert forecasts, or precomputed predictions.
---

# Model Football Scorelines

Run the bundled deterministic model; do not replace it with informal estimates or a raw-goals-only calculation.

## Workflow

1. Accept a fixture as the only required user input. Normalize translated team names to exact dataset labels when necessary.
2. Use a local raw-results CSV when the user supplies one. Otherwise let the runner download its default raw-results CSV directly.
3. Never browse or use odds, tips, forecasts, expert predictions, social-media predictions, or precomputed prediction articles.
4. Run `scripts/run_model.py` with Python 3.10+ and the packages in `scripts/requirements.txt`.
5. Read `hard_audit.json` first. Reject the run if `all_passed` is false.
6. Read `prediction.json` and report every requested dataset, opponent, raw-versus-adjusted, prior-impact, parameter, outcome, scoreline, and diagnostic field.

## Command

```powershell
python scripts/run_model.py `
  --fixture "Argentina vs Egypt" `
  --output-dir output
```

The runner locates the future unscored fixture, reads its date and neutral World Cup metadata, and uses the day before the match as the inclusive cutoff. Add `--source C:\path\results.csv` to use an existing raw CSV. Use `--cutoff` or `--target-date` only for an explicit user override. Keep at least 50,000 simulations.

## Required behavior

- Treat the target as neutral and add no home advantage.
- Use completed matches from 2018-01-01 through the cutoff only.
- Treat 2022 onward as the primary window and 2018-2021 as a weak prior.
- Keep every effective pre-2022 team-level weight at or below 8%.
- Keep the explicit historical-prior lambda blend at or below 5% and rescale it if either lambda moves by more than 5%.
- Require opponent ELO and opponent-quality adjustment; never rank attack from raw goals alone.
- Use the elite threshold and fallback rules implemented by the model.
- Estimate the World Cup knockout effect only from valid completed observations.
- Apply Dixon-Coles only after final lambdas are computed.
- Return the exact adjusted score matrix plus 50,000-draw simulation probabilities and Wilson 95% intervals.
- Verify that the reported most-likely score equals the matrix argmax.

Read [references/model-spec.md](references/model-spec.md) when auditing formulas, weights, diagnostics, or output semantics. The executable model is in `scripts/model_engine.py`; invoke it through `scripts/run_model.py` so configuration and hard checks are applied.
