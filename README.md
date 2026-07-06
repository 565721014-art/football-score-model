# football-score-model

A Codex Skill for opponent-adjusted national-team football score modeling from raw match results. It uses international ELO, weighted Poisson regression, elite-opponent correction, a low-weight historical prior, Dixon-Coles adjustment, and Monte Carlo simulation. It does not use betting odds or external predictions.

## Install with Codex

Send Codex this request:

```text
Use $skill-installer to install https://github.com/565721014-art/football-score-model/tree/main/football-score-model
```

Restart Codex after installation.

## Use

Provide only the fixture:

```text
Use $football-score-model to calculate Argentina vs Egypt.
```

The Skill resolves the fixture context and cutoff, then reports expected goals, win/draw/loss probabilities with confidence intervals, top scorelines, opponent-strength diagnostics, and historical-prior audits.

## Scope

- International national-team fixtures represented in the raw source data
- Python 3.10+
- Dependencies in `football-score-model/scripts/requirements.txt`
