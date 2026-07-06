# Model specification

## Data boundaries

- Accept the fixture name as the only required input.
- Resolve its date, competition class, knockout flag, neutral status, and home/away order from the raw fixture row.
- Set the automatic cutoff to the earliest of the calendar day before the target, the run date, and the source's latest completed match date. Allow an explicit user override.
- Accept completed raw national-team results only.
- Keep dates from 2018-01-01 through the inclusive cutoff.
- Use 2022 through the cutoff as the primary window.
- Use 2018-2021 only as a capped stabilizing prior.
- Exclude missing scores, post-cutoff rows, duplicates, and the target fixture.

## ELO and opponent adjustment

Initialize all teams at 1500 and update chronologically. Use zero home advantage for neutral matches. Multiply pre-2022 ELO K by 0.35.

Use competition K values: World Cup knockout 45, World Cup group 35, World Cup qualifier 25, major continental tournament 28, continental nations league 18, major continental qualifier 20, other continental tournament 25, other continental qualifier 18, friendly 10, other 15.

For a team-match observation:

`OpponentQuality = clamp(0.55, 1.35, exp((OpponentPreMatchELO - 1500) / 600))`

## Weights

Recency: 2026 1.50, 2025 1.25, 2024 1.10, 2023 0.90, 2022 0.75, 2021 0.12, 2020 0.10, 2019 0.08, 2018 0.06.

Competition: World Cup knockout 5.0, World Cup group 3.0, World Cup qualifier 2.0, major continental tournament 2.4, continental nations league 1.5, major continental qualifier 1.6, other continental tournament 1.8, other continental qualifier 1.4, friendly 0.6, other 1.0.

Use `recency × competition × opponent quality` for adjusted observations. Rescale pre-2022 observations so their effective share is at most 8% globally and for each target team.

## Rates and regression

Compute the weighted global goals-per-team baseline. Report raw and opponent-adjusted scored/conceded rates and their baseline-normalized attack/defense ratios.

Fit weighted Poisson observations for both sides of every match:

`log(lambda) = intercept + scoring attack + opponent defense + applicable non-neutral home effect + target competition effect + applicable World Cup knockout effect`

Select ridge strength by held-out weighted negative log likelihood. Center attack and defense effects at zero.

## Elite and historical components

Use opponent pre-match ELO >=1600. Fall back to 1550 if either team has fewer than six matches; warn if either still has fewer than five.

Blend regression, elite subset, and historical prior at 70%/25%/5%, or 82%/13%/5% when elite data is limited. If the prior moves either final lambda by more than 5%, solve for a smaller prior blend and rerun the final blend. Never exceed 5% historical-prior blend.

## Score probabilities

Estimate Dixon-Coles rho from valid low-score observations when possible; otherwise use 0.08. Apply correction only to 0-0, 1-0, 0-1, and 1-1 after final lambdas are computed.

Draw at least 50,000 samples from the corrected joint score matrix. Report win/draw/loss probabilities with Wilson 95% intervals, top ten exact corrected scorelines, and the matrix argmax.

## Mandatory audit

Reject output unless:

- probability matrix and outcomes sum to one;
- target and post-cutoff rows are absent from training;
- historical effective weights are <=8%;
- historical blend and lambda sensitivity are <=5%;
- regression converges and attack/defense effects are centered;
- Dixon-Coles multipliers remain positive;
- reported most-likely score equals the matrix argmax.
# Model specification

## Data boundaries

- Accept the fixture name as the only required input.
- Resolve its date and neutral World Cup context from the future unscored raw fixture row.
- Set the automatic cutoff to the calendar day before the target unless the user explicitly supplies another cutoff.
- Accept completed raw national-team results only.
- Keep dates from 2018-01-01 through the inclusive cutoff.
- Use 2022 through the cutoff as the primary window.
- Use 2018-2021 only as a capped stabilizing prior.
- Exclude missing scores, post-cutoff rows, duplicates, and the target fixture.

## ELO and opponent adjustment

Initialize all teams at 1500 and update chronologically. Use zero home advantage for neutral matches. Multiply pre-2022 ELO K by 0.35.

Use competition K values: World Cup knockout 45, World Cup group 35, World Cup qualifier 25, major continental tournament 28, continental nations league 18, major continental qualifier 20, other continental tournament 25, other continental qualifier 18, friendly 10, other 15.

For a team-match observation:

`OpponentQuality = clamp(0.55, 1.35, exp((OpponentPreMatchELO - 1500) / 600))`

## Weights

Recency: 2026 1.50, 2025 1.25, 2024 1.10, 2023 0.90, 2022 0.75, 2021 0.12, 2020 0.10, 2019 0.08, 2018 0.06.

Competition: World Cup knockout 5.0, World Cup group 3.0, World Cup qualifier 2.0, major continental tournament 2.4, continental nations league 1.5, major continental qualifier 1.6, other continental tournament 1.8, other continental qualifier 1.4, friendly 0.6, other 1.0.

Use `recency × competition × opponent quality` for adjusted observations. Rescale pre-2022 observations so their effective share is at most 8% globally and for each target team.

## Rates and regression

Compute the weighted global goals-per-team baseline. Report raw and opponent-adjusted scored/conceded rates and their baseline-normalized attack/defense ratios.

Fit weighted Poisson observations for both sides of every match:

`log(lambda) = intercept + scoring attack + opponent defense + non-neutral home effect + competition effect + World Cup knockout effect`

Select ridge strength by held-out weighted negative log likelihood. Center attack and defense effects at zero.

## Elite and historical components

Use opponent pre-match ELO >=1600. Fall back to 1550 if either team has fewer than six matches; warn if either still has fewer than five.

Blend regression, elite subset, and historical prior at 70%/25%/5%, or 82%/13%/5% when elite data is limited. If the prior moves either final lambda by more than 5%, solve for a smaller prior blend and rerun the final blend. Never exceed 5% historical-prior blend.

## Score probabilities

Estimate Dixon-Coles rho from valid low-score observations when possible; otherwise use 0.08. Apply correction only to 0-0, 1-0, 0-1, and 1-1 after final lambdas are computed.

Draw at least 50,000 samples from the corrected joint score matrix. Report win/draw/loss probabilities with Wilson 95% intervals, top ten exact corrected scorelines, and the matrix argmax.

## Mandatory audit

Reject output unless:

- probability matrix and outcomes sum to one;
- target and post-cutoff rows are absent from training;
- historical effective weights are <=8%;
- historical blend and lambda sensitivity are <=5%;
- regression converges and attack/defense effects are centered;
- Dixon-Coles multipliers remain positive;
- reported most-likely score equals the matrix argmax.
