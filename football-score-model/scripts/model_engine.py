from __future__ import annotations

# Internal validated engine template. Portugal/Croatia and the 2026 dates are
# placeholders replaced safely by run_model.py before execution. Invoke this
# engine through run_model.py; do not run this file directly.

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "raw_recent_international" / "results_github_latest.csv"
OUTPUT = ROOT / "portugal_croatia_outputs"
START_DATE = pd.Timestamp("2018-01-01")
PRIMARY_START = pd.Timestamp("2022-01-01")
CUTOFF_DATE = pd.Timestamp("2026-06-30")
TARGET_DATE = pd.Timestamp("2026-07-02")
TARGET_TEAMS = ("Portugal", "Croatia")
TARGET_COMPETITION_CLASS = "world_cup_knockout"
TARGET_REGRESSION_TYPE = "world_cup"
TARGET_KNOCKOUT = 1.0
TARGET_NEUTRAL = True
TARGET_TEAM_A_IS_HOME = True
N_SIMULATIONS = 50_000
HISTORICAL_CAP = 0.08

RECENCY_WEIGHTS = {
    2018: 0.06,
    2019: 0.08,
    2020: 0.10,
    2021: 0.12,
    2022: 0.75,
    2023: 0.90,
    2024: 1.10,
    2025: 1.25,
    2026: 1.50,
}

COMPETITION_WEIGHTS = {
    "world_cup_knockout": 5.0,
    "world_cup_group": 3.0,
    "world_cup_qualifier": 2.0,
    "uefa_euro": 2.4,
    "uefa_nations_league": 1.5,
    "uefa_qualifier": 1.6,
    "continental_tournament": 1.8,
    "continental_qualifier": 1.4,
    "friendly": 0.6,
    "other": 1.0,
}

ELO_K = {
    "world_cup_knockout": 45.0,
    "world_cup_group": 35.0,
    "world_cup_qualifier": 25.0,
    "uefa_euro": 28.0,
    "uefa_nations_league": 18.0,
    "uefa_qualifier": 20.0,
    "continental_tournament": 25.0,
    "continental_qualifier": 18.0,
    "friendly": 10.0,
    "other": 15.0,
}

CONTINENTAL_TOURNAMENTS = {
    "AFC Asian Cup",
    "AFF Championship",
    "ASEAN Championship",
    "African Cup of Nations",
    "Arab Cup",
    "CAFA Nations Cup",
    "CONCACAF Nations League",
    "CONCACAF Series",
    "COSAFA Cup",
    "Copa América",
    "EAFF Championship",
    "Gold Cup",
    "Gulf Cup",
    "Oceania Nations Cup",
    "Pacific Games",
    "SAFF Cup",
    "WAFF Championship",
    "CONMEBOL–UEFA Cup of Champions",
}

CONTINENTAL_QUALIFIERS = {
    "AFC Asian Cup qualification",
    "AFF Championship qualification",
    "ASEAN Championship qualification",
    "African Cup of Nations qualification",
    "Arab Cup qualification",
    "CONCACAF Nations League qualification",
    "Copa América qualification",
    "EAFF Championship qualification",
    "Gold Cup qualification",
    "Oceania Nations Cup qualification",
}

REGRESSION_TYPES = [
    "friendly",
    "other",
    "continental_qualifier",
    "continental_tournament",
    "uefa_qualifier",
    "uefa_nations_league",
    "uefa_euro",
    "world_cup_qualifier",
    "world_cup",
]


def is_world_cup_knockout(date: pd.Timestamp) -> bool:
    if date.year == 2018:
        return date >= pd.Timestamp("2018-06-30")
    if date.year == 2022:
        return date >= pd.Timestamp("2022-12-03")
    if date.year == 2026:
        return date >= pd.Timestamp("2026-07-01")
    return False


def classify_competition(row: pd.Series) -> str:
    tournament = str(row["tournament"])
    if tournament == "FIFA World Cup":
        return "world_cup_knockout" if is_world_cup_knockout(row["date"]) else "world_cup_group"
    if tournament == "FIFA World Cup qualification":
        return "world_cup_qualifier"
    if tournament == "UEFA Euro":
        return "uefa_euro"
    if tournament == "UEFA Nations League":
        return "uefa_nations_league"
    if tournament == "UEFA Euro qualification":
        return "uefa_qualifier"
    if tournament in CONTINENTAL_QUALIFIERS:
        return "continental_qualifier"
    if tournament in CONTINENTAL_TOURNAMENTS:
        return "continental_tournament"
    if tournament == "Friendly":
        return "friendly"
    return "other"


def cap_scaler(hist_weight: float, recent_weight: float, cap: float = HISTORICAL_CAP) -> float:
    if hist_weight <= 0.0:
        return 1.0
    maximum_hist = cap * recent_weight / (1.0 - cap)
    return float(min(1.0, maximum_hist / hist_weight))


def load_data() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(SOURCE)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    invalid_dates = int(raw["date"].isna().sum())
    parsed = raw.dropna(subset=["date"]).copy()
    scores_present = parsed[["home_score", "away_score"]].notna().all(axis=1)
    pre_2018_completed = int(((parsed["date"] < START_DATE) & scores_present).sum())
    after_cutoff = int((parsed["date"] > CUTOFF_DATE).sum())
    after_cutoff_completed = int(((parsed["date"] > CUTOFF_DATE) & scores_present).sum())
    in_window = parsed[(parsed["date"] >= START_DATE) & (parsed["date"] <= CUTOFF_DATE)].copy()
    missing_in_window = int(in_window[["home_score", "away_score"]].isna().any(axis=1).sum())
    data = in_window.dropna(subset=["home_score", "away_score"]).copy()
    target_rows_in_raw = raw[
        ((raw["home_team"] == "Portugal") & (raw["away_team"] == "Croatia"))
        | ((raw["home_team"] == "Croatia") & (raw["away_team"] == "Portugal"))
    ].copy()
    target_future_rows = target_rows_in_raw[pd.to_datetime(target_rows_in_raw["date"], errors="coerce") > CUTOFF_DATE]
    data["home_score"] = data["home_score"].astype(int)
    data["away_score"] = data["away_score"].astype(int)
    duplicate_key = ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    duplicates_removed = int(data.duplicated(duplicate_key, keep="first").sum())
    data = data.drop_duplicates(duplicate_key, keep="first")
    data["competition_class"] = data.apply(classify_competition, axis=1)
    data["regression_type"] = np.where(
        data["competition_class"].isin(["world_cup_group", "world_cup_knockout"]),
        "world_cup",
        data["competition_class"],
    )
    data["knockout"] = data["competition_class"].eq("world_cup_knockout").astype(int)
    data["recency_weight"] = data["date"].dt.year.map(RECENCY_WEIGHTS).astype(float)
    data["competition_weight"] = data["competition_class"].map(COMPETITION_WEIGHTS).astype(float)
    data["base_weight_uncapped"] = data["recency_weight"] * data["competition_weight"]
    data = data.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    metadata = {
        "source_rows": int(len(raw)),
        "invalid_date_rows": invalid_dates,
        "pre_2018_completed_matches_discarded": pre_2018_completed,
        "matches_after_2026_06_30_discarded": after_cutoff,
        "completed_matches_after_cutoff_discarded": after_cutoff_completed,
        "missing_score_rows_excluded_within_valid_window": missing_in_window,
        "missing_score_rows_in_entire_source": int(parsed[["home_score", "away_score"]].isna().any(axis=1).sum()),
        "duplicate_rows_removed": duplicates_removed,
        "target_future_rows_excluded": int(len(target_future_rows)),
        "global_baseline_matches_used": int(len(data)),
        "actual_start": data["date"].min().strftime("%Y-%m-%d"),
        "actual_end": data["date"].max().strftime("%Y-%m-%d"),
        "competition_counts": {
            str(k): int(v) for k, v in data["competition_class"].value_counts().sort_index().items()
        },
    }
    return data, metadata


def run_elo(data: pd.DataFrame, home_advantage_elo: float, keep_pre_match: bool) -> dict:
    ratings: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    pre_home = np.zeros(len(data), dtype=float)
    pre_away = np.zeros(len(data), dtype=float)
    squared_errors: list[float] = []
    for _, day_rows in data.groupby("date", sort=True):
        day_changes: defaultdict[str, float] = defaultdict(float)
        for idx, row in day_rows.iterrows():
            home_rating = ratings[row["home_team"]]
            away_rating = ratings[row["away_team"]]
            pre_home[idx] = home_rating
            pre_away[idx] = away_rating
            advantage = 0.0 if bool(row["neutral"]) else home_advantage_elo
            expected = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - advantage) / 400.0))
            actual = 1.0 if row["home_score"] > row["away_score"] else 0.0 if row["home_score"] < row["away_score"] else 0.5
            k = ELO_K[row["competition_class"]]
            if row["date"] < PRIMARY_START:
                k *= 0.35
            change = k * (actual - expected)
            day_changes[row["home_team"]] += change
            day_changes[row["away_team"]] -= change
            squared_errors.append((actual - expected) ** 2)
        for team, change in day_changes.items():
            ratings[team] += change
    return {
        "ratings": dict(ratings),
        "pre_home": pre_home if keep_pre_match else None,
        "pre_away": pre_away if keep_pre_match else None,
        "brier": float(np.mean(squared_errors)),
    }


def build_elo(data: pd.DataFrame) -> dict:
    candidates = np.arange(0.0, 151.0, 5.0)
    trials = [(run_elo(data, float(value), False)["brier"], float(value)) for value in candidates]
    best_brier, best_advantage = min(trials)
    result = run_elo(data, best_advantage, True)
    result["initial"] = 1500.0
    result["home_advantage_elo_non_neutral"] = best_advantage
    result["brier"] = best_brier
    return result


def opponent_quality(values: np.ndarray) -> np.ndarray:
    return np.clip(np.exp((values - 1500.0) / 600.0), 0.55, 1.35)


def attach_weights(data: pd.DataFrame, elo: dict) -> tuple[pd.DataFrame, dict]:
    result = data.copy()
    result["pre_elo_home"] = elo["pre_home"]
    result["pre_elo_away"] = elo["pre_away"]
    result["home_opponent_quality"] = opponent_quality(result["pre_elo_away"].to_numpy(dtype=float))
    result["away_opponent_quality"] = opponent_quality(result["pre_elo_home"].to_numpy(dtype=float))
    result["home_obs_weight_uncapped"] = result["base_weight_uncapped"] * result["home_opponent_quality"]
    result["away_obs_weight_uncapped"] = result["base_weight_uncapped"] * result["away_opponent_quality"]
    historical = result["date"] < PRIMARY_START
    hist_global = float(result.loc[historical, ["home_obs_weight_uncapped", "away_obs_weight_uncapped"]].to_numpy().sum())
    recent_global = float(result.loc[~historical, ["home_obs_weight_uncapped", "away_obs_weight_uncapped"]].to_numpy().sum())
    global_scaler = cap_scaler(hist_global, recent_global)

    target_scalers: dict[str, float] = {}
    for team in TARGET_TEAMS:
        involved = (result["home_team"] == team) | (result["away_team"] == team)
        home_side = result["home_team"] == team
        team_weights = np.where(home_side, result["home_obs_weight_uncapped"], result["away_obs_weight_uncapped"])
        hist_weight = float(np.sum(team_weights[involved & historical]))
        recent_weight = float(np.sum(team_weights[involved & ~historical]))
        target_scalers[team] = cap_scaler(hist_weight, recent_weight)

    home_scaler = np.full(len(result), global_scaler, dtype=float)
    away_scaler = np.full(len(result), global_scaler, dtype=float)
    for team, scaler in target_scalers.items():
        touches = ((result["home_team"] == team) | (result["away_team"] == team)).to_numpy()
        home_scaler[touches] = np.minimum(home_scaler[touches], scaler)
        away_scaler[touches] = np.minimum(away_scaler[touches], scaler)
    home_scaler[~historical.to_numpy()] = 1.0
    away_scaler[~historical.to_numpy()] = 1.0
    result["home_obs_weight"] = result["home_obs_weight_uncapped"] * home_scaler
    result["away_obs_weight"] = result["away_obs_weight_uncapped"] * away_scaler
    result["match_weight"] = 0.5 * (result["home_obs_weight"] + result["away_obs_weight"])

    hist_after = float(result.loc[historical, ["home_obs_weight", "away_obs_weight"]].to_numpy().sum())
    recent_after = float(result.loc[~historical, ["home_obs_weight", "away_obs_weight"]].to_numpy().sum())
    audit = {
        "global_historical_cap_scaler": global_scaler,
        "target_historical_cap_scalers": target_scalers,
        "global_effective_historical_weight_pct": 100.0 * hist_after / (hist_after + recent_after),
        "global_historical_weight_before_cap": hist_global,
        "global_recent_weight": recent_global,
    }
    return result, audit


def weighted_rate(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return float(np.sum(value_array * weight_array) / np.sum(weight_array))


def global_baseline(data: pd.DataFrame, recent_only: bool = False) -> float:
    rows = data[data["date"] >= PRIMARY_START] if recent_only else data
    goals = np.concatenate([rows["home_score"].to_numpy(dtype=float), rows["away_score"].to_numpy(dtype=float)])
    weights = np.concatenate([rows["home_obs_weight"].to_numpy(dtype=float), rows["away_obs_weight"].to_numpy(dtype=float)])
    return weighted_rate(goals, weights)


def team_match_rows(data: pd.DataFrame, team: str) -> tuple[pd.DataFrame, dict]:
    rows = data[(data["home_team"] == team) | (data["away_team"] == team)].copy()
    is_home = rows["home_team"].eq(team)
    rows["team"] = team
    rows["opponent"] = np.where(is_home, rows["away_team"], rows["home_team"])
    rows["goals_scored"] = np.where(is_home, rows["home_score"], rows["away_score"])
    rows["goals_conceded"] = np.where(is_home, rows["away_score"], rows["home_score"])
    rows["opponent_pre_elo"] = np.where(is_home, rows["pre_elo_away"], rows["pre_elo_home"])
    rows["opponent_quality"] = opponent_quality(rows["opponent_pre_elo"].to_numpy(dtype=float))
    rows["raw_weight_uncapped"] = rows["base_weight_uncapped"]
    rows["adjusted_weight_uncapped"] = rows["base_weight_uncapped"] * rows["opponent_quality"]
    historical = rows["date"] < PRIMARY_START
    adjusted_scaler = cap_scaler(
        float(rows.loc[historical, "adjusted_weight_uncapped"].sum()),
        float(rows.loc[~historical, "adjusted_weight_uncapped"].sum()),
    )
    raw_scaler = cap_scaler(
        float(rows.loc[historical, "raw_weight_uncapped"].sum()),
        float(rows.loc[~historical, "raw_weight_uncapped"].sum()),
    )
    rows["adjusted_weight"] = rows["adjusted_weight_uncapped"]
    rows["raw_weight"] = rows["raw_weight_uncapped"]
    rows.loc[historical, "adjusted_weight"] *= adjusted_scaler
    rows.loc[historical, "raw_weight"] *= raw_scaler
    hist_adjusted = float(rows.loc[historical, "adjusted_weight"].sum())
    total_adjusted = float(rows["adjusted_weight"].sum())
    audit = {
        "adjusted_historical_cap_scaler": adjusted_scaler,
        "raw_historical_cap_scaler": raw_scaler,
        "effective_historical_weight_pct": 100.0 * hist_adjusted / total_adjusted,
    }
    return rows, audit


def summarize_team(rows: pd.DataFrame, baseline: float, audit: dict) -> dict:
    recent = rows[rows["date"] >= PRIMARY_START]
    prior = rows[rows["date"] < PRIMARY_START]
    raw_scored = weighted_rate(rows["goals_scored"], rows["raw_weight"])
    raw_conceded = weighted_rate(rows["goals_conceded"], rows["raw_weight"])
    adjusted_scored = weighted_rate(rows["goals_scored"], rows["adjusted_weight"])
    adjusted_conceded = weighted_rate(rows["goals_conceded"], rows["adjusted_weight"])
    return {
        "primary_matches_2022_2026": int(len(recent)),
        "historical_prior_matches_2018_2021": int(len(prior)),
        "actual_start": rows["date"].min().strftime("%Y-%m-%d"),
        "actual_end": rows["date"].max().strftime("%Y-%m-%d"),
        "average_opponent_elo": float(rows["opponent_pre_elo"].mean()),
        "weighted_average_opponent_elo": weighted_rate(rows["opponent_pre_elo"], rows["adjusted_weight"]),
        "average_opponent_elo_2022_2026": float(recent["opponent_pre_elo"].mean()),
        "average_opponent_elo_2018_2021": float(prior["opponent_pre_elo"].mean()),
        "opponent_quality_correction_factor": weighted_rate(rows["opponent_quality"], rows["raw_weight"]),
        "raw_weighted_goals_scored": raw_scored,
        "opponent_adjusted_goals_scored": adjusted_scored,
        "raw_weighted_goals_conceded": raw_conceded,
        "opponent_adjusted_goals_conceded": adjusted_conceded,
        "raw_attack": raw_scored / baseline,
        "adjusted_attack": adjusted_scored / baseline,
        "raw_defense": raw_conceded / baseline,
        "adjusted_defense": adjusted_conceded / baseline,
        "attack_correction_factor": adjusted_scored / raw_scored,
        "defense_correction_factor": adjusted_conceded / raw_conceded,
        **audit,
    }


def build_observations(data: pd.DataFrame) -> dict:
    teams = sorted(set(data["home_team"]) | set(data["away_team"]))
    team_index = {team: idx for idx, team in enumerate(teams)}
    comp_index = {name: idx for idx, name in enumerate(REGRESSION_TYPES)}
    home_team = data["home_team"].map(team_index).to_numpy(dtype=int)
    away_team = data["away_team"].map(team_index).to_numpy(dtype=int)
    comp = data["regression_type"].map(comp_index).to_numpy(dtype=int)
    non_neutral = (~data["neutral"].astype(bool)).to_numpy(dtype=float)
    zero = np.zeros(len(data), dtype=float)
    return {
        "y": np.concatenate([data["home_score"].to_numpy(dtype=float), data["away_score"].to_numpy(dtype=float)]),
        "scoring": np.concatenate([home_team, away_team]),
        "opponent": np.concatenate([away_team, home_team]),
        "home_indicator": np.concatenate([non_neutral, zero]),
        "competition": np.concatenate([comp, comp]),
        "knockout": np.concatenate([data["knockout"].to_numpy(dtype=float), data["knockout"].to_numpy(dtype=float)]),
        "weight": np.concatenate([data["home_obs_weight"].to_numpy(dtype=float), data["away_obs_weight"].to_numpy(dtype=float)]),
        "date": np.concatenate([data["date"].to_numpy(), data["date"].to_numpy()]),
        "teams": teams,
        "team_index": team_index,
        "comp_index": comp_index,
        "match_count": len(data),
    }


def fit_poisson_coordinate(obs: dict, mask: np.ndarray, ridge: float, max_iter: int = 500) -> dict:
    y = obs["y"][mask]
    scoring = obs["scoring"][mask]
    opponent = obs["opponent"][mask]
    home_ind = obs["home_indicator"][mask]
    comp = obs["competition"][mask]
    knockout = obs["knockout"][mask]
    weight = obs["weight"][mask]
    team_count = len(obs["teams"])
    comp_count = len(obs["comp_index"])
    attack = np.zeros(team_count)
    defense = np.zeros(team_count)
    comp_effect = np.zeros(comp_count)
    home_effect = 0.0
    knockout_effect = 0.0
    intercept = math.log(float(np.sum(weight * y) / np.sum(weight)))
    reference_index = obs["comp_index"]["friendly"]
    previous_objective = math.inf
    converged = False

    def eta_mu() -> tuple[np.ndarray, np.ndarray]:
        eta = intercept + attack[scoring] + defense[opponent] + home_effect * home_ind + comp_effect[comp] + knockout_effect * knockout
        return eta, np.exp(np.clip(eta, -10.0, 5.0))

    for iteration in range(1, max_iter + 1):
        max_change = 0.0
        _, mu = eta_mu()
        delta = float(np.clip(np.sum(weight * (y - mu)) / np.sum(weight * mu), -0.5, 0.5) * 0.7)
        intercept += delta
        max_change = max(max_change, abs(delta))

        _, mu = eta_mu()
        grad = np.bincount(scoring, weights=weight * (y - mu), minlength=team_count) - ridge * attack
        hess = np.bincount(scoring, weights=weight * mu, minlength=team_count) + ridge
        delta_vec = np.clip(grad / hess, -0.5, 0.5) * 0.7
        attack += delta_vec
        center = float(attack.mean())
        attack -= center
        intercept += center
        max_change = max(max_change, float(np.max(np.abs(delta_vec))))

        _, mu = eta_mu()
        grad = np.bincount(opponent, weights=weight * (y - mu), minlength=team_count) - ridge * defense
        hess = np.bincount(opponent, weights=weight * mu, minlength=team_count) + ridge
        delta_vec = np.clip(grad / hess, -0.5, 0.5) * 0.7
        defense += delta_vec
        center = float(defense.mean())
        defense -= center
        intercept += center
        max_change = max(max_change, float(np.max(np.abs(delta_vec))))

        _, mu = eta_mu()
        grad_scalar = float(np.sum(weight * home_ind * (y - mu)) - ridge * home_effect)
        hess_scalar = float(np.sum(weight * home_ind * mu) + ridge)
        delta = float(np.clip(grad_scalar / hess_scalar, -0.5, 0.5) * 0.7)
        home_effect += delta
        max_change = max(max_change, abs(delta))

        for category in range(comp_count):
            if category == reference_index:
                continue
            _, mu = eta_mu()
            select = comp == category
            grad_scalar = float(np.sum(weight[select] * (y[select] - mu[select])) - ridge * comp_effect[category])
            hess_scalar = float(np.sum(weight[select] * mu[select]) + ridge)
            delta = float(np.clip(grad_scalar / hess_scalar, -0.5, 0.5) * 0.7)
            comp_effect[category] += delta
            max_change = max(max_change, abs(delta))

        _, mu = eta_mu()
        grad_scalar = float(np.sum(weight * knockout * (y - mu)) - ridge * knockout_effect)
        hess_scalar = float(np.sum(weight * knockout * mu) + ridge)
        delta = float(np.clip(grad_scalar / hess_scalar, -0.5, 0.5) * 0.7)
        knockout_effect += delta
        max_change = max(max_change, abs(delta))

        eta, mu = eta_mu()
        objective = float(np.sum(weight * (mu - y * eta)) + 0.5 * ridge * (
            np.sum(attack ** 2) + np.sum(defense ** 2) + np.sum(comp_effect ** 2) + home_effect ** 2 + knockout_effect ** 2
        ))
        if abs(previous_objective - objective) < 1e-8 * (1.0 + abs(objective)) and max_change < 1e-5:
            converged = True
            break
        previous_objective = objective
    return {
        "intercept": intercept,
        "attack": attack,
        "defense": defense,
        "home_effect": home_effect,
        "competition_effect": comp_effect,
        "knockout_effect": knockout_effect,
        "ridge": ridge,
        "iterations": iteration,
        "converged": converged,
    }


def predict_observations(obs: dict, model: dict, mask: np.ndarray) -> np.ndarray:
    eta = (
        model["intercept"]
        + model["attack"][obs["scoring"][mask]]
        + model["defense"][obs["opponent"][mask]]
        + model["home_effect"] * obs["home_indicator"][mask]
        + model["competition_effect"][obs["competition"][mask]]
        + model["knockout_effect"] * obs["knockout"][mask]
    )
    return np.exp(np.clip(eta, -10.0, 5.0))


def weighted_poisson_nll(obs: dict, model: dict, mask: np.ndarray) -> float:
    y = obs["y"][mask]
    weight = obs["weight"][mask]
    mu = predict_observations(obs, model, mask)
    factorial = np.asarray([math.lgamma(value + 1.0) for value in y])
    return float(np.sum(weight * (mu - y * np.log(mu) + factorial)) / np.sum(weight))


def fit_weighted_poisson(obs: dict) -> dict:
    dates = pd.to_datetime(obs["date"])
    train = np.asarray(dates < pd.Timestamp("2025-01-01"))
    validate = ~train
    candidates = [0.1, 0.3, 1.0, 3.0, 10.0]
    validation: list[dict] = []
    for ridge in candidates:
        candidate = fit_poisson_coordinate(obs, train, ridge)
        validation.append({"ridge": ridge, "weighted_nll": weighted_poisson_nll(obs, candidate, validate)})
    best_ridge = min(validation, key=lambda item: item["weighted_nll"])["ridge"]
    full_mask = np.ones(len(obs["y"]), dtype=bool)
    result = fit_poisson_coordinate(obs, full_mask, best_ridge)
    result["ridge_validation"] = validation
    result["full_weighted_nll"] = weighted_poisson_nll(obs, result, full_mask)
    return result


def target_home_indicator(scoring_team: str) -> float:
    if TARGET_NEUTRAL:
        return 0.0
    team_a_is_scoring_home = scoring_team == TARGET_TEAMS[0] and TARGET_TEAM_A_IS_HOME
    team_b_is_scoring_home = scoring_team == TARGET_TEAMS[1] and not TARGET_TEAM_A_IS_HOME
    return float(team_a_is_scoring_home or team_b_is_scoring_home)


def target_context_factor(model: dict, obs: dict, scoring_team: str) -> float:
    competition_index = obs["comp_index"][TARGET_REGRESSION_TYPE]
    eta = (
        model["competition_effect"][competition_index]
        + model["knockout_effect"] * TARGET_KNOCKOUT
        + model["home_effect"] * target_home_indicator(scoring_team)
    )
    return float(math.exp(eta))


def target_regression_lambda(model: dict, obs: dict, scoring_team: str, opponent: str) -> float:
    eta = (
        model["intercept"]
        + model["attack"][obs["team_index"][scoring_team]]
        + model["defense"][obs["team_index"][opponent]]
        + math.log(target_context_factor(model, obs, scoring_team))
    )
    return float(math.exp(eta))


def choose_elite_threshold(team_rows: dict[str, pd.DataFrame]) -> tuple[float, dict, dict, bool]:
    counts_1600 = {team: int((rows["opponent_pre_elo"] >= 1600.0).sum()) for team, rows in team_rows.items()}
    threshold = 1600.0 if all(count >= 6 for count in counts_1600.values()) else 1550.0
    counts = {team: int((rows["opponent_pre_elo"] >= threshold).sum()) for team, rows in team_rows.items()}
    return threshold, counts, counts_1600, any(count < 5 for count in counts.values())


def subset_rates(rows: pd.DataFrame, baseline: float, threshold: float | None = None, recent_only: bool = False) -> dict:
    subset = rows.copy()
    if threshold is not None:
        subset = subset[subset["opponent_pre_elo"] >= threshold]
    if recent_only:
        subset = subset[subset["date"] >= PRIMARY_START]
    scored = weighted_rate(subset["goals_scored"], subset["adjusted_weight"])
    conceded = weighted_rate(subset["goals_conceded"], subset["adjusted_weight"])
    return {
        "matches": int(len(subset)),
        "primary_matches": int((subset["date"] >= PRIMARY_START).sum()),
        "prior_matches": int((subset["date"] < PRIMARY_START).sum()),
        "adjusted_goals_scored": scored,
        "adjusted_goals_conceded": conceded,
        "adjusted_attack": scored / baseline,
        "adjusted_defense": conceded / baseline,
        "weighted_average_opponent_elo": weighted_rate(subset["opponent_pre_elo"], subset["adjusted_weight"]),
    }


def historical_prior_rates(rows: pd.DataFrame, baseline: float) -> dict:
    prior = rows[rows["date"] < PRIMARY_START]
    scored = weighted_rate(prior["goals_scored"], prior["adjusted_weight_uncapped"])
    conceded = weighted_rate(prior["goals_conceded"], prior["adjusted_weight_uncapped"])
    return {
        "matches": int(len(prior)),
        "adjusted_goals_scored": scored,
        "adjusted_goals_conceded": conceded,
        "adjusted_attack": scored / baseline,
        "adjusted_defense": conceded / baseline,
    }


def fit_dixon_coles(data: pd.DataFrame, obs: dict, model: dict) -> tuple[float, dict]:
    full_mask = np.ones(len(obs["y"]), dtype=bool)
    predicted = predict_observations(obs, model, full_mask)
    count = obs["match_count"]
    lambda_home, lambda_away = predicted[:count], predicted[count:]
    home_goals = data["home_score"].to_numpy(dtype=int)
    away_goals = data["away_score"].to_numpy(dtype=int)
    weight = data["match_weight"].to_numpy(dtype=float)
    low_score_matches = int(((home_goals <= 1) & (away_goals <= 1)).sum())
    if low_score_matches < 30:
        return 0.08, {"estimated": False, "warning": "Too few low-score matches; rho fixed at 0.08."}
    best: tuple[float, float] | None = None
    for rho in np.arange(-0.20, 0.20001, 0.0002):
        tau = np.ones(count, dtype=float)
        mask00 = (home_goals == 0) & (away_goals == 0)
        mask10 = (home_goals == 1) & (away_goals == 0)
        mask01 = (home_goals == 0) & (away_goals == 1)
        mask11 = (home_goals == 1) & (away_goals == 1)
        tau[mask00] = 1.0 - lambda_home[mask00] * lambda_away[mask00] * rho
        tau[mask10] = 1.0 + lambda_away[mask10] * rho
        tau[mask01] = 1.0 + lambda_home[mask01] * rho
        tau[mask11] = 1.0 - rho
        if np.any(tau <= 0.0):
            continue
        score = float(np.sum(weight * np.log(tau)))
        if best is None or score > best[0]:
            best = (score, float(rho))
    if best is None:
        return 0.08, {"estimated": False, "warning": "Rho optimization failed; rho fixed at 0.08."}
    return best[1], {"estimated": True, "low_score_matches": low_score_matches, "search_range": [-0.20, 0.20]}


def poisson_vector(lam: float, tolerance: float = 1e-14) -> np.ndarray:
    values = [math.exp(-lam)]
    total = values[0]
    goal = 1
    while total < 1.0 - tolerance and goal < 50:
        values.append(values[-1] * lam / goal)
        total += values[-1]
        goal += 1
    return np.asarray(values, dtype=float)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def simulate(lambda_portugal: float, lambda_croatia: float, rho: float) -> tuple[dict, pd.DataFrame, np.ndarray]:
    portugal_pmf = poisson_vector(lambda_portugal)
    croatia_pmf = poisson_vector(lambda_croatia)
    maximum = max(len(portugal_pmf), len(croatia_pmf)) - 1
    portugal_pmf = np.pad(portugal_pmf, (0, maximum + 1 - len(portugal_pmf)))
    croatia_pmf = np.pad(croatia_pmf, (0, maximum + 1 - len(croatia_pmf)))
    joint = np.outer(portugal_pmf, croatia_pmf)
    tau = {
        "0-0": 1.0 - lambda_portugal * lambda_croatia * rho,
        "1-0": 1.0 + lambda_croatia * rho,
        "0-1": 1.0 + lambda_portugal * rho,
        "1-1": 1.0 - rho,
    }
    if min(tau.values()) <= 0.0:
        raise ValueError("Dixon-Coles produced a non-positive low-score multiplier")
    joint[0, 0] *= tau["0-0"]
    joint[1, 0] *= tau["1-0"]
    joint[0, 1] *= tau["0-1"]
    joint[1, 1] *= tau["1-1"]
    joint /= joint.sum()
    seed_bytes = SOURCE.read_bytes() + b"|Portugal|Croatia|2026-07-02|50000"
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = rng.choice(joint.size, size=N_SIMULATIONS, p=joint.ravel())
    portugal_goals, croatia_goals = np.unravel_index(samples, joint.shape)
    counts = np.bincount(samples, minlength=joint.size).reshape(joint.shape)
    outcome_counts = {
        "Portugal win": int(np.sum(portugal_goals > croatia_goals)),
        "Draw": int(np.sum(portugal_goals == croatia_goals)),
        "Croatia win": int(np.sum(portugal_goals < croatia_goals)),
    }
    outcomes = {}
    for label, count in outcome_counts.items():
        low, high = wilson_interval(count, N_SIMULATIONS)
        outcomes[label] = {
            "count": count,
            "probability": count / N_SIMULATIONS,
            "ci95_low": low,
            "ci95_high": high,
        }
    rows: list[dict] = []
    for portugal in range(maximum + 1):
        for croatia in range(maximum + 1):
            rows.append({
                "portugal_goals": portugal,
                "croatia_goals": croatia,
                "score": f"{portugal}-{croatia}",
                "dixon_coles_probability": float(joint[portugal, croatia]),
                "simulation_count": int(counts[portugal, croatia]),
                "simulation_probability": float(counts[portugal, croatia] / N_SIMULATIONS),
            })
    distribution = pd.DataFrame(rows).sort_values(
        ["dixon_coles_probability", "portugal_goals", "croatia_goals"],
        ascending=[False, True, True],
    )
    simulation = {
        "outcomes": outcomes,
        "simulations": N_SIMULATIONS,
        "seed": str(seed),
        "most_likely_score": str(distribution.iloc[0]["score"]),
        "most_likely_score_probability": float(distribution.iloc[0]["dixon_coles_probability"]),
        "tau": tau,
        "monte_carlo_mean_portugal": float(portugal_goals.mean()),
        "monte_carlo_mean_croatia": float(croatia_goals.mean()),
    }
    return simulation, distribution, joint


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    data, dataset = load_data()
    elo = build_elo(data)
    data, weight_audit = attach_weights(data, elo)
    baseline = global_baseline(data)
    baseline_recent = global_baseline(data, recent_only=True)

    team_rows: dict[str, pd.DataFrame] = {}
    team_audits: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for team in TARGET_TEAMS:
        team_rows[team], team_audits[team] = team_match_rows(data, team)
        summaries[team] = summarize_team(team_rows[team], baseline, team_audits[team])

    threshold, elite_counts, counts_1600, elite_limited = choose_elite_threshold(team_rows)
    elite = {team: subset_rates(team_rows[team], baseline, threshold=threshold) for team in TARGET_TEAMS}
    elite_recent = {
        team: subset_rates(team_rows[team], baseline_recent, threshold=threshold, recent_only=True)
        for team in TARGET_TEAMS
    }
    for team in TARGET_TEAMS:
        summaries[team]["elite_opponent_matches"] = elite_counts[team]
        summaries[team]["elite_opponent_matches_at_1600"] = counts_1600[team]

    observations = build_observations(data)
    regression = fit_weighted_poisson(observations)
    knockout_matches = int(data["knockout"].sum())
    knockout_warning = knockout_matches < 10
    if knockout_warning:
        regression["knockout_effect"] = 0.0

    recent_mask = np.asarray(pd.to_datetime(observations["date"]) >= PRIMARY_START)
    regression_recent = fit_poisson_coordinate(observations, recent_mask, regression["ridge"])
    if knockout_warning:
        regression_recent["knockout_effect"] = 0.0

    lambda_reg = {
        "Portugal": target_regression_lambda(regression, observations, "Portugal", "Croatia"),
        "Croatia": target_regression_lambda(regression, observations, "Croatia", "Portugal"),
    }
    lambda_reg_recent = {
        "Portugal": target_regression_lambda(regression_recent, observations, "Portugal", "Croatia"),
        "Croatia": target_regression_lambda(regression_recent, observations, "Croatia", "Portugal"),
    }
    context_factor = {
        team: target_context_factor(regression, observations, team) for team in TARGET_TEAMS
    }
    context_factor_recent = {
        team: target_context_factor(regression_recent, observations, team) for team in TARGET_TEAMS
    }

    lambda_elite = {
        "Portugal": baseline * elite["Portugal"]["adjusted_attack"] * elite["Croatia"]["adjusted_defense"] * context_factor["Portugal"],
        "Croatia": baseline * elite["Croatia"]["adjusted_attack"] * elite["Portugal"]["adjusted_defense"] * context_factor["Croatia"],
    }
    lambda_elite_recent = {
        "Portugal": baseline_recent * elite_recent["Portugal"]["adjusted_attack"] * elite_recent["Croatia"]["adjusted_defense"] * context_factor_recent["Portugal"],
        "Croatia": baseline_recent * elite_recent["Croatia"]["adjusted_attack"] * elite_recent["Portugal"]["adjusted_defense"] * context_factor_recent["Croatia"],
    }
    lambda_raw = {
        "Portugal": baseline * summaries["Portugal"]["raw_attack"] * summaries["Croatia"]["raw_defense"] * context_factor["Portugal"],
        "Croatia": baseline * summaries["Croatia"]["raw_attack"] * summaries["Portugal"]["raw_defense"] * context_factor["Croatia"],
    }

    prior_data = data[data["date"] < PRIMARY_START]
    prior_goals = np.concatenate([prior_data["home_score"].to_numpy(dtype=float), prior_data["away_score"].to_numpy(dtype=float)])
    prior_weights = np.concatenate([
        prior_data["home_obs_weight_uncapped"].to_numpy(dtype=float),
        prior_data["away_obs_weight_uncapped"].to_numpy(dtype=float),
    ])
    baseline_prior = weighted_rate(prior_goals, prior_weights)
    prior_rates = {team: historical_prior_rates(team_rows[team], baseline_prior) for team in TARGET_TEAMS}
    lambda_prior = {
        "Portugal": baseline_prior * prior_rates["Portugal"]["adjusted_attack"] * prior_rates["Croatia"]["adjusted_defense"] * context_factor["Portugal"],
        "Croatia": baseline_prior * prior_rates["Croatia"]["adjusted_attack"] * prior_rates["Portugal"]["adjusted_defense"] * context_factor["Croatia"],
    }

    regression_weight, elite_weight = (0.82, 0.13) if elite_limited else (0.70, 0.25)
    prior_weight = 0.05
    lambda_final = {
        team: regression_weight * lambda_reg[team] + elite_weight * lambda_elite[team] + prior_weight * lambda_prior[team]
        for team in TARGET_TEAMS
    }
    lambda_without_prior = {
        team: (regression_weight * lambda_reg_recent[team] + elite_weight * lambda_elite_recent[team]) / (regression_weight + elite_weight)
        for team in TARGET_TEAMS
    }
    prior_impact = {}
    for team in TARGET_TEAMS:
        absolute = lambda_final[team] - lambda_without_prior[team]
        percent = 100.0 * absolute / lambda_without_prior[team]
        prior_impact[team] = {
            "lambda_without_historical_prior": lambda_without_prior[team],
            "lambda_with_historical_prior": lambda_final[team],
            "absolute_change": absolute,
            "percentage_change": percent,
        }
    maximum_prior_change = max(abs(item["percentage_change"]) for item in prior_impact.values())
    total_historical_influence_scaler = 1.0
    if maximum_prior_change > 5.0:
        recent_share = regression_weight + elite_weight
        regression_ratio = regression_weight / recent_share
        elite_ratio = elite_weight / recent_share

        def blend_at(candidate_prior_weight: float) -> tuple[dict[str, float], float, float]:
            remaining = 1.0 - candidate_prior_weight
            candidate_regression_weight = remaining * regression_ratio
            candidate_elite_weight = remaining * elite_ratio
            candidate_lambda = {
                team: candidate_regression_weight * lambda_reg[team]
                + candidate_elite_weight * lambda_elite[team]
                + candidate_prior_weight * lambda_prior[team]
                for team in TARGET_TEAMS
            }
            return candidate_lambda, candidate_regression_weight, candidate_elite_weight

        # Find the largest explicit prior weight that keeps every target lambda
        # within the requested 5% sensitivity bound. A proportional one-shot
        # shrink is not exact because the recent components are renormalized.
        lower, upper = 0.0, prior_weight
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            candidate_lambda, _, _ = blend_at(midpoint)
            candidate_change = max(
                abs(100.0 * (candidate_lambda[team] - lambda_without_prior[team]) / lambda_without_prior[team])
                for team in TARGET_TEAMS
            )
            if candidate_change <= 5.0:
                lower = midpoint
            else:
                upper = midpoint
        effective_prior_weight = lower
        lambda_final, adjusted_regression_weight, adjusted_elite_weight = blend_at(effective_prior_weight)
        regression_weight, elite_weight, prior_weight = adjusted_regression_weight, adjusted_elite_weight, effective_prior_weight
        for team in TARGET_TEAMS:
            absolute = lambda_final[team] - lambda_without_prior[team]
            prior_impact[team] = {
                "lambda_without_historical_prior": lambda_without_prior[team],
                "lambda_with_historical_prior": lambda_final[team],
                "absolute_change": absolute,
                "percentage_change": 100.0 * absolute / lambda_without_prior[team],
            }

    # The fitted regression and elite components also contain capped historical
    # observations. If their combined residual influence still breaches 5%
    # after the explicit prior reaches zero, shrink the total historical
    # displacement toward the recent-only result.
    residual_history_change = max(abs(item["percentage_change"]) for item in prior_impact.values())
    if residual_history_change > 5.0:
        total_historical_influence_scaler = 5.0 / residual_history_change
        lambda_final = {
            team: lambda_without_prior[team]
            + total_historical_influence_scaler * (lambda_final[team] - lambda_without_prior[team])
            for team in TARGET_TEAMS
        }
        for team in TARGET_TEAMS:
            absolute = lambda_final[team] - lambda_without_prior[team]
            prior_impact[team] = {
                "lambda_without_historical_prior": lambda_without_prior[team],
                "lambda_with_historical_prior": lambda_final[team],
                "absolute_change": absolute,
                "percentage_change": 100.0 * absolute / lambda_without_prior[team],
            }

    rho, rho_audit = fit_dixon_coles(data, observations, regression)
    simulation, distribution, probability_matrix = simulate(lambda_final["Portugal"], lambda_final["Croatia"], rho)

    schedule_inflated = False
    p, c = summaries["Portugal"], summaries["Croatia"]
    if p["raw_attack"] > 1.20 * c["raw_attack"] and p["average_opponent_elo"] < c["average_opponent_elo"] - 25.0:
        schedule_inflated = True
    if c["raw_attack"] > 1.20 * p["raw_attack"] and c["average_opponent_elo"] < p["average_opponent_elo"] - 25.0:
        schedule_inflated = True
    weak_opponent_driver = any(summary["attack_correction_factor"] < 0.90 for summary in summaries.values())
    diagnostics = []
    if schedule_inflated:
        diagnostics.append("Raw attack is schedule-inflated; opponent-adjusted model is more reliable.")
    else:
        diagnostics.append("No large raw-attack ranking distortion from schedule strength was detected.")
    if weak_opponent_driver:
        diagnostics.append("Adjusted scoring is materially lower after opponent-quality weighting; weak-opponent scoring influenced the raw rates.")
    else:
        diagnostics.append("The adjusted result is not mainly driven by weak-opponent inflated scoring.")
    if any(abs(item["percentage_change"]) > 3.0 for item in prior_impact.values()):
        diagnostics.append("Historical-prior sensitivity detected; recent-data-only result should be treated as the primary signal.")

    result = {
        "model_name": "Opponent-Adjusted Recent International ELO + Low-Weight Historical Prior + Weighted Poisson Regression + Elite-Opponent Correction + Dixon-Coles + Monte Carlo",
        "fixture": "Portugal vs Croatia",
        "target_date": TARGET_DATE.strftime("%Y-%m-%d"),
        "cutoff_date": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "target_neutral": TARGET_NEUTRAL,
        "target_competition_class": TARGET_COMPETITION_CLASS,
        "target_regression_type": TARGET_REGRESSION_TYPE,
        "source": {
            "file": str(SOURCE),
            "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
            "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        },
        "dataset": dataset,
        "historical_weight_audit": weight_audit,
        "teams": summaries,
        "elite": {
            "threshold": threshold,
            "limited_warning": elite_limited,
            "counts_at_1600": counts_1600,
            "teams": elite,
            "recent_only_teams": elite_recent,
        },
        "elo": {
            "initial": elo["initial"],
            "home_advantage_elo_non_neutral_estimated": elo["home_advantage_elo_non_neutral"],
            "target_home_advantage_elo": 0.0 if TARGET_NEUTRAL else elo["home_advantage_elo_non_neutral"],
            "brier": elo["brier"],
            "Portugal_final": elo["ratings"]["Portugal"],
            "Croatia_final": elo["ratings"]["Croatia"],
            "historical_K_multiplier": 0.35,
            "K": ELO_K,
        },
        "regression": {
            "ridge": regression["ridge"],
            "ridge_validation": regression["ridge_validation"],
            "converged": regression["converged"],
            "iterations": regression["iterations"],
            "weighted_nll": regression["full_weighted_nll"],
            "intercept": regression["intercept"],
            "home_effect": regression["home_effect"],
            "knockout_effect": regression["knockout_effect"],
            "knockout_matches": knockout_matches,
            "knockout_warning": knockout_warning,
            "competition_effects": {
                name: float(regression["competition_effect"][observations["comp_index"][name]])
                for name in REGRESSION_TYPES
            },
            "Portugal_attack_effect": float(regression["attack"][observations["team_index"]["Portugal"]]),
            "Portugal_defense_effect": float(regression["defense"][observations["team_index"]["Portugal"]]),
            "Croatia_attack_effect": float(regression["attack"][observations["team_index"]["Croatia"]]),
            "Croatia_defense_effect": float(regression["defense"][observations["team_index"]["Croatia"]]),
            "attack_mean": float(regression["attack"].mean()),
            "defense_mean": float(regression["defense"].mean()),
        },
        "blend": {
            "weighted_poisson_weight": regression_weight,
            "elite_subset_weight": elite_weight,
            "historical_prior_weight": prior_weight,
            "total_historical_influence_scaler": total_historical_influence_scaler,
        },
        "prior_component": {
            "GlobalWeightedGoalsPerTeam_2018_2021": baseline_prior,
            "team_rates": prior_rates,
            "lambda": lambda_prior,
        },
        "historical_prior_impact": prior_impact,
        "parameters": {
            "GlobalWeightedGoalsPerTeam": baseline,
            "Portugal_adjusted_attack": summaries["Portugal"]["adjusted_attack"],
            "Portugal_adjusted_defense": summaries["Portugal"]["adjusted_defense"],
            "Croatia_adjusted_attack": summaries["Croatia"]["adjusted_attack"],
            "Croatia_adjusted_defense": summaries["Croatia"]["adjusted_defense"],
            "lambda_portugal_raw": lambda_raw["Portugal"],
            "lambda_croatia_raw": lambda_raw["Croatia"],
            "lambda_portugal_adjusted": lambda_reg["Portugal"],
            "lambda_croatia_adjusted": lambda_reg["Croatia"],
            "lambda_portugal_elite": lambda_elite["Portugal"],
            "lambda_croatia_elite": lambda_elite["Croatia"],
            "lambda_portugal_historical_prior": lambda_prior["Portugal"],
            "lambda_croatia_historical_prior": lambda_prior["Croatia"],
            "lambda_portugal_final": lambda_final["Portugal"],
            "lambda_croatia_final": lambda_final["Croatia"],
            "dixon_coles_rho": rho,
            "dixon_coles_rho_audit": rho_audit,
        },
        "simulation": simulation,
        "top_10_scores": distribution.head(10).to_dict(orient="records"),
        "diagnostics": diagnostics,
    }

    export_data = data.copy()
    export_data["date"] = export_data["date"].dt.strftime("%Y-%m-%d")
    target_export = pd.concat([team_rows["Portugal"], team_rows["Croatia"]], ignore_index=True)
    target_export["elite_opponent"] = target_export["opponent_pre_elo"] >= threshold
    target_export["date"] = target_export["date"].dt.strftime("%Y-%m-%d")
    export_data.to_csv(OUTPUT / "global_matches_with_elo_and_weights.csv", index=False, encoding="utf-8-sig")
    target_export.to_csv(OUTPUT / "portugal_croatia_match_diagnostics.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(OUTPUT / "full_score_distribution.csv", index=False, encoding="utf-8-sig")
    np.savetxt(OUTPUT / "probability_matrix.csv", probability_matrix, delimiter=",", fmt="%.12f")
    with (OUTPUT / "prediction.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with (OUTPUT / "run_output.txt").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
