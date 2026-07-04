from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "model_engine.py"
DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the opponent-adjusted national-team scoreline model."
    )
    parser.add_argument("--fixture", help='Fixture in "Team A vs Team B" form')
    parser.add_argument("--team-a", help="First team label in the raw CSV")
    parser.add_argument("--team-b", help="Second team label in the raw CSV")
    parser.add_argument("--cutoff", help="Optional inclusive cutoff override, YYYY-MM-DD")
    parser.add_argument("--target-date", help="Optional target-date override, YYYY-MM-DD")
    parser.add_argument("--source", type=Path, help="Existing raw results CSV")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="Raw-results CSV URL")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--simulations", type=int, default=50_000)
    return parser.parse_args()


def validate_name(value: str) -> str:
    value = value.strip()
    if not value or any(char in value for char in ['"', "\\", "\r", "\n"]):
        raise ValueError(f"Unsafe or empty team label: {value!r}")
    return value


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        slug = "team"
    if slug[0].isdigit():
        slug = "team_" + slug
    return slug


def requested_teams(args: argparse.Namespace) -> tuple[str, str]:
    if args.fixture:
        parts = re.split(r"\s+(?:vs\.?|v\.?)\s+", args.fixture.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError('Fixture must use "Team A vs Team B" form')
        fixture_a, fixture_b = (validate_name(value) for value in parts)
        if args.team_a and validate_name(args.team_a) != fixture_a:
            raise ValueError("--fixture and --team-a disagree")
        if args.team_b and validate_name(args.team_b) != fixture_b:
            raise ValueError("--fixture and --team-b disagree")
        return fixture_a, fixture_b
    if not args.team_a or not args.team_b:
        raise ValueError("Give --fixture, or give both --team-a and --team-b")
    return validate_name(args.team_a), validate_name(args.team_b)


def prepare_source(args: argparse.Namespace, output: Path) -> tuple[Path, str]:
    if args.source:
        source = args.source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        return source, f"file:{source}"
    source = output / "raw_results.csv"
    request = urllib.request.Request(
        args.source_url,
        headers={"User-Agent": "model-football-scorelines/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        source.write_bytes(response.read())
    return source, args.source_url


def validate_configuration(
    args: argparse.Namespace,
    source: Path,
    team_a: str,
    team_b: str,
) -> tuple[str, str, pd.Timestamp, pd.Timestamp]:
    if team_a == team_b:
        raise ValueError("The two teams must be different")
    header = pd.read_csv(source, nrows=0)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Raw CSV missing columns: {sorted(missing)}")
    frame = pd.read_csv(source)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    available = set(frame["home_team"].dropna()) | set(frame["away_team"].dropna())
    absent = [team for team in (team_a, team_b) if team not in available]
    if absent:
        raise ValueError(f"Team labels absent from source: {absent}")
    matchup = (
        (frame["home_team"].eq(team_a) & frame["away_team"].eq(team_b))
        | (frame["home_team"].eq(team_b) & frame["away_team"].eq(team_a))
    )
    scores_missing = frame[["home_score", "away_score"]].isna().any(axis=1)
    candidates = frame[matchup & scores_missing & frame["date"].notna()].sort_values("date")
    if args.target_date:
        target_date = pd.Timestamp(args.target_date)
        target_rows = frame[matchup & frame["date"].eq(target_date)]
    else:
        completed = frame[frame[["home_score", "away_score"]].notna().all(axis=1)]
        latest_completed = completed["date"].max()
        candidates = candidates[candidates["date"] > latest_completed]
        if candidates.empty:
            raise ValueError("No future unscored fixture row found; supply --target-date")
        target_date = pd.Timestamp(candidates.iloc[0]["date"])
        target_rows = candidates[candidates["date"].eq(target_date)]
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else target_date - pd.Timedelta(days=1)
    if cutoff.year < 2022 or cutoff.year > 2026:
        raise ValueError("This model version supports cutoff years 2022 through 2026")
    if target_date <= cutoff:
        raise ValueError("Target date must be after the cutoff")
    if not target_rows.empty:
        target_row = target_rows.iloc[0]
        neutral = str(target_row["neutral"]).strip().lower() in {"true", "1", "yes"}
        if not neutral:
            raise ValueError("This skill version requires a neutral target fixture")
        if str(target_row["tournament"]) != "FIFA World Cup":
            raise ValueError("This skill version requires a FIFA World Cup target fixture")
    if not (target_date.year == 2026 and target_date >= pd.Timestamp("2026-07-01")):
        raise ValueError("This skill version requires a 2026 World Cup knockout fixture")
    return team_a, team_b, cutoff, target_date


def build_namespace(
    team_a: str,
    team_b: str,
    cutoff: pd.Timestamp,
    target_date: pd.Timestamp,
    source_path: Path,
    output: Path,
    simulations: int,
) -> dict:
    source = ENGINE.read_text(encoding="utf-8")
    old_seed = 'seed_bytes = SOURCE.read_bytes() + b"|Portugal|Croatia|2026-07-02|50000"'
    new_seed = (
        'seed_bytes = SOURCE.read_bytes() + '
        'f"|{TARGET_TEAMS[0]}|{TARGET_TEAMS[1]}|{TARGET_DATE.date()}|{N_SIMULATIONS}".encode("utf-8")'
    )
    if old_seed not in source:
        raise RuntimeError("Bundled engine seed contract changed")
    source = source.replace(old_seed, new_seed)
    source = source.replace("Portugal", team_a).replace("Croatia", team_b)
    source = source.replace("portugal", slugify(team_a)).replace("croatia", slugify(team_b))
    source = source.replace("2026-06-30", cutoff.strftime("%Y-%m-%d"))
    source = source.replace("2026-07-02", target_date.strftime("%Y-%m-%d"))
    namespace = {
        "__file__": str(ENGINE),
        "__name__": "model_football_scorelines_engine",
    }
    exec(compile(source, str(ENGINE), "exec"), namespace)
    namespace["SOURCE"] = source_path
    namespace["OUTPUT"] = output
    namespace["N_SIMULATIONS"] = simulations

    major_tournaments = {
        "AFC Asian Cup", "African Cup of Nations", "Copa América", "Gold Cup",
        "Oceania Nations Cup", "UEFA Euro",
    }
    nations_leagues = {"CONCACAF Nations League", "UEFA Nations League"}
    major_qualifiers = {
        "AFC Asian Cup qualification", "African Cup of Nations qualification",
        "Copa América qualification", "Gold Cup qualification",
        "Oceania Nations Cup qualification", "UEFA Euro qualification",
    }
    other_tournaments = {
        "AFF Championship", "ASEAN Championship", "Arab Cup", "CAFA Nations Cup",
        "CONCACAF Series", "COSAFA Cup", "EAFF Championship", "Gulf Cup",
        "Pacific Games", "SAFF Cup", "WAFF Championship",
        "CONMEBOL–UEFA Cup of Champions",
    }
    other_qualifiers = {
        "AFF Championship qualification", "ASEAN Championship qualification",
        "Arab Cup qualification", "CONCACAF Nations League qualification",
        "EAFF Championship qualification",
    }

    def classify_competition(row: pd.Series) -> str:
        tournament = str(row["tournament"])
        if tournament == "FIFA World Cup":
            return "world_cup_knockout" if namespace["is_world_cup_knockout"](row["date"]) else "world_cup_group"
        if tournament == "FIFA World Cup qualification":
            return "world_cup_qualifier"
        if tournament in nations_leagues:
            return "continental_nations_league"
        if tournament in major_tournaments:
            return "continental_tournament"
        if tournament in major_qualifiers:
            return "continental_qualifier"
        if tournament in other_tournaments:
            return "other_continental_tournament"
        if tournament in other_qualifiers:
            return "other_continental_qualifier"
        if tournament == "Friendly":
            return "friendly"
        return "other"

    namespace["classify_competition"] = classify_competition
    namespace["COMPETITION_WEIGHTS"] = {
        "world_cup_knockout": 5.0, "world_cup_group": 3.0,
        "world_cup_qualifier": 2.0, "continental_tournament": 2.4,
        "continental_nations_league": 1.5, "continental_qualifier": 1.6,
        "other_continental_tournament": 1.8, "other_continental_qualifier": 1.4,
        "friendly": 0.6, "other": 1.0,
    }
    namespace["ELO_K"] = {
        "world_cup_knockout": 45.0, "world_cup_group": 35.0,
        "world_cup_qualifier": 25.0, "continental_tournament": 28.0,
        "continental_nations_league": 18.0, "continental_qualifier": 20.0,
        "other_continental_tournament": 25.0, "other_continental_qualifier": 18.0,
        "friendly": 10.0, "other": 15.0,
    }
    namespace["REGRESSION_TYPES"] = [
        "friendly", "other", "other_continental_qualifier",
        "other_continental_tournament", "continental_qualifier",
        "continental_nations_league", "continental_tournament",
        "world_cup_qualifier", "world_cup",
    ]
    return namespace


def audit(output: Path, result: dict, team_a: str, team_b: str, cutoff: pd.Timestamp) -> dict:
    matrix = np.loadtxt(output / "probability_matrix.csv", delimiter=",")
    max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    training = pd.read_csv(output / "global_matches_with_elo_and_weights.csv")
    dates = pd.to_datetime(training["date"])
    target_rows = (
        (training["home_team"].eq(team_a) & training["away_team"].eq(team_b))
        | (training["home_team"].eq(team_b) & training["away_team"].eq(team_a))
    )
    history_values = [
        result["historical_weight_audit"]["global_effective_historical_weight_pct"],
        result["teams"][team_a]["effective_historical_weight_pct"],
        result["teams"][team_b]["effective_historical_weight_pct"],
    ]
    checks = {
        "matrix_sum": abs(float(matrix.sum()) - 1.0) < 1e-9,
        "outcome_sum": abs(sum(v["probability"] for v in result["simulation"]["outcomes"].values()) - 1.0) < 1e-12,
        "matrix_argmax_matches_report": f"{max_index[0]}-{max_index[1]}" == result["simulation"]["most_likely_score"],
        "no_post_cutoff_training_rows": bool((dates <= cutoff).all()),
        "target_fixture_not_in_training": int(target_rows.sum()) == 0,
        "historical_effective_weights_at_most_8pct": max(history_values) <= 8.0 + 1e-10,
        "historical_lambda_blend_at_most_5pct": result["blend"]["historical_prior_weight"] <= 0.05 + 1e-12,
        "historical_lambda_change_at_most_5pct": max(abs(v["percentage_change"]) for v in result["historical_prior_impact"].values()) <= 5.0 + 1e-10,
        "regression_converged": bool(result["regression"]["converged"]),
        "attack_effects_centered": abs(result["regression"]["attack_mean"]) < 1e-10,
        "defense_effects_centered": abs(result["regression"]["defense_mean"]) < 1e-10,
        "dixon_coles_multipliers_positive": min(result["simulation"]["tau"].values()) > 0.0,
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "matrix_argmax": f"{max_index[0]}-{max_index[1]}",
        "source_sha256": result["source"]["sha256"],
    }


def main() -> None:
    args = parse_args()
    if args.simulations < 50_000:
        raise ValueError("The model requires at least 50,000 simulations")
    tentative_a, tentative_b = requested_teams(args)
    output = (args.output_dir or Path.cwd() / f"{slugify(tentative_a)}_{slugify(tentative_b)}_outputs").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path, source_identity = prepare_source(args, output)
    team_a, team_b, cutoff, target_date = validate_configuration(
        args, source_path, tentative_a, tentative_b
    )
    namespace = build_namespace(team_a, team_b, cutoff, target_date, source_path, output, args.simulations)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        namespace["main"]()
    result_path = output / "prediction.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["source"]["url"] = source_identity
    result["source"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    result["run_config"] = {
        "team_a": team_a,
        "team_b": team_b,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "neutral": True,
        "target_context": "FIFA World Cup knockout",
        "simulations": args.simulations,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run_output.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_result = audit(output, result, team_a, team_b, cutoff)
    (output / "hard_audit.json").write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not audit_result["all_passed"]:
        failed = [name for name, passed in audit_result["checks"].items() if not passed]
        raise RuntimeError(f"Hard audit failed: {failed}")
    summary = {
        "fixture": result["fixture"],
        "lambda": {
            team_a: result["parameters"][f"lambda_{slugify(team_a)}_final"],
            team_b: result["parameters"][f"lambda_{slugify(team_b)}_final"],
        },
        "outcomes": result["simulation"]["outcomes"],
        "most_likely_score": result["simulation"]["most_likely_score"],
        "audit": "PASS",
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
