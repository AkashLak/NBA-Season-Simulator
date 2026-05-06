"""
League-wide season simulation engine.

simulate_league() is the core function. It processes the full NBA schedule
chronologically (~1,230 games), maintaining rolling win% state for all 30 teams
simultaneously. After each game, both teams' win_history is updated using the
predicted win probability — no random sampling, pure expected value.

WHY full-league simulation is required
────────────────────────────────────────
The game model uses rolling_win_pct_10 — a feature that changes after every game.
Team B's rolling stats at game 50 depend on their results in games 1-49, including
games against teams other than the team you're predicting. If you simulated only
one team, Team B would only "exist" in the 2 games against your team; their other
80 games are missing, making their rolling stats wrong.

Simulating the full league solves this: every team plays its full schedule, so
every team's rolling stats are correctly maintained regardless of who's asking.

KNOWN DISTRIBUTION SHIFT
─────────────────────────
Training: rolling_win_pct computed from actual binary outcomes (0 or 1).
Simulation: updated with win_prob ∈ (0, 1) — a continuous expected value.
This makes simulated rolling stats "smoother" than training data. The effect is
most pronounced in the first ~10 games (cold-start) and diminishes mid-season.
This is an accepted tradeoff for avoiding Monte Carlo sampling.
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from etl.transform_games import GAME_FEATURE_COLS

load_dotenv()

PROCESSED_DIR = "processed"
LEAGUE_AVG_WIN_PCT = 0.5
LEAGUE_AVG_NET_RATING = 0.0
LEAGUE_AVG_OFF_RATING = 113.0
LEAGUE_AVG_DEF_RATING = 113.0


# ── Schedule loading ──────────────────────────────────────────────────────────

def _load_schedule_from_db(season_year: int, engine) -> pd.DataFrame:
    """One row per game (home team perspective) from nba_game_features."""
    from sqlalchemy import text
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT DISTINCT game_id, game_date, "
                "team_id AS home_id, opponent_id AS away_id "
                "FROM nba_game_features "
                "WHERE season_year = :sy AND home_flag = 1 "
                "ORDER BY game_date"
            ),
            conn,
            params={"sy": season_year},
        )


def _load_schedule_from_parquet(season_year: int) -> pd.DataFrame:
    """Parquet fallback for Streamlit Cloud (no DATABASE_URL)."""
    path = os.path.join(PROCESSED_DIR, "game_features.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["game_id", "game_date", "season_year",
                                         "team_id", "opponent_id", "home_flag"])
    df = df[(df["season_year"] == season_year) & (df["home_flag"] == 1)].copy()
    df = df.rename(columns={"team_id": "home_id", "opponent_id": "away_id"})
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df[["game_id", "game_date", "home_id", "away_id"]].drop_duplicates("game_id")


def _get_schedule(season_year: int, engine=None, _depth: int = 0) -> pd.DataFrame:
    """
    Load one-row-per-game schedule for a season.
    Falls back to the prior year's schedule if this season is not yet in the DB
    (e.g. simulating a future season). Recursion guard at depth 3.
    """
    if _depth >= 3 or season_year < 2000:
        raise ValueError(
            f"No schedule data available for season_year={season_year} or any "
            "of the 3 prior seasons. Run the ETL pipeline first."
        )

    df = pd.DataFrame()
    if engine is not None:
        try:
            df = _load_schedule_from_db(season_year, engine)
        except Exception:
            pass

    if df.empty:
        df = _load_schedule_from_parquet(season_year)

    if df.empty:
        return _get_schedule(season_year - 1, engine, _depth + 1)

    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values("game_date").reset_index(drop=True)


# ── Prior-season feature loading ──────────────────────────────────────────────

def _load_ml_features(engine=None) -> pd.DataFrame:
    """Load ml_training_features — PostgreSQL first, Parquet fallback."""
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT team_id, season_year, off_rating, def_rating, net_rating "
                "FROM ml_training_features ORDER BY team_id, season_year",
                engine,
            )
        except Exception:
            pass
    path = os.path.join(PROCESSED_DIR, "ml_features.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        cols = [c for c in ["team_id", "season_year", "off_rating", "def_rating", "net_rating"]
                if c in df.columns]
        return df[cols]
    return pd.DataFrame()


def _build_prior_features_lookup(season_year: int, engine=None) -> dict:
    """
    Return a dict {team_id: {net_rating, off_rating, def_rating, win_pct}}
    using season_year - 1 stats as the prior-season baseline.

    Teams with no prior-season data (first season in dataset or expansion teams)
    fall back to league averages.
    """
    ml_df = _load_ml_features(engine)
    if ml_df.empty:
        return {}

    prior = ml_df[ml_df["season_year"] == season_year - 1].copy()

    lookup = {}
    for _, row in prior.iterrows():
        lookup[int(row["team_id"])] = {
            "net_rating": float(row.get("net_rating") or LEAGUE_AVG_NET_RATING),
            "off_rating": float(row.get("off_rating") or LEAGUE_AVG_OFF_RATING),
            "def_rating": float(row.get("def_rating") or LEAGUE_AVG_DEF_RATING),
        }
    return lookup


def _get_team_prior(team_id: int, lookup: dict) -> dict:
    """Return prior-season features for a team, defaulting to league averages."""
    return lookup.get(team_id, {
        "net_rating": LEAGUE_AVG_NET_RATING,
        "off_rating": LEAGUE_AVG_OFF_RATING,
        "def_rating": LEAGUE_AVG_DEF_RATING,
    })


# ── Simulation helpers ────────────────────────────────────────────────────────

def _rolling_pct(win_history: list, n: int) -> float:
    """
    Rolling win% from the last n entries of win_history.
    Returns 0.5 (league average) when history is empty — a neutral prior.
    """
    recent = win_history[-n:] if len(win_history) >= n else win_history
    return float(np.mean(recent)) if recent else LEAGUE_AVG_WIN_PCT


def _rest_days(last_game_date, current_game_date) -> int:
    """Days since the team's previous game, capped at 7. Returns 7 for season opener."""
    if last_game_date is None:
        return 7
    delta = (pd.Timestamp(current_game_date) - pd.Timestamp(last_game_date)).days
    return min(int(delta), 7)


def _build_feature_row(home_state: dict, away_state: dict, game_date) -> dict:
    """
    Build a single feature row for a game from current team states.
    Column order matches GAME_FEATURE_COLS exactly.
    """
    row = {
        "team_rolling_win_pct_5":   _rolling_pct(home_state["win_history"], 5),
        "team_rolling_win_pct_10":  _rolling_pct(home_state["win_history"], 10),
        "opp_rolling_win_pct_5":    _rolling_pct(away_state["win_history"], 5),
        "opp_rolling_win_pct_10":   _rolling_pct(away_state["win_history"], 10),
        "team_prev_net_rating":     home_state["prev_net_rating"],
        "team_prev_off_rating":     home_state["prev_off_rating"],
        "team_prev_def_rating":     home_state["prev_def_rating"],
        "opp_prev_net_rating":      away_state["prev_net_rating"],
        "opp_prev_off_rating":      away_state["prev_off_rating"],
        "opp_prev_def_rating":      away_state["prev_def_rating"],
        "home_flag":                1,          # always home team's perspective
        "rest_days":                _rest_days(home_state["last_game_date"], game_date),
        "opp_rest_days":            _rest_days(away_state["last_game_date"], game_date),
        "days_into_season":         home_state["games_played"] + 1,
    }
    # Guard: every key in the row must match GAME_FEATURE_COLS
    assert set(row.keys()) == set(GAME_FEATURE_COLS), (
        f"Feature mismatch in simulation. "
        f"Extra: {set(row.keys()) - set(GAME_FEATURE_COLS)}  "
        f"Missing: {set(GAME_FEATURE_COLS) - set(row.keys())}"
    )
    return row


def _init_team_state(team_id: int, prior: dict) -> dict:
    return {
        "win_history":    [],
        "last_game_date": None,
        "games_played":   0,
        "expected_wins":  0.0,
        "game_probs":     [],
        "prev_net_rating": prior.get("net_rating", LEAGUE_AVG_NET_RATING),
        "prev_off_rating": prior.get("off_rating", LEAGUE_AVG_OFF_RATING),
        "prev_def_rating": prior.get("def_rating", LEAGUE_AVG_DEF_RATING),
    }


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_league(
    season_year: int,
    game_model,
    engine=None,
    team_overrides: dict = None,
) -> dict:
    """
    Simulate the full NBA season for all 30 teams simultaneously.

    Processes all ~1,230 games in chronological order. After each game:
      - home team's win_history appended with win_prob
      - away team's win_history appended with (1 - win_prob)
    Rolling features for the next game are computed from these histories.

    Args:
        season_year:     Season to simulate (e.g. 2025 for the 2024-25 season).
        game_model:      Fitted classifier with predict_proba().
        engine:          SQLAlchemy engine (None → Parquet fallback).
        team_overrides:  {team_id: {"prev_net_rating": float, ...}} for roster sim.

    Returns:
        {team_id: {"expected_wins": float, "game_probs": [float, ...]}}

    Total expected wins across all 30 teams ≈ 1,230 (each game contributes
    exactly 1.0: win_prob + (1 - win_prob) = 1.0).
    """
    schedule = _get_schedule(season_year, engine)
    prior_lookup = _build_prior_features_lookup(season_year, engine)

    all_team_ids = set(schedule["home_id"]) | set(schedule["away_id"])
    team_state = {}
    for tid in all_team_ids:
        prior = _get_team_prior(tid, prior_lookup)
        if team_overrides and tid in team_overrides:
            prior = {**prior, **team_overrides[tid]}
        team_state[tid] = _init_team_state(tid, prior)

    for _, game in schedule.iterrows():
        home_id  = int(game["home_id"])
        away_id  = int(game["away_id"])
        gdate    = game["game_date"]
        hs       = team_state[home_id]
        aws      = team_state[away_id]

        feature_row = _build_feature_row(hs, aws, gdate)
        X = pd.DataFrame([feature_row])[GAME_FEATURE_COLS]   # enforce column order

        win_prob = float(np.clip(game_model.predict_proba(X)[0][1], 0.0, 1.0))

        # Update home team
        hs["win_history"].append(win_prob)
        hs["game_probs"].append(win_prob)
        hs["expected_wins"] += win_prob
        hs["games_played"]  += 1
        hs["last_game_date"] = gdate

        # Update away team (away win prob = 1 - home win prob)
        aws["win_history"].append(1.0 - win_prob)
        aws["game_probs"].append(1.0 - win_prob)
        aws["expected_wins"] += (1.0 - win_prob)
        aws["games_played"]  += 1
        aws["last_game_date"] = gdate

    return {
        tid: {
            "expected_wins": round(state["expected_wins"], 1),
            "game_probs":    state["game_probs"],
        }
        for tid, state in team_state.items()
    }


def simulate_team(
    team_id: int,
    season_year: int,
    game_model,
    engine=None,
    team_quality_override: dict = None,
) -> dict:
    """
    Convenience wrapper: run full league simulation, return one team's results.

    Internally calls simulate_league() — the full schedule is always simulated
    so that every opponent's rolling stats are correctly maintained. The override
    applies only to the requested team's prior-season quality.

    Used by:
      - Page 1 (Season Forecast): season_year+1 prediction, no override
      - Page 2 (Roster Simulator): season_year+1 prediction, override with
        adjusted net_rating from player swap

    Returns: {"expected_wins": float, "game_probs": [float, ...]}
    """
    overrides = {team_id: team_quality_override} if team_quality_override else None
    results = simulate_league(season_year, game_model, engine, team_overrides=overrides)

    if team_id not in results:
        raise ValueError(
            f"team_id {team_id} not found in simulation results for season {season_year}. "
            "Check that the team has games in the schedule for this season."
        )
    return results[team_id]
