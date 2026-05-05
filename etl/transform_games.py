"""
Game-level feature engineering for the NBA Win Predictor.

FEATURE CONTRACT
────────────────
DYNAMIC features (shift(1) in training; updated via expected probability in simulation):
    team_rolling_win_pct_5/10   — team win% over last 5/10 games
    opp_rolling_win_pct_5/10    — opponent win% over last 5/10 games

STATIC features (prior season only — season_year - 1, never current-season data):
    team_prev_net_rating        — prior season net rating
    team_prev_off_rating        — prior season offensive rating
    team_prev_def_rating        — prior season defensive rating
    opp_prev_net_rating/off/def — same for opponent

SCHEDULE features (always exact — derived from actual game dates/matchups):
    home_flag                   — 1 if team is home
    rest_days                   — days since last game (capped at 7)
    opp_rest_days               — opponent's rest days
    days_into_season            — game number within season for this team
"""

import numpy as np
import pandas as pd

# ── Feature column list ────────────────────────────────────────────────────────
# Order matters: must match exactly between training and simulate_league().

GAME_FEATURE_COLS = [
    # Dynamic: rolling form
    "team_rolling_win_pct_5",
    "team_rolling_win_pct_10",
    "opp_rolling_win_pct_5",
    "opp_rolling_win_pct_10",
    # Static: prior-season quality baseline
    "team_prev_net_rating",
    "team_prev_off_rating",
    "team_prev_def_rating",
    "opp_prev_net_rating",
    "opp_prev_off_rating",
    "opp_prev_def_rating",
    # Schedule
    "home_flag",
    "rest_days",
    "opp_rest_days",
    "days_into_season",
]

GAME_TARGET = "win"


# ── Rolling features ───────────────────────────────────────────────────────────

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling win% and rest-day features for each team.

    All rolling windows use shift(1) — the current game's outcome is never
    included in that game's feature values. min_periods=1 handles cold-start
    (first games of a season where fewer than N prior games exist).

    df must contain: team_id, game_date, season_year, win
    """
    df = df.sort_values(["team_id", "game_date"]).copy()

    for window in [5, 10]:
        df[f"team_rolling_win_pct_{window}"] = (
            df.groupby("team_id")["win"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Rest days: actual days elapsed since the team's previous game.
    # fillna(7) for the first game of the season (well-rested) and clip at 7.
    df["rest_days"] = (
        df.groupby("team_id")["game_date"]
        .transform(lambda x: x.diff().dt.days)
        .fillna(7)
        .clip(upper=7)
        .astype(int)
    )

    # Sequential game number within this team's season (1 = first game).
    df["days_into_season"] = (
        df.groupby(["team_id", "season_year"]).cumcount() + 1
    )

    return df


def add_opponent_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Self-join on game_id to attach the opponent's rolling features and rest days.

    Each game_id appears exactly twice in df (once per team). We rename the
    opponent's team_id column to opponent_id and join on that key.
    """
    opp_src = df[["game_id", "team_id",
                  "team_rolling_win_pct_5", "team_rolling_win_pct_10",
                  "rest_days"]].rename(columns={
        "team_id": "opponent_id",
        "team_rolling_win_pct_5":  "opp_rolling_win_pct_5",
        "team_rolling_win_pct_10": "opp_rolling_win_pct_10",
        "rest_days": "opp_rest_days",
    })

    df = df.merge(opp_src, on=["game_id", "opponent_id"], how="left")
    return df


# ── Prior-season quality features ─────────────────────────────────────────────

def add_prior_season_features(
    df: pd.DataFrame,
    season_ml_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join each game row with both teams' prior-season net/off/def rating.

    season_ml_df is the ml_training_features table (one row per team per season).
    We shift by one year (join_year = season_year + 1) so that a game in season
    2024 gets features from season 2023 — no current-season data is used.
    """
    prior = season_ml_df[[
        "team_id", "season_year",
        "off_rating", "def_rating", "net_rating",
    ]].copy()
    prior["join_year"] = prior["season_year"] + 1

    # Team side
    team_prior = prior.rename(columns={
        "off_rating": "team_prev_off_rating",
        "def_rating": "team_prev_def_rating",
        "net_rating": "team_prev_net_rating",
    })
    df = df.merge(
        team_prior[["team_id", "join_year",
                    "team_prev_off_rating", "team_prev_def_rating", "team_prev_net_rating"]],
        left_on=["team_id", "season_year"],
        right_on=["team_id", "join_year"],
        how="left",
    ).drop(columns=["join_year"], errors="ignore")

    # Opponent side
    opp_prior = prior.rename(columns={
        "team_id": "opponent_id",
        "off_rating": "opp_prev_off_rating",
        "def_rating": "opp_prev_def_rating",
        "net_rating": "opp_prev_net_rating",
    })
    df = df.merge(
        opp_prior[["opponent_id", "join_year",
                   "opp_prev_off_rating", "opp_prev_def_rating", "opp_prev_net_rating"]],
        left_on=["opponent_id", "season_year"],
        right_on=["opponent_id", "join_year"],
        how="left",
    ).drop(columns=["join_year"], errors="ignore")

    return df


# ── Leakage assertion ──────────────────────────────────────────────────────────

def _assert_no_leakage(df: pd.DataFrame) -> None:
    """
    Verify two leakage invariants:

    1. Game dates are monotonically increasing per (team_id, season_year) —
       ensures rolling transforms respected chronological order.

    2. Spot-check rolling_win_pct_5 at game N equals the mean of actual wins
       for games N-5 through N-1 (shift(1) correctly applied).
    """
    for (team_id, season_year), group in df.groupby(["team_id", "season_year"]):
        dates = group["game_date"].reset_index(drop=True)
        if not dates.is_monotonic_increasing:
            raise AssertionError(
                f"Non-monotonic dates for team {team_id}, season {season_year}. "
                "Sort by (team_id, game_date) before computing rolling features."
            )

    # Spot-check first team found
    team_id = df["team_id"].iloc[0]
    season_year = df[df["team_id"] == team_id]["season_year"].iloc[0]
    sample = (
        df[(df["team_id"] == team_id) & (df["season_year"] == season_year)]
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    for i in range(1, min(8, len(sample))):
        start = max(0, i - 5)
        expected_pct = sample.iloc[start:i]["win"].mean()
        got = sample.iloc[i]["team_rolling_win_pct_5"]
        if not pd.isna(got) and abs(expected_pct - got) > 1e-5:
            raise AssertionError(
                f"Leakage detected at index {i} for team {team_id}: "
                f"expected rolling_win_pct_5={expected_pct:.4f}, got {got:.4f}. "
                "Current game's outcome must not appear in its own rolling feature."
            )


# ── Top-level orchestrator ─────────────────────────────────────────────────────

def transform_games(
    raw_df: pd.DataFrame,
    season_ml_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full game-level transform pipeline:
      1. Rolling win% and rest-day features (shift(1), no leakage)
      2. Opponent rolling features (self-join on game_id)
      3. Prior-season quality baseline (join season_year - 1)
      4. Leakage assertion

    raw_df:       Output of ingest_games() — one row per team per game.
    season_ml_df: ml_training_features table — one row per team per season.

    Returns a DataFrame ready for upsert into nba_game_features.
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Ensure game_date is datetime before any date arithmetic
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    df = compute_rolling_features(raw_df)
    df = add_opponent_rolling_features(df)
    df = add_prior_season_features(df, season_ml_df)
    _assert_no_leakage(df)

    # Warn about rows with no prior-season stats (first season in dataset or new teams)
    missing_prior = df["team_prev_net_rating"].isna().sum()
    if missing_prior > 0:
        pct = 100 * missing_prior / len(df)
        print(
            f"  Note: {missing_prior} rows ({pct:.1f}%) missing prior-season stats "
            "(expected for first season per team — these rows train with NaN prior features)."
        )

    return df
