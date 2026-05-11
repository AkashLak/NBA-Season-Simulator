"""Tests for etl/transform_games.py — game-level feature engineering."""

import pandas as pd
import pytest

from etl.transform_games import (
    GAME_FEATURE_COLS,
    _assert_no_leakage,
    add_opponent_rolling_features,
    add_prior_season_features,
    compute_rolling_features,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_game_logs(wins=None, n=10, team_id=1, opp_id=2):
    """One team's game log with configurable win sequence."""
    if wins is None:
        wins = [int(i % 2 == 0) for i in range(n)]
    dates = pd.date_range("2023-10-01", periods=len(wins), freq="2D")
    return pd.DataFrame(
        {
            "game_id": [f"G{i:03d}" for i in range(len(wins))],
            "game_date": dates,
            "season_year": [2024] * len(wins),
            "team_id": [team_id] * len(wins),
            "opponent_id": [opp_id] * len(wins),
            "home_flag": [1] * len(wins),
            "win": wins,
            "pts": [110.0] * len(wins),
        }
    )


def _make_two_team_logs():
    """Both teams sharing the same GAME_IDs (mirrors real nba_api output)."""
    n = 10
    dates = pd.date_range("2023-10-01", periods=n, freq="2D")
    game_ids = [f"G{i:03d}" for i in range(n)]

    team1 = pd.DataFrame(
        {
            "game_id": game_ids,
            "game_date": dates,
            "season_year": [2024] * n,
            "team_id": [1] * n,
            "opponent_id": [2] * n,
            "home_flag": [1] * n,
            "win": [int(i % 2 == 0) for i in range(n)],
            "pts": [112.0] * n,
        }
    )
    team2 = pd.DataFrame(
        {
            "game_id": game_ids,
            "game_date": dates,
            "season_year": [2024] * n,
            "team_id": [2] * n,
            "opponent_id": [1] * n,
            "home_flag": [0] * n,
            "win": [int(i % 2 != 0) for i in range(n)],
            "pts": [108.0] * n,
        }
    )
    return pd.concat([team1, team2], ignore_index=True)


# ── compute_rolling_features ───────────────────────────────────────────────────


def test_rolling_uses_shift1():
    """
    CRITICAL leakage check: rolling_win_pct_5 at game N must equal the mean of
    wins[N-5 : N], NOT including game N itself.
    """
    wins = [1, 0, 1, 1, 0, 1, 0, 1]
    df = _make_game_logs(wins=wins)
    result = compute_rolling_features(df).reset_index(drop=True)

    # Game index 5 (6th game): lookback = wins[0:5] = [1,0,1,1,0] → mean = 0.6
    expected = sum(wins[:5]) / 5
    got = result.iloc[5]["team_rolling_win_pct_5"]
    assert abs(expected - got) < 1e-6, (
        f"rolling_win_pct_5 at game 5: expected {expected:.4f}, got {got:.4f}. "
        "Current game's outcome must not appear in its own rolling feature."
    )

    # Game index 6: lookback = wins[1:6] = [0,1,1,0,1] → mean = 0.6
    expected6 = sum(wins[1:6]) / 5
    got6 = result.iloc[6]["team_rolling_win_pct_5"]
    assert (
        abs(expected6 - got6) < 1e-6
    ), f"rolling_win_pct_5 at game 6: expected {expected6:.4f}, got {got6:.4f}"


def test_rolling_win_pct_10_uses_shift1():
    wins = list(range(12))  # dummy non-repeating to make it obvious
    wins_bin = [int(w % 2 == 0) for w in wins]
    df = _make_game_logs(wins=wins_bin)
    result = compute_rolling_features(df).reset_index(drop=True)

    expected = sum(wins_bin[1:11]) / 10  # games 1-10 for game at index 11
    got = result.iloc[11]["team_rolling_win_pct_10"]
    assert abs(expected - got) < 1e-6


def test_rest_days_computed_from_schedule():
    """Rest days must come from actual game_date differences, never hardcoded."""
    df = _make_game_logs(n=5)
    # Override dates: Jan 1, Jan 3, Jan 4 (gaps of 2, 1 days)
    df["game_date"] = pd.to_datetime(
        ["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-07", "2024-01-09"]
    )
    result = compute_rolling_features(df).reset_index(drop=True)

    assert result.iloc[0]["rest_days"] == 7  # first game → well-rested default
    assert result.iloc[1]["rest_days"] == 2  # Jan 1 → Jan 3
    assert result.iloc[2]["rest_days"] == 1  # Jan 3 → Jan 4
    assert result.iloc[3]["rest_days"] == 3  # Jan 4 → Jan 7
    assert result.iloc[4]["rest_days"] == 2  # Jan 7 → Jan 9


def test_rest_days_capped_at_7():
    df = _make_game_logs(n=3)
    df["game_date"] = pd.to_datetime(["2024-01-01", "2024-01-20", "2024-01-25"])
    result = compute_rolling_features(df).reset_index(drop=True)
    assert result.iloc[1]["rest_days"] == 7  # 19 days capped at 7


def test_days_into_season_increments():
    df = _make_game_logs(n=5)
    result = compute_rolling_features(df).reset_index(drop=True)
    assert list(result["days_into_season"]) == [1, 2, 3, 4, 5]


# ── add_opponent_rolling_features ──────────────────────────────────────────────


def test_opponent_rolling_correctly_joined():
    """Each row's opp_rolling_win_pct_10 must equal the opponent's own value."""
    df = _make_two_team_logs()
    df = compute_rolling_features(df)
    df = add_opponent_rolling_features(df)

    for game_id in df["game_id"].unique():
        game_rows = df[df["game_id"] == game_id]
        if len(game_rows) < 2:
            continue
        team1_row = game_rows[game_rows["team_id"] == 1].iloc[0]
        team2_row = game_rows[game_rows["team_id"] == 2].iloc[0]

        # Team 1's opp_rolling = Team 2's team_rolling
        if pd.notna(team1_row["opp_rolling_win_pct_10"]) and pd.notna(
            team2_row["team_rolling_win_pct_10"]
        ):
            assert (
                abs(
                    team1_row["opp_rolling_win_pct_10"]
                    - team2_row["team_rolling_win_pct_10"]
                )
                < 1e-9
            ), "Opponent rolling features not correctly mirrored"


def test_opponent_rest_days_correctly_joined():
    df = _make_two_team_logs()
    df = compute_rolling_features(df)
    df = add_opponent_rolling_features(df)

    # Both teams play same games → each team's rest_days should appear as
    # the other's opp_rest_days
    for game_id in df["game_id"].unique():
        rows = df[df["game_id"] == game_id]
        if len(rows) < 2:
            continue
        t1 = rows[rows["team_id"] == 1].iloc[0]
        t2 = rows[rows["team_id"] == 2].iloc[0]
        assert t1["opp_rest_days"] == t2["rest_days"]
        assert t2["opp_rest_days"] == t1["rest_days"]


# ── add_prior_season_features ──────────────────────────────────────────────────


def test_prior_season_features_joined_from_season_minus_1():
    """Prior-season stats must come from season_year - 1, never the current season."""
    game_df = _make_game_logs(n=3)
    game_df = compute_rolling_features(game_df)
    game_df["opponent_id"] = 2

    # ml_df has rows for 2023 (the prior season for 2024 games)
    ml_df = pd.DataFrame(
        {
            "team_id": [1, 2],
            "season_year": [2023, 2023],
            "off_rating": [115.0, 112.0],
            "def_rating": [110.0, 113.0],
            "net_rating": [5.0, -1.0],
        }
    )

    result = add_prior_season_features(game_df, ml_df)
    assert "team_prev_net_rating" in result.columns
    assert "opp_prev_net_rating" in result.columns
    # Team 1's prior net_rating = 5.0 from season 2023
    assert abs(result["team_prev_net_rating"].iloc[0] - 5.0) < 1e-9
    # Opponent (team 2) prior net_rating = -1.0
    assert abs(result["opp_prev_net_rating"].iloc[0] - (-1.0)) < 1e-9


# ── _assert_no_leakage ─────────────────────────────────────────────────────────


def test_assert_no_leakage_passes_clean_data():
    df = _make_two_team_logs()
    df = compute_rolling_features(df)
    _assert_no_leakage(df)  # must not raise


def test_assert_no_leakage_catches_non_monotonic_dates():
    df = _make_game_logs(n=5)
    result = compute_rolling_features(df)
    # Scramble dates for one team
    result = result.sort_values("game_date", ascending=False)
    with pytest.raises(AssertionError, match="Non-monotonic"):
        _assert_no_leakage(result)


# ── GAME_FEATURE_COLS contract ─────────────────────────────────────────────────


def test_no_rolling_in_feature_col_names():
    """Dynamic rolling features are allowed (we use them), but this test verifies
    they are in GAME_FEATURE_COLS, not accidentally excluded."""
    rolling_cols = [c for c in GAME_FEATURE_COLS if "rolling" in c]
    assert (
        len(rolling_cols) == 4
    ), f"Expected 4 rolling cols in GAME_FEATURE_COLS, got {len(rolling_cols)}: {rolling_cols}"


def test_game_feature_cols_count():
    assert (
        len(GAME_FEATURE_COLS) == 14
    ), f"GAME_FEATURE_COLS should have 14 features, has {len(GAME_FEATURE_COLS)}"
