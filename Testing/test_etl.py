"""Tests for etl/transform.py — season-level ETL pipeline."""

import pandas as pd
import pytest

from etl.transform import (
    add_lag_features,
    aggregate_player_features,
    build_feature_row,
    normalize_team_stats,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_player_df():
    """Minimal player DataFrame for aggregation tests."""
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1, 2, 2, 2],
            "season_year": [2023] * 4 + [2023] * 3,
            "player_id": [10, 11, 12, 13, 20, 21, 22],
            "games_played": [60, 55, 50, 45, 60, 55, 50],
            "minutes_pg": [35.0, 28.0, 22.0, 15.0, 32.0, 30.0, 18.0],
            "pie": [0.15, 0.12, 0.10, 0.09, 0.14, 0.13, 0.11],
            "age": [28, 25, 30, 24, 27, 29, 26],
        }
    )


def _make_two_season_team_df():
    """Two teams × two seasons for lag feature tests."""
    return pd.DataFrame(
        {
            "team_id": [1, 1, 2, 2],
            "season_year": [2022, 2023, 2022, 2023],
            "wins": [50, 40, 30, 35],
            "wins_normalized": [50.0, 40.0, 30.0, 35.0],
            "off_rating": [110.0, 108.0, 105.0, 107.0],
            "def_rating": [108.0, 109.0, 111.0, 110.0],
            "net_rating": [2.0, -1.0, -6.0, -3.0],
            "pts_pg": [115.0, 112.0, 108.0, 110.0],
            "ts_pct": [0.58, 0.57, 0.55, 0.56],
            "ast_to_ratio": [2.0, 1.9, 1.8, 1.85],
            "pace": [100.0, 101.0, 99.0, 100.5],
            "oreb_pct": [0.25, 0.26, 0.24, 0.25],
            "dreb_pct": [0.75, 0.74, 0.76, 0.75],
            "tm_tov_pct": [0.12, 0.13, 0.14, 0.13],
            "team_avg_pie": [0.10, 0.11, 0.09, 0.10],
            "playoff_team": [True, False, False, False],
            "std_dev_pie": [0.02, 0.03, 0.01, 0.02],
            "top_3_minutes_share": [0.72, 0.75, 0.70, 0.73],
        }
    )


# ── normalize_team_stats ───────────────────────────────────────────────────────


def test_normalize_team_stats_renames_columns():
    df = pd.DataFrame(
        {
            "TEAM_ID": [1],
            "GP": [82],
            "W": [50],
            "AST": [25.0],
            "TOV": [12.5],
        }
    )
    result = normalize_team_stats(df)
    assert "team_id" in result.columns
    assert "games_played" in result.columns
    assert "wins" in result.columns
    assert "ast_to_ratio" in result.columns
    assert abs(result["ast_to_ratio"].iloc[0] - 2.0) < 1e-6


def test_normalize_team_stats_handles_missing_columns():
    df = pd.DataFrame({"TEAM_ID": [1], "W": [40]})
    result = normalize_team_stats(df)
    assert "team_id" in result.columns
    assert "ast_to_ratio" not in result.columns  # can't compute without AST and TOV


# ── aggregate_player_features ──────────────────────────────────────────────────


def test_aggregate_player_features_produces_new_columns():
    player_df = _make_player_df()
    result = aggregate_player_features(player_df)
    assert "std_dev_pie" in result.columns
    assert "top_3_minutes_share" in result.columns
    assert "team_avg_pie" in result.columns
    assert "team_avg_age" in result.columns


def test_top_3_minutes_share_correct():
    player_df = _make_player_df()
    result = aggregate_player_features(player_df)
    t1 = result[result["team_id"] == 1].iloc[0]
    # top-3 minutes for team 1: 35+28+22=85, total=35+28+22+15=100
    assert abs(t1["top_3_minutes_share"] - 0.85) < 1e-6


def test_std_dev_pie_correct():
    player_df = _make_player_df()
    result = aggregate_player_features(player_df)
    t1 = result[result["team_id"] == 1].iloc[0]
    expected_std = pd.Series([0.15, 0.12, 0.10, 0.09]).std()
    assert abs(t1["std_dev_pie"] - expected_std) < 1e-6


def test_std_dev_pie_single_player_returns_zero():
    single = pd.DataFrame(
        {
            "team_id": [1],
            "season_year": [2023],
            "player_id": [10],
            "games_played": [60],
            "minutes_pg": [35.0],
            "pie": [0.15],
            "age": [28],
        }
    )
    result = aggregate_player_features(single)
    assert result.iloc[0]["std_dev_pie"] == 0.0


# ── add_lag_features ───────────────────────────────────────────────────────────


def test_lag_features_do_not_cross_team_boundaries():
    """
    CRITICAL: team 2's prev_wins in 2023 must equal team 2's own 2022 wins (30),
    NOT team 1's 2023 wins (40). Validates groupby(team_id).shift(1) is used.
    """
    df = _make_two_season_team_df()
    result = add_lag_features(df)

    team2_2023 = result[(result["team_id"] == 2) & (result["season_year"] == 2023)]
    assert not team2_2023.empty
    assert team2_2023["prev_wins"].iloc[0] == 30.0, (
        f"Expected 30.0 (team 2's own 2022 wins), " f"got {team2_2023['prev_wins'].iloc[0]}"
    )


def test_first_season_per_team_has_nan_lag():
    df = _make_two_season_team_df()
    result = add_lag_features(df)
    first_rows = result[result["season_year"] == 2022]
    assert first_rows["prev_wins"].isna().all(), "First season per team must have NaN lag features"


def test_lag_features_include_new_columns():
    df = _make_two_season_team_df()
    result = add_lag_features(df)
    assert "prev_std_dev_pie" in result.columns
    assert "prev_top_3_minutes_share" in result.columns

    team1_2023 = result[(result["team_id"] == 1) & (result["season_year"] == 2023)].iloc[0]
    assert abs(team1_2023["prev_std_dev_pie"] - 0.02) < 1e-9
    assert abs(team1_2023["prev_top_3_minutes_share"] - 0.72) < 1e-9


# ── build_feature_row ──────────────────────────────────────────────────────────


def _make_ml_df():
    return pd.DataFrame(
        {
            "team_id": [1],
            "season_year": [2023],
            "wins": [47],
            "wins_normalized": [47.0],
            "off_rating": [115.0],
            "def_rating": [112.0],
            "net_rating": [3.0],
            "pts_pg": [115.0],
            "ts_pct": [0.57],
            "ast_to_ratio": [2.1],
            "pace": [100.0],
            "oreb_pct": [0.25],
            "dreb_pct": [0.75],
            "tm_tov_pct": [0.13],
            "team_avg_pie": [0.11],
            "playoff_team": [True],
            "team_avg_age": [27.0],
            "roster_turnover_pct": [0.30],
            "avg_games_played": [65.0],
            "star_age_flag": [False],
            "std_dev_pie": [0.025],
            "top_3_minutes_share": [0.72],
        }
    )


def test_build_feature_row_preseason_contains_new_features():
    ml_df = _make_ml_df()
    row = build_feature_row(1, 2024, ml_df, current_partial_stats=None, games_played=0)
    assert "prev_std_dev_pie" in row.columns
    assert "prev_top_3_minutes_share" in row.columns
    assert abs(row["prev_std_dev_pie"].iloc[0] - 0.025) < 1e-9
    assert abs(row["prev_top_3_minutes_share"].iloc[0] - 0.72) < 1e-9


def test_feature_contract_preseason_raises_with_current_stats():
    """Preseason mode must not accept current-season stats — data leakage guard."""
    ml_df = _make_ml_df()
    with pytest.raises(AssertionError):
        build_feature_row(
            1,
            2024,
            ml_df,
            current_partial_stats={"net_rating": 5.0},
            games_played=5,  # preseason (< 20)
        )


def test_build_feature_row_midseason_does_not_raise():
    ml_df = _make_ml_df()
    partial = {
        c: 1.0
        for c in [
            "net_rating",
            "off_rating",
            "def_rating",
            "pts_pg",
            "ts_pct",
            "ast_to_ratio",
            "pace",
            "oreb_pct",
            "dreb_pct",
            "tm_tov_pct",
            "team_avg_pie",
            "avg_games_played",
        ]
    }
    row = build_feature_row(1, 2024, ml_df, current_partial_stats=partial, games_played=30)
    assert row["prediction_mode"].iloc[0] == "mid-season"


def test_build_feature_row_raises_for_missing_prior_season():
    ml_df = _make_ml_df()
    with pytest.raises(ValueError):
        build_feature_row(1, 2026, ml_df)  # no 2025 row in ml_df
