"""
Integration tests — require live network access to nba_api.
Run with: pytest Testing/test_integration.py -m integration -v
Skip in CI with: pytest -m "not integration"
"""

import pytest


@pytest.mark.integration
def test_fetch_game_logs_returns_30_teams():
    """LeagueGameLog for one season must return all 30 team entries per game."""
    from etl.ingest_games import fetch_game_logs

    df = fetch_game_logs("2023-24")
    assert not df.empty, "fetch_game_logs returned empty DataFrame"
    n_teams = df["team_id"].nunique()
    assert n_teams == 30, f"Expected 30 unique teams, got {n_teams}"


@pytest.mark.integration
def test_fetch_game_logs_has_required_columns():
    from etl.ingest_games import fetch_game_logs

    df = fetch_game_logs("2023-24")
    required = [
        "game_id",
        "game_date",
        "season_year",
        "team_id",
        "opponent_id",
        "home_flag",
        "win",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


@pytest.mark.integration
def test_fetch_game_logs_win_flag_is_binary():
    from etl.ingest_games import fetch_game_logs

    df = fetch_game_logs("2023-24")
    assert set(df["win"].dropna().unique()).issubset(
        {0, 1}
    ), "win column must contain only 0 and 1"


@pytest.mark.integration
def test_ingest_games_full_season_row_count():
    """One full regular season should produce ~2,460 rows (30 teams × 82 games)."""
    from etl.ingest_games import ingest_games

    df = ingest_games(["2023-24"])
    assert len(df) >= 2400, f"Expected ~2460 rows, got {len(df)}"


@pytest.mark.integration
def test_team_season_stats_returns_30_teams():
    """LeagueDashTeamStats must return exactly 30 teams."""
    from etl.ingest import fetch_team_season_stats

    df = fetch_team_season_stats("2023-24")
    assert len(df) == 30, f"Expected 30 teams, got {len(df)}"


@pytest.mark.integration
def test_player_season_stats_returns_enough_players():
    """LeagueDashPlayerStats must return at least 400 players."""
    from etl.ingest import fetch_player_season_stats

    df = fetch_player_season_stats("2023-24")
    assert len(df) >= 400, f"Expected 400+ players, got {len(df)}"


@pytest.mark.integration
def test_ingest_data_returns_correct_structure():
    from etl.ingest import ingest_data

    result = ingest_data(["2023-24"])
    assert "team_stats" in result
    assert "player_stats" in result
    assert len(result["team_stats"]) == 30
    assert len(result["player_stats"]) >= 400
