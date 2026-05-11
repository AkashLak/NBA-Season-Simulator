import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

LAKERS_TEAM_ID = 1610612747

# 1996-97 through 2024-25 — practical floor for reliable nba_api advanced stats
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2025)]

_RATE_LIMIT_SLEEP = 0.6


def _sleep():
    time.sleep(_RATE_LIMIT_SLEEP)


def season_to_year(season: str) -> int:
    """
    Convert nba_api season string to end year.
    '2023-24' → 2024,  '1998-99' → 1999,  '1999-00' → 2000
    Suffix >= 50 means 1900s; suffix < 50 means 2000s.
    """
    suffix = season.split("-")[1]
    n = int(suffix)
    return 1900 + n if n >= 50 else 2000 + n


def fetch_team_season_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-game base + advanced team stats for all 30 teams for one season.
    Returns a merged DataFrame with both offensive/defensive metrics.
    """
    base = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    _sleep()

    advanced = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    _sleep()

    adv_cols = [
        c
        for c in [
            "TEAM_ID",
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "PACE",
            "TS_PCT",
            "EFG_PCT",
            "OREB_PCT",
            "DREB_PCT",  # rebounding splits — OREB_PCT is a key four-factor
            "TM_TOV_PCT",  # team turnover rate — more precise than AST/TOV ratio
            "PIE",
        ]
        if c in advanced.columns
    ]
    merged = base.merge(advanced[adv_cols], on="TEAM_ID", how="left")
    merged["season"] = season
    merged["season_year"] = season_to_year(season)
    merged["is_lakers"] = merged["TEAM_ID"] == LAKERS_TEAM_ID
    return merged


def fetch_player_season_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-game base + advanced player stats for all players for one season.
    AGE and GP come from base; PIE, USG_PCT, TS_PCT come from advanced.
    """
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    _sleep()

    advanced = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    _sleep()

    base_cols = [
        c
        for c in ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "AGE", "GP", "MIN", "PTS"]
        if c in base.columns
    ]
    adv_cols = [
        c
        for c in ["PLAYER_ID", "TEAM_ID", "PIE", "NET_RATING", "USG_PCT", "TS_PCT"]
        if c in advanced.columns
    ]

    merged = base[base_cols].merge(
        advanced[adv_cols], on=["PLAYER_ID", "TEAM_ID"], how="left"
    )
    merged["season_year"] = season_to_year(season)
    return merged


def ingest_data(seasons: list = None) -> dict:
    """
    Pull team + player stats for all requested seasons from nba_api.

    Returns dict with keys:
        "team_stats"   — one row per team per season (~750 rows for full history)
        "player_stats" — one row per player per team per season (~11K+ rows)
    """
    if seasons is None:
        seasons = SEASONS

    all_team_stats = []
    all_player_stats = []

    for i, season in enumerate(seasons):
        print(f"Fetching season {season} ({i + 1}/{len(seasons)})...")
        try:
            team_df = fetch_team_season_stats(season)
            all_team_stats.append(team_df)
        except Exception as e:
            print(f"  Warning: team stats failed for {season}: {e}")

        try:
            player_df = fetch_player_season_stats(season)
            all_player_stats.append(player_df)
        except Exception as e:
            print(f"  Warning: player stats failed for {season}: {e}")

    team_stats = (
        pd.concat(all_team_stats, ignore_index=True)
        if all_team_stats
        else pd.DataFrame()
    )
    player_stats = (
        pd.concat(all_player_stats, ignore_index=True)
        if all_player_stats
        else pd.DataFrame()
    )

    print(
        f"Ingestion complete: {len(team_stats)} team-season rows, "
        f"{len(player_stats)} player-season rows"
    )
    return {"team_stats": team_stats, "player_stats": player_stats}
