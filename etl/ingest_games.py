import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Reuse constants from season-level ingest
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2025)]
_RATE_LIMIT_SLEEP = 0.6


def _sleep():
    time.sleep(_RATE_LIMIT_SLEEP)


def season_to_year(season: str) -> int:
    """
    Convert nba_api season string to end year.
    '2023-24' → 2024,  '1998-99' → 1999,  '1999-00' → 2000
    """
    suffix = season.split("-")[1]
    n = int(suffix)
    return 1900 + n if n >= 50 else 2000 + n


def fetch_game_logs(season: str) -> pd.DataFrame:
    """
    Fetch all regular season game logs for all 30 teams for one season.
    Uses LeagueGameLog — one API call per season (~25 total for full history).

    Returns one row per team per game (~2,460 rows per full season).
    Key columns: GAME_ID, GAME_DATE, TEAM_ID, MATCHUP, WL, PTS, PLUS_MINUS.

    home_flag: 1 if team is home (MATCHUP contains "vs."), 0 if away ("@").
    opponent_id: derived by finding the other team sharing the same GAME_ID.
    """
    logs = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        direction="ASC",
    ).get_data_frames()[0]
    _sleep()

    if logs.empty:
        print(f"  Warning: no game logs returned for {season}")
        return pd.DataFrame()

    logs["season_year"] = season_to_year(season)
    logs["home_flag"] = logs["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)
    logs["win"] = (logs["WL"] == "W").astype(int)

    # Derive opponent_id: each GAME_ID appears exactly twice (once per team).
    # Group by GAME_ID, collect both TEAM_IDs, then map each row to the other one.
    game_teams = logs.groupby("GAME_ID")["TEAM_ID"].apply(list).to_dict()
    logs["opponent_id"] = logs.apply(
        lambda r: next(
            (t for t in game_teams.get(r["GAME_ID"], []) if t != r["TEAM_ID"]),
            None,
        ),
        axis=1,
    )

    # Drop any rows where opponent could not be resolved (rare: incomplete data)
    missing_opp = logs["opponent_id"].isna().sum()
    if missing_opp > 0:
        print(f"  Warning: {missing_opp} rows missing opponent_id for {season}, dropping")
        logs = logs.dropna(subset=["opponent_id"])

    logs["opponent_id"] = logs["opponent_id"].astype(int)

    # Normalise column names to snake_case for consistency with the rest of the pipeline
    rename = {
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbr",
        "PTS": "pts",
        "PLUS_MINUS": "plus_minus",
    }
    logs = logs.rename(columns={k: v for k, v in rename.items() if k in logs.columns})

    logs["game_date"] = pd.to_datetime(logs["game_date"])

    keep = [
        "game_id",
        "game_date",
        "season_year",
        "team_id",
        "team_abbr",
        "opponent_id",
        "home_flag",
        "win",
        "pts",
        "plus_minus",
    ]
    return logs[[c for c in keep if c in logs.columns]].copy()


def ingest_games(seasons: list = None) -> pd.DataFrame:
    """
    Fetch game logs for all requested seasons.

    Returns a single concatenated DataFrame (one row per team per game).
    Seasons that fail are skipped with a warning.
    """
    seasons = seasons or SEASONS
    frames = []
    for i, season in enumerate(seasons):
        print(f"Fetching game logs: {season} ({i + 1}/{len(seasons)})...")
        try:
            df = fetch_game_logs(season)
            if not df.empty:
                frames.append(df)
                print(f"  {len(df)} rows")
        except Exception as e:
            print(f"  Warning: game logs failed for {season}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f"Game ingestion complete: {len(result)} total rows across {len(seasons)} seasons")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Ingest only last 5 seasons (~2 min, for dev/demo)",
    )
    args = parser.parse_args()

    target_seasons = SEASONS[-5:] if args.fast else SEASONS
    print(f"Ingesting {len(target_seasons)} seasons of game logs...")
    df = ingest_games(target_seasons)

    import os

    os.makedirs("processed", exist_ok=True)
    df.to_parquet("processed/raw_game_logs.parquet", index=False)
    print(f"Saved to processed/raw_game_logs.parquet ({len(df)} rows)")
