import os
from dotenv import load_dotenv

from etl.ingest import ingest_data, SEASONS
from etl.transform import transform_data
from etl.load import get_engine, save_to_parquet, upsert_to_postgres

load_dotenv()

PARQUET_DIR = "processed"


def run_etl(seasons: list = None):
    """
    Full ETL pipeline: ingest from nba_api → transform → load to PostgreSQL + Parquet.

    Parquet snapshots in processed/ serve as the Streamlit Community Cloud fallback
    when DATABASE_URL is not available.
    """
    os.makedirs(PARQUET_DIR, exist_ok=True)

    # 1. Ingest live data from nba_api
    raw_data = ingest_data(seasons or SEASONS)

    # 2. Transform: normalize, aggregate players, compute lag features
    transformed = transform_data(raw_data)

    # 3. Load to PostgreSQL (primary storage)
    engine = get_engine()
    upsert_to_postgres(
        transformed["team_stats"],
        "nba_team_season_stats",
        engine,
        conflict_cols=["team_id", "season"],
    )
    upsert_to_postgres(
        transformed["player_stats"],
        "nba_player_season_stats",
        engine,
        conflict_cols=["player_id", "team_id", "season_year"],
    )
    upsert_to_postgres(
        transformed["ml_features"],
        "ml_training_features",
        engine,
        conflict_cols=["team_id", "season_year"],
    )

    # 4. Save Parquet snapshots (Streamlit Cloud fallback)
    save_to_parquet(transformed["team_stats"], f"{PARQUET_DIR}/team_stats.parquet")
    save_to_parquet(transformed["player_stats"], f"{PARQUET_DIR}/player_stats.parquet")
    save_to_parquet(transformed["ml_features"], f"{PARQUET_DIR}/ml_features.parquet")

    print("ETL complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA season-level ETL pipeline")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Ingest only the last 5 seasons (~5 min). Use for dev/demo setup.",
    )
    args = parser.parse_args()
    run_etl(seasons=SEASONS[-5:] if args.fast else None)
