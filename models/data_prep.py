import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Features used for training and inference.
# All are knowable before a season starts:
#   prev_* = prior season's final stats (lag features)
#   team_avg_* / roster_* = current roster composition derived from prior individual stats
FEATURE_COLS = [
    # Prior season team performance
    "prev_wins_normalized",
    "prev_off_rating",
    "prev_def_rating",
    "prev_net_rating",
    "prev_pts_pg",
    "prev_ts_pct",
    "prev_ast_to_ratio",
    "prev_pace",
    "prev_oreb_pct",
    "prev_dreb_pct",
    "prev_tm_tov_pct",
    "prev_team_avg_pie",
    "prev_playoff_team",
    # Current roster composition (knowable from offseason transactions)
    "team_avg_pie",
    "team_avg_age",
    "roster_turnover_pct",
    "avg_games_played",
    "star_age_flag",
    "prev_std_dev_pie",
    "prev_top_3_minutes_share",
]

TARGET_COL = "wins_normalized"      # model trains on 82-game pace wins
TARGET_RAW_COL = "wins"             # raw wins stored for app display
PLAYOFF_TARGET_COL = "playoff_team"


def load_from_postgres(table: str = "ml_training_features") -> pd.DataFrame:
    """Load ML features from PostgreSQL (primary data source)."""
    from sqlalchemy import create_engine
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set")
    engine = create_engine(url)
    return pd.read_sql(
        f"SELECT * FROM {table} ORDER BY team_id, season_year", engine
    )


def load_from_parquet(path: str = "processed/ml_features.parquet") -> pd.DataFrame:
    """Load ML features from Parquet snapshot (Streamlit Cloud fallback)."""
    return pd.read_parquet(path)


def load_data() -> pd.DataFrame:
    """
    Load ML features — PostgreSQL first, Parquet fallback.
    Raises if neither source is available.
    """
    try:
        df = load_from_postgres()
        print(f"Loaded {len(df)} rows from PostgreSQL")
        return df
    except Exception as e:
        print(f"PostgreSQL unavailable ({e}), falling back to Parquet")

    path = "processed/ml_features.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data source available: PostgreSQL failed and {path} not found. "
            "Run etl/run_etl.py first."
        )
    df = load_from_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Return (X, y_wins, y_playoff) ready for model training.

    Drops rows where any lag feature is NaN — these are each team's first
    season where no prior-season data exists. They are retained in the DB
    for inference context but excluded from training.
    """
    # Cast boolean columns to int so all models handle them uniformly
    bool_cols = ["prev_playoff_team", "star_age_flag"]
    for col in bool_cols:
        if col in df.columns:
            df = df.copy()
            df[col] = df[col].astype(float)

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} feature columns not found in data: {missing}")

    X = df[available].copy()
    y_wins = df[TARGET_COL].copy()
    y_playoff = df[PLAYOFF_TARGET_COL].astype(int).copy()

    # Drop rows with NaN in any feature or target
    valid_mask = X.notna().all(axis=1) & y_wins.notna() & y_playoff.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN lag features (first season per team)")

    return X[valid_mask], y_wins[valid_mask], y_playoff[valid_mask]


def get_team_latest_row(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Return the most recent feature row for a given team (used for inference)."""
    team_rows = df[df["team_id"] == team_id].sort_values("season_year")
    if team_rows.empty:
        raise ValueError(f"No data found for team_id {team_id}")
    return team_rows.iloc[[-1]]


def chronological_split(
    X: pd.DataFrame,
    y_wins: pd.Series,
    y_playoff: pd.Series,
    df_full: pd.DataFrame,
    holdout_seasons: int = 3,
) -> tuple:
    """
    Split into train and holdout sets by season year.
    Holdout = last `holdout_seasons` × 30 teams ≈ 90 rows.
    Training = all earlier seasons.
    """
    season_years = df_full.loc[X.index, "season_year"]
    cutoff = sorted(season_years.unique())[-holdout_seasons]
    train_mask = season_years < cutoff
    test_mask = season_years >= cutoff

    return (
        X[train_mask], X[test_mask],
        y_wins[train_mask], y_wins[test_mask],
        y_playoff[train_mask], y_playoff[test_mask],
    )
