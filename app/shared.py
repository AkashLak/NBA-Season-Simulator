"""
Shared data-loading and model-loading functions for the NBA Win Predictor app.

All functions are cached so they execute once per session and are reused
across all 6 pages. PostgreSQL is the primary source; Parquet files in
processed/ are the Streamlit Community Cloud fallback when DATABASE_URL is
not set.
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on the path when pages import this module
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LAKERS_ID = 1610612747
PROCESSED = "processed"
MODELS_DIR = "models"

# Canonical current team names keyed by nba_api team_id.
# Covers all relocations/rebrands: Seattle→OKC, NJ→Brooklyn, Vancouver→Memphis, etc.
CURRENT_TEAMS: dict[int, str] = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls",
    1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612754: "Indiana Pacers",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans",
    1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612764: "Washington Wizards",
}


# ── Engine ─────────────────────────────────────────────────────────────────────


@st.cache_resource
def get_engine():
    """SQLAlchemy engine. Returns None if DATABASE_URL is not set (Parquet fallback)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    try:
        from sqlalchemy import create_engine

        return create_engine(url)
    except Exception:
        return None


# ── Season-level data ──────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_ml_features() -> pd.DataFrame:
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT * FROM ml_training_features ORDER BY team_id, season_year",
                engine,
            )
        except Exception:
            pass
    path = os.path.join(PROCESSED, "ml_features.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_team_stats() -> pd.DataFrame:
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT * FROM nba_team_season_stats ORDER BY team_id, season_year",
                engine,
            )
        except Exception:
            pass
    path = os.path.join(PROCESSED, "team_stats.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_player_stats() -> pd.DataFrame:
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT * FROM nba_player_season_stats ORDER BY team_id, season_year",
                engine,
            )
        except Exception:
            pass
    path = os.path.join(PROCESSED, "player_stats.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_team_names() -> dict:
    """Return {team_id: current_team_name} for all 30 active franchises."""
    return CURRENT_TEAMS


@st.cache_data(ttl=3600)
def load_predictions_history() -> pd.DataFrame:
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT * FROM nba_predictions ORDER BY predicted_at",
                engine,
            )
        except Exception:
            pass
    return pd.DataFrame()


# ── Game-level data ────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_game_features() -> pd.DataFrame:
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql(
                "SELECT * FROM nba_game_features ORDER BY game_date",
                engine,
            )
        except Exception:
            pass
    path = os.path.join(PROCESSED, "game_features.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


# ── Models ─────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_models() -> tuple:
    """
    Returns (season_model, game_model, playoff_model).
    Any that are missing return None — pages must handle None gracefully.
    """

    def _load(path):
        return joblib.load(path) if os.path.exists(path) else None

    return (
        _load(os.path.join(MODELS_DIR, "best_wins_model.pkl")),
        _load(os.path.join(MODELS_DIR, "game_win_model.pkl")),
        _load(os.path.join(MODELS_DIR, "playoff_classifier.pkl")),
    )


# ── Report helpers ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_season_report() -> dict:
    path = os.path.join(MODELS_DIR, "model_selection_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_game_report() -> dict:
    path = os.path.join(MODELS_DIR, "game_model_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_learning_curve() -> dict:
    path = os.path.join(PROCESSED, "learning_curve.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── UI helpers ─────────────────────────────────────────────────────────────────


def team_selector(label: str = "Team", key: str = "team_sel") -> int:
    """Dropdown of all 30 current franchises sorted alphabetically, Lakers default."""
    sorted_teams = sorted(CURRENT_TEAMS.items(), key=lambda x: x[1])
    default_idx = next(
        (i for i, (tid, _) in enumerate(sorted_teams) if tid == LAKERS_ID), 0
    )
    return st.selectbox(
        label,
        options=[tid for tid, _ in sorted_teams],
        format_func=lambda tid: CURRENT_TEAMS.get(tid, str(tid)),
        index=default_idx,
        key=key,
    )


def season_selector(
    label: str = "Season", key: str = "season_sel", df: pd.DataFrame = None
) -> int:
    """Dropdown of available seasons, most recent selected by default."""
    if df is None:
        df = load_ml_features()
    if df.empty or "season_year" not in df.columns:
        return 2025
    seasons = sorted(df["season_year"].unique(), reverse=True)
    return st.selectbox(label, options=seasons, index=0, key=key)


def wins_color(wins: float) -> str:
    if wins >= 50:
        return "#2ecc71"  # green
    if wins >= 44:
        return "#f39c12"  # amber
    return "#e74c3c"  # red


def no_data_warning(msg: str = "Run the ETL pipeline first to populate data."):
    st.warning(f"No data available. {msg}")
