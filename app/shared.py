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
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on the path when pages import this module
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LAKERS_ID    = 1610612747
PROCESSED    = "processed"
MODELS_DIR   = "models"


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


@st.cache_data(ttl=3600)
def load_team_names() -> dict:
    """Return {team_id: team_name} from nba_team_season_stats."""
    ts = load_team_stats()
    if ts.empty or "team_id" not in ts.columns or "team_name" not in ts.columns:
        return {LAKERS_ID: "Los Angeles Lakers"}
    return (
        ts.drop_duplicates("team_id")
        .set_index("team_id")["team_name"]
        .to_dict()
    )


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
    """Dropdown of all teams sorted alphabetically, Lakers selected by default."""
    names = load_team_names()
    sorted_teams = sorted(names.items(), key=lambda x: x[1])
    default_idx = next(
        (i for i, (tid, _) in enumerate(sorted_teams) if tid == LAKERS_ID), 0
    )
    selected = st.selectbox(
        label,
        options=[tid for tid, _ in sorted_teams],
        format_func=lambda tid: names.get(tid, str(tid)),
        index=default_idx,
        key=key,
    )
    return selected


def season_selector(label: str = "Season", key: str = "season_sel",
                    df: pd.DataFrame = None) -> int:
    """Dropdown of available seasons, most recent selected by default."""
    if df is None:
        df = load_ml_features()
    if df.empty or "season_year" not in df.columns:
        return 2025
    seasons = sorted(df["season_year"].unique(), reverse=True)
    return st.selectbox(label, options=seasons, index=0, key=key)


def wins_color(wins: float) -> str:
    if wins >= 50:
        return "#2ecc71"   # green
    if wins >= 44:
        return "#f39c12"   # amber
    return "#e74c3c"       # red


def no_data_warning(msg: str = "Run the ETL pipeline first to populate data."):
    st.warning(f"No data available. {msg}")
