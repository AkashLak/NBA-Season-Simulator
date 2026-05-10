import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import (
    load_ml_features, load_models, team_selector, season_selector, no_data_warning,
)

st.header("Why This Prediction")
st.info(
    "SHAP values explain which features drove the game model's win probability "
    "predictions. With ~60K training rows, contributions are directionally meaningful. "
    "Treat individual values as approximate, not precise."
)

# ── Feature importance (global) ────────────────────────────────────────────────
shap_path = os.path.join("processed", "game_shap_summary.parquet")
if os.path.exists(shap_path):
    import plotly.express as px
    shap_df = pd.read_parquet(shap_path)
    if "feature" in shap_df.columns and "mean_abs_shap" in shap_df.columns:
        st.subheader("Global Feature Importance")
        fig = px.bar(
            shap_df.head(14).sort_values("mean_abs_shap"),
            x="mean_abs_shap", y="feature", orientation="h",
            title="Mean |SHAP Value| across all games",
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"},
            color="mean_abs_shap", color_continuous_scale="Viridis",
        )
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("SHAP summary not available. Train the game model to generate it.")

st.divider()

# ── Per-game waterfall (live) ──────────────────────────────────────────────────
st.subheader("Single-Game Prediction Breakdown")

ml_df = load_ml_features()
if ml_df.empty:
    no_data_warning()
    st.stop()

_, game_model, _ = load_models()
if game_model is None:
    st.info("Game model not trained yet.")
    st.stop()

col_t, col_s = st.columns(2)
with col_t:
    team_id = team_selector(key="p3_team")
with col_s:
    selected_season = season_selector(key="p3_season", df=ml_df)

predict_season = min(selected_season + 1, int(ml_df["season_year"].max()) + 1)

try:
    from models.simulate_season import (
        _get_schedule, _build_prior_features_lookup,
        _get_team_prior, _build_feature_row, _init_team_state,
        _rolling_pct, _rest_days,
    )
    from app.shared import get_engine
    from etl.transform_games import GAME_FEATURE_COLS
    from models.shap_analysis import get_prediction_explanation

    engine = get_engine()
    schedule = _get_schedule(predict_season, engine)
    prior_lookup = _build_prior_features_lookup(predict_season, engine)

    team_games = schedule[schedule["home_id"] == team_id].head(1)
    if team_games.empty:
        st.warning("No home games found for this team/season in the schedule.")
        st.stop()

    game = team_games.iloc[0]
    opp_id = int(game["away_id"])
    game_date = pd.Timestamp(game["game_date"]).strftime("%B %d, %Y")

    from app.shared import CURRENT_TEAMS
    team_name = CURRENT_TEAMS.get(team_id, f"Team {team_id}")
    opp_name  = CURRENT_TEAMS.get(opp_id,  f"Team {opp_id}")

    st.caption(
        f"Matchup: **{team_name}** (home) vs **{opp_name}** (away) — {game_date}. "
        f"Uses the first home game from the {predict_season - 1}–{str(predict_season)[2:]} "
        f"schedule as a representative example, with both teams' prior-season stats as inputs."
    )

    home_state = _init_team_state(team_id, _get_team_prior(team_id, prior_lookup))
    away_state = _init_team_state(
        opp_id,
        _get_team_prior(opp_id, prior_lookup)
    )

    feat_row = _build_feature_row(home_state, away_state, game["game_date"])
    X = pd.DataFrame([feat_row])[GAME_FEATURE_COLS]

    # Background sample from game features for SHAP
    gf_path = os.path.join("processed", "game_features.parquet")
    if os.path.exists(gf_path):
        bg_df = pd.read_parquet(gf_path)
        avail = [c for c in GAME_FEATURE_COLS if c in bg_df.columns]
        X_bg  = bg_df[avail].dropna().sample(min(200, len(bg_df)), random_state=42)
    else:
        X_bg = X.copy()

    explanation = get_prediction_explanation(game_model, X, X_bg)

    # SHAP waterfall via plotly
    import plotly.graph_objects as go
    contribs = explanation["contributions"]
    features = list(contribs.keys())
    values   = list(contribs.values())
    base     = explanation["base_value"]
    pred     = explanation["predicted_wins"]

    measure = ["relative"] * len(features) + ["total"]
    y_labels = features + ["Prediction"]
    x_vals   = values   + [0]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=measure,
        y=y_labels,
        x=x_vals,
        base=base,
        connector={"line": {"color": "rgba(100,100,100,0.3)"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    base_prob = sigmoid(base)
    pred_prob = sigmoid(pred)

    fig.update_layout(
        title=f"{team_name} vs {opp_name} — {game_date}",
        xaxis_title="Log-Odds Contribution (positive = helps win probability)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Values are in **log-odds space**, not probability. "
        f"Baseline win probability: {base_prob:.1%} (league average) → "
        f"Predicted win probability: **{pred_prob:.1%}** for this game."
    )
    st.caption(
        "Note: `off_rating`, `def_rating`, and `net_rating` are correlated features "
        "(net = off − def), so their individual bars can appear large and opposite-signed "
        "while mostly canceling out. `team_prev_net_rating` is the cleaner single signal to read."
    )

except Exception as e:
    st.warning(f"Could not compute live SHAP: {e}")
