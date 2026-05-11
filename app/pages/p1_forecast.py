import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import (
    load_game_report,
    load_ml_features,
    load_models,
    load_season_report,
    no_data_warning,
    season_selector,
    team_selector,
    wins_color,
)

st.header("Season Forecast")

ml_df = load_ml_features()
if ml_df.empty:
    no_data_warning()
    st.stop()

season_model, game_model, playoff_model = load_models()

col_team, col_season = st.columns(2)
with col_team:
    team_id = team_selector(key="p1_team")
with col_season:
    # Predict next season — cap at max available so prior-season data exists
    max_season = int(ml_df["season_year"].max())
    selected_season = season_selector(key="p1_season", df=ml_df)
    predict_season = min(selected_season + 1, max_season + 1)

st.caption(
    f"Predicting **{predict_season}** season using **{selected_season}** "
    "as the prior-season baseline."
)

# ── Season model (fast, preseason estimate) ────────────────────────────────────
season_pred, sim_pred = None, None
season_report = load_season_report()
game_report = load_game_report()

if season_model is not None:
    try:
        from etl.transform import build_feature_row
        from models.data_prep import FEATURE_COLS

        row = build_feature_row(team_id, predict_season, ml_df, None, 0)
        avail = [c for c in FEATURE_COLS if c in row.columns]
        X = row[avail]
        season_pred = float(np.clip(season_model.predict(X)[0], 0, 82))
    except Exception as e:
        st.warning(f"Season model unavailable: {e}")

# ── Game simulation (full league, expected value) ─────────────────────────────
if game_model is not None:
    try:
        from app.shared import get_engine
        from models.simulate_season import simulate_team

        with st.spinner("Running league simulation…"):
            result = simulate_team(team_id, predict_season, game_model, get_engine())
        sim_pred = result["expected_wins"]
    except Exception as e:
        st.warning(f"Simulation unavailable: {e}")

# ── Playoff probability ────────────────────────────────────────────────────────
playoff_prob = None
if playoff_model is not None and season_model is not None and season_pred is not None:
    try:
        from etl.transform import build_feature_row
        from models.data_prep import FEATURE_COLS

        row = build_feature_row(team_id, predict_season, ml_df, None, 0)
        avail = [c for c in FEATURE_COLS if c in row.columns]
        playoff_prob = float(playoff_model.predict_proba(row[avail])[0][1])
    except Exception:
        pass

# ── Display ────────────────────────────────────────────────────────────────────
st.divider()

rmse = (
    season_report.get("results", {})
    .get(season_report.get("winner", ""), {})
    .get("rmse", 5.0)
)
game_logloss = game_report.get("winner_metrics", {}).get("holdout_logloss")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Preseason Estimate")
    st.caption("Season-level regression model (fast, uses season averages)")
    if season_pred is not None:
        color = wins_color(season_pred)
        lo = max(0, season_pred - 1.96 * rmse)
        hi = min(82, season_pred + 1.96 * rmse)
        st.markdown(
            f"<div style='text-align:center;padding:24px;border-radius:12px;"
            f"background:{color}22;border:2px solid {color}'>"
            f"<span style='font-size:72px;font-weight:bold;color:{color}'>"
            f"{season_pred:.1f}</span><br>"
            f"<span style='color:#888'>Approx 95% range: {lo:.0f}–{hi:.0f} wins</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Season model not trained yet. Run `python -m models.train_model`.")

with col2:
    st.subheader("Full Simulation")
    st.caption("Game-level model × 82-game schedule (expected value, no sampling)")
    if sim_pred is not None:
        color = wins_color(sim_pred)
        game_rmse = game_report.get("winner_metrics", {}).get("holdout_logloss", 0.63)
        st.markdown(
            f"<div style='text-align:center;padding:24px;border-radius:12px;"
            f"background:{color}22;border:2px solid {color}'>"
            f"<span style='font-size:72px;font-weight:bold;color:{color}'>"
            f"{sim_pred:.1f}</span><br>"
            f"<span style='color:#888'>Game model log loss: {game_rmse:.4f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Game model not trained yet. Run `python -m models.train_game_model`.")

# ── Playoff probability gauge ──────────────────────────────────────────────────
if playoff_prob is not None:
    st.divider()
    st.subheader("Playoff Probability")
    col_g, col_pct = st.columns([3, 1])
    with col_g:
        st.progress(playoff_prob)
    with col_pct:
        st.metric("", f"{playoff_prob * 100:.1f}%")

# ── Baseline comparison ────────────────────────────────────────────────────────
baseline = season_report.get("baseline", {})
if baseline:
    st.divider()
    b_rmse = baseline.get("baseline_rmse", 0)
    improvement = season_report.get("improvement_over_baseline", 0)
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Model RMSE", f"±{rmse:.1f} wins")
    col_b2.metric(
        "Naive Baseline RMSE",
        f"±{b_rmse:.1f} wins",
        help="Predict: this season's wins = last season's wins",
    )
    col_b3.metric(
        "Improvement",
        f"{improvement:+.1f} wins",
        delta=f"{improvement:+.1f}",
        delta_color="normal",
    )

    if season_report.get("baseline_warning"):
        st.warning(season_report["baseline_warning"])
