import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import (
    load_ml_features, load_models, load_predictions_history, load_season_report,
    team_selector, no_data_warning,
)

st.header("Historical Performance")

ml_df = load_ml_features()
if ml_df.empty:
    no_data_warning()
    st.stop()

team_id = team_selector(key="p4_team")
team_df = ml_df[ml_df["team_id"] == team_id].sort_values("season_year")

if team_df.empty:
    st.warning("No historical data for this team.")
    st.stop()

season_model, _, _ = load_models()
report = load_season_report()

# ── Predicted vs Actual vs Naive Baseline ─────────────────────────────────────
st.subheader("Predicted vs Actual Wins")

actuals  = team_df["wins_normalized"].tolist()
seasons  = team_df["season_year"].tolist()
baseline = team_df["prev_wins_normalized"].tolist()   # naive: predict last year

predicted = []
if season_model is not None:
    from models.data_prep import FEATURE_COLS
    for _, row in team_df.iterrows():
        feat_vals = {c: row.get(c) for c in FEATURE_COLS if c in row.index}
        X = pd.DataFrame([feat_vals])
        if X.notna().all(axis=1).iloc[0]:
            p = float(np.clip(season_model.predict(X)[0], 0, 82))
        else:
            p = None
        predicted.append(p)
else:
    predicted = [None] * len(seasons)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=seasons, y=actuals, name="Actual Wins",
    mode="lines+markers", line=dict(color="#3498db", width=2),
))
fig.add_trace(go.Scatter(
    x=seasons, y=predicted, name="Model Predicted",
    mode="lines+markers", line=dict(color="#2ecc71", width=2, dash="dash"),
    connectgaps=True,
))
fig.add_trace(go.Scatter(
    x=seasons, y=baseline, name="Naive Baseline (prev season)",
    mode="lines", line=dict(color="#e74c3c", width=1.5, dash="dot"),
    connectgaps=True,
))
fig.update_layout(
    xaxis_title="Season", yaxis_title="Wins",
    legend=dict(orientation="h", y=-0.2), height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ── Accuracy summary ──────────────────────────────────────────────────────────
if any(p is not None for p in predicted):
    errs = [(s, abs(a - p)) for s, a, p in zip(seasons, actuals, predicted)
            if p is not None and a is not None]
    if errs:
        err_df = pd.DataFrame(errs, columns=["Season", "Abs Error (wins)"])
        col_b, col_w = st.columns(2)
        with col_b:
            st.subheader("Best Accuracy Seasons")
            st.dataframe(err_df.nsmallest(3, "Abs Error (wins)"),
                         hide_index=True, use_container_width=True)
        with col_w:
            st.subheader("Worst Accuracy Seasons")
            st.dataframe(err_df.nlargest(3, "Abs Error (wins)"),
                         hide_index=True, use_container_width=True)

# ── Weekly forecast evolution ──────────────────────────────────────────────────
st.divider()
st.subheader("In-Season Forecast Evolution")
st.caption("How this season's projected win total changed week by week (from Airflow DAG runs).")

preds_df = load_predictions_history()
if not preds_df.empty and "team_id" in preds_df.columns:
    team_preds = preds_df[preds_df["team_id"] == team_id].copy()
    if not team_preds.empty and "predicted_at" in team_preds.columns:
        team_preds["predicted_at"] = pd.to_datetime(team_preds["predicted_at"])
        fig2 = px.line(
            team_preds.sort_values("predicted_at"),
            x="predicted_at", y="predicted_wins",
            color="season_year" if "season_year" in team_preds.columns else None,
            markers=True,
            title="Weekly Predicted Wins Over Season",
            labels={"predicted_at": "Date", "predicted_wins": "Predicted Wins"},
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No weekly predictions yet. Run the Airflow DAG to populate.")
else:
    st.info("No prediction history yet. The Airflow DAG appends one row per team each Monday.")
