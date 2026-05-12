import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import (
    load_ml_features,
    load_models,
    load_predictions_history,
    load_season_report,
    no_data_warning,
    team_selector,
)

st.header("Historical Performance")

ml_df = load_ml_features()
if ml_df.empty:
    no_data_warning()
    st.stop()

team_id = team_selector(key="p4_team")
team_df = ml_df[ml_df["team_id"] == team_id].sort_values("season_year").copy()

if team_df.empty:
    st.warning("No historical data for this team.")
    st.stop()

season_model, _, _ = load_models()
report = load_season_report()

# ── Predicted vs Actual vs Naive Baseline ─────────────────────────────────────
st.subheader("Predicted vs Actual Wins")

actuals = team_df["wins_normalized"].tolist()
seasons = team_df["season_year"].tolist()
baseline = team_df["prev_wins_normalized"].tolist()

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
fig.add_trace(
    go.Scatter(
        x=seasons,
        y=actuals,
        name="Actual Wins",
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=seasons,
        y=predicted,
        name="Model Predicted",
        mode="lines+markers",
        line=dict(color="#2ecc71", width=2, dash="dash"),
        connectgaps=True,
    )
)
fig.add_trace(
    go.Scatter(
        x=seasons,
        y=baseline,
        name="Naive Baseline (prev season)",
        mode="lines",
        line=dict(color="#e74c3c", width=1.5, dash="dot"),
        connectgaps=True,
    )
)
fig.update_layout(
    xaxis_title="Season",
    yaxis_title="Wins",
    legend=dict(orientation="h", y=-0.2),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ── Accuracy summary ──────────────────────────────────────────────────────────
if any(p is not None for p in predicted):
    errs = [
        (s, abs(a - p))
        for s, a, p in zip(seasons, actuals, predicted)
        if p is not None and a is not None
    ]
    if errs:
        err_df = pd.DataFrame(errs, columns=["Season", "Abs Error (wins)"])
        col_b, col_w = st.columns(2)
        with col_b:
            st.subheader("Best Accuracy Seasons")
            st.dataframe(
                err_df.nsmallest(3, "Abs Error (wins)"),
                hide_index=True,
                use_container_width=True,
            )
        with col_w:
            st.subheader("Worst Accuracy Seasons")
            st.dataframe(
                err_df.nlargest(3, "Abs Error (wins)"),
                hide_index=True,
                use_container_width=True,
            )

# ── Franchise Trend Analysis ──────────────────────────────────────────────────
st.divider()
st.subheader("Franchise Trend Analysis")

tab1, tab2, tab3 = st.tabs(["Efficiency Ratings", "Year-over-Year Change", "Roster Quality"])

# ── Tab 1: Net / Off / Def rating over time ───────────────────────────────────
with tab1:
    rating_cols = [c for c in ["net_rating", "off_rating", "def_rating"] if c in team_df.columns]
    if not rating_cols:
        st.info("Efficiency rating data not available.")
    else:
        fig_eff = go.Figure()
        colors = {
            "net_rating": "#9b59b6",
            "off_rating": "#e67e22",
            "def_rating": "#1abc9c",
        }
        labels = {
            "net_rating": "Net Rating",
            "off_rating": "Off Rating",
            "def_rating": "Def Rating",
        }
        for col in rating_cols:
            vals = team_df[col].where(team_df[col].notna()).tolist()
            fig_eff.add_trace(
                go.Scatter(
                    x=team_df["season_year"].tolist(),
                    y=vals,
                    name=labels[col],
                    mode="lines+markers",
                    line=dict(color=colors[col], width=2),
                    connectgaps=True,
                )
            )
        fig_eff.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig_eff.update_layout(
            xaxis_title="Season",
            yaxis_title="Rating",
            legend=dict(orientation="h", y=-0.2),
            height=400,
        )
        st.plotly_chart(fig_eff, use_container_width=True)
        st.caption(
            "Net Rating = Off − Def. League average is 0. "
            "Def Rating is inverted — lower is better defensively, but shown on the same axis."
        )

# ── Tab 2: Year-over-year win delta + net rating delta ───────────────────────
with tab2:
    team_df["win_delta"] = team_df["wins_normalized"] - team_df["prev_wins_normalized"]

    has_net = "net_rating" in team_df.columns
    if has_net:
        team_df["net_rating_delta"] = team_df["net_rating"].diff()

    delta_df = team_df.dropna(subset=["win_delta"]).copy()

    if delta_df.empty:
        st.info("Not enough seasons to compute year-over-year changes.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Win Total Change vs Prior Season**")
            bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in delta_df["win_delta"]]
            fig_wd = go.Figure(
                go.Bar(
                    x=delta_df["season_year"].tolist(),
                    y=delta_df["win_delta"].tolist(),
                    marker_color=bar_colors,
                    text=[f"{v:+.0f}" for v in delta_df["win_delta"]],
                    textposition="outside",
                )
            )
            fig_wd.add_hline(y=0, line_color="white", line_width=1)
            fig_wd.update_layout(
                xaxis_title="Season",
                yaxis_title="Win Change",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig_wd, use_container_width=True)

        with col_right:
            if has_net:
                st.markdown("**Net Rating Change vs Prior Season**")
                nr_delta = delta_df["net_rating_delta"].tolist()
                nr_colors = [
                    "#2ecc71" if (v is not None and v >= 0) else "#e74c3c" for v in nr_delta
                ]
                fig_nd = go.Figure(
                    go.Bar(
                        x=delta_df["season_year"].tolist(),
                        y=nr_delta,
                        marker_color=nr_colors,
                        text=[f"{v:+.1f}" if v is not None else "" for v in nr_delta],
                        textposition="outside",
                    )
                )
                fig_nd.add_hline(y=0, line_color="white", line_width=1)
                fig_nd.update_layout(
                    xaxis_title="Season",
                    yaxis_title="Net Rating Change",
                    height=380,
                    showlegend=False,
                )
                st.plotly_chart(fig_nd, use_container_width=True)
            else:
                st.info("Net rating data not available.")

        biggest_up = delta_df.loc[delta_df["win_delta"].idxmax()]
        biggest_down = delta_df.loc[delta_df["win_delta"].idxmin()]
        st.caption(
            f"Biggest improvement: **{biggest_up['season_year']:.0f}** "
            f"({biggest_up['win_delta']:+.0f} wins) · "
            f"Biggest decline: **{biggest_down['season_year']:.0f}** "
            f"({biggest_down['win_delta']:+.0f} wins)"
        )

# ── Tab 3: Roster quality (PIE trend + star dependence) ───────────────────────
with tab3:
    pie_cols = [
        c for c in ["team_avg_pie", "std_dev_pie", "top_3_minutes_share"] if c in team_df.columns
    ]

    if not pie_cols:
        st.info("Roster quality data not available.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            if "team_avg_pie" in team_df.columns:
                st.markdown("**Roster Talent (Avg PIE)**")
                pie_vals = team_df["team_avg_pie"].tolist()
                rolling_pie = team_df["team_avg_pie"].rolling(3, min_periods=1).mean().tolist()
                fig_pie = go.Figure()
                fig_pie.add_trace(
                    go.Scatter(
                        x=seasons,
                        y=pie_vals,
                        name="Avg PIE",
                        mode="lines+markers",
                        line=dict(color="#3498db", width=2),
                    )
                )
                fig_pie.add_trace(
                    go.Scatter(
                        x=seasons,
                        y=rolling_pie,
                        name="3-Year Rolling Avg",
                        mode="lines",
                        line=dict(color="#f39c12", width=2, dash="dash"),
                    )
                )
                fig_pie.update_layout(
                    xaxis_title="Season",
                    yaxis_title="PIE",
                    legend=dict(orientation="h", y=-0.2),
                    height=360,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption("PIE (Player Impact Estimate) — higher = more efficient roster.")

        with col_r:
            if "top_3_minutes_share" in team_df.columns and "std_dev_pie" in team_df.columns:
                st.markdown("**Star Dependence**")
                fig_star = go.Figure()
                fig_star.add_trace(
                    go.Scatter(
                        x=seasons,
                        y=team_df["top_3_minutes_share"].tolist(),
                        name="Top-3 Min Share",
                        mode="lines+markers",
                        line=dict(color="#e74c3c", width=2),
                        yaxis="y",
                    )
                )
                fig_star.add_trace(
                    go.Scatter(
                        x=seasons,
                        y=team_df["std_dev_pie"].tolist(),
                        name="Std Dev PIE",
                        mode="lines+markers",
                        line=dict(color="#9b59b6", width=2, dash="dash"),
                        yaxis="y2",
                    )
                )
                fig_star.update_layout(
                    xaxis_title="Season",
                    yaxis=dict(title="Top-3 Min Share", tickformat=".0%"),
                    yaxis2=dict(title="Std Dev PIE", overlaying="y", side="right"),
                    legend=dict(orientation="h", y=-0.2),
                    height=360,
                )
                st.plotly_chart(fig_star, use_container_width=True)
                st.caption(
                    "High top-3 min share + high std dev PIE = star-dependent roster. "
                    "Low values on both = balanced depth."
                )

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
            x="predicted_at",
            y="predicted_wins",
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
