import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import (
    load_ml_features, load_team_stats, load_player_stats,
    team_selector, season_selector, no_data_warning,
)

st.header("Team & Roster Dashboard")

ml_df     = load_ml_features()
team_stats = load_team_stats()
player_df  = load_player_stats()

if ml_df.empty:
    no_data_warning()
    st.stop()

col_t, col_s = st.columns(2)
with col_t:
    team_id = team_selector(key="p5_team")
with col_s:
    selected_season = season_selector(key="p5_season", df=ml_df)

# ── Season metric cards ────────────────────────────────────────────────────────
st.subheader("Team Stats vs Prior Season")

def get_row(df, team_id, season_year):
    rows = df[(df["team_id"] == team_id) & (df["season_year"] == season_year)]
    return rows.iloc[0] if not rows.empty else None

current_ts = get_row(team_stats, team_id, selected_season) if not team_stats.empty else None
prior_ts   = get_row(team_stats, team_id, selected_season - 1) if not team_stats.empty else None

def delta(cur_row, pri_row, col):
    if cur_row is None or pri_row is None:
        return None
    c = cur_row.get(col)
    p = pri_row.get(col)
    return round(float(c) - float(p), 2) if (c is not None and p is not None) else None

col1, col2, col3, col4 = st.columns(4)

if current_ts is not None:
    col1.metric("Points / Game",  f"{current_ts.get('pts_pg', 'N/A'):.1f}",
                delta=delta(current_ts, prior_ts, "pts_pg"))
    col2.metric("Net Rating",     f"{current_ts.get('net_rating', 'N/A'):.1f}",
                delta=delta(current_ts, prior_ts, "net_rating"))
    col3.metric("True Shooting%", f"{current_ts.get('ts_pct', 0):.3f}",
                delta=delta(current_ts, prior_ts, "ts_pct"))
    col4.metric("AST / TO",       f"{current_ts.get('ast_to_ratio', 'N/A'):.2f}",
                delta=delta(current_ts, prior_ts, "ast_to_ratio"))
else:
    st.info("Team stats not available for this season.")

# ── Roster quality callouts ────────────────────────────────────────────────────
ml_row = get_row(ml_df, team_id, selected_season)
if ml_row is not None:
    st.divider()
    col_a, col_b = st.columns(2)
    std_dev  = ml_row.get("std_dev_pie")
    top3_shr = ml_row.get("top_3_minutes_share")
    if std_dev is not None:
        col_a.metric(
            "Roster Balance (std PIE)",
            f"{std_dev:.3f}",
            help="Higher = star-dependent. Lower = balanced roster.",
        )
    if top3_shr is not None:
        col_b.metric(
            "Top-3 Minutes Share",
            f"{top3_shr:.1%}",
            help="Fraction of minutes from top-3 players. Higher = more reliant on stars.",
        )

# ── Roster table ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Roster")
st.caption("🔴 Fewer than 50 games played last season (injury risk proxy)")

if not player_df.empty:
    roster = player_df[
        (player_df["team_id"] == team_id) &
        (player_df["season_year"] == selected_season)
    ].copy()

    if roster.empty:
        st.info("No player data for this team/season.")
    else:
        display_cols = [c for c in
                        ["player_name", "age", "games_played", "minutes_pg", "pie", "pts_pg"]
                        if c in roster.columns]
        roster_sorted = roster.sort_values("minutes_pg", ascending=False) \
            if "minutes_pg" in roster.columns else roster

        def highlight_injury(row):
            gp = row.get("games_played", 82)
            color = "background-color: #ffe0e0" if pd.notna(gp) and gp < 50 else ""
            return [color] * len(row)

        styled = roster_sorted[display_cols].style.apply(highlight_injury, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    no_data_warning("Player stats not populated yet.")
