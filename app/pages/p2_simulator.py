import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import (
    get_engine,
    load_ml_features,
    load_models,
    load_player_stats,
    no_data_warning,
    season_selector,
    team_selector,
)

st.header("Roster Simulator")
st.caption(
    "Swap players and see how the predicted win total changes. "
    "⚠️ Assumes performance scales with role (diminishing returns for larger roles). "
    "Real outcomes depend on fit, teammates, and coaching."
)

ml_df = load_ml_features()
player_df = load_player_stats()
_, game_model, _ = load_models()

if ml_df.empty or player_df.empty:
    no_data_warning()
    st.stop()

if game_model is None:
    st.info("Game model not trained yet. Run `python -m models.train_game_model`.")
    st.stop()

# ── Selectors ─────────────────────────────────────────────────────────────────
col_t, col_s = st.columns(2)
with col_t:
    team_id = team_selector(key="p2_team")
with col_s:
    selected_season = season_selector(key="p2_season", df=ml_df)

predict_season = min(selected_season + 1, int(ml_df["season_year"].max()) + 1)
st.caption(f"Simulating **{predict_season}** based on **{selected_season}** roster.")


# ── Compute NET_RATING_PER_PIE scaling factor from data ───────────────────────
@st.cache_data(ttl=3600)
def _net_rating_per_pie(ml_df_hash) -> float:
    """Regression coefficient: how much net_rating changes per unit of team_avg_pie."""
    df = load_ml_features()
    sub = df[df["net_rating"].notna() & df["team_avg_pie"].notna()]
    if len(sub) < 10:
        return 60.0  # league-average fallback
    cov = np.cov(sub["net_rating"], sub["team_avg_pie"])
    return float(cov[0, 1] / (np.var(sub["team_avg_pie"]) + 1e-9))


net_rating_scale = _net_rating_per_pie(len(ml_df))

# ── Load and store roster in session state ─────────────────────────────────────
state_key = f"roster_{team_id}_{selected_season}"
orig_key = f"orig_ids_{team_id}_{selected_season}"

if state_key not in st.session_state:
    roster = (
        player_df[(player_df["team_id"] == team_id) & (player_df["season_year"] == selected_season)]
        .copy()
        .reset_index(drop=True)
    )
    st.session_state[state_key] = roster
    st.session_state[orig_key] = set(roster["player_id"].tolist()) if not roster.empty else set()

roster_df = st.session_state[state_key].copy()
original_ids = st.session_state[orig_key]

if roster_df.empty:
    st.warning("No player data for this team/season. Run ETL to populate.")
    st.stop()


# ── Diminishing returns helper ─────────────────────────────────────────────────
def adjusted_pie(player_B: pd.Series, player_A: pd.Series) -> float:
    role_increase = max(
        0.0,
        float(player_A.get("minutes_pg", 20)) - float(player_B.get("minutes_pg", 20)),
    )
    factor = 1.0 - 0.05 * (role_increase / 10.0)
    return float(player_B.get("pie", 0.10)) * max(0.7, factor)


# ── Recompute team features from current roster ────────────────────────────────
def recompute_team_features(roster: pd.DataFrame) -> dict:
    qualified = (
        roster[roster.get("games_played", pd.Series(dtype=float)) >= 10].copy()
        if "games_played" in roster.columns
        else roster.copy()
    )
    if qualified.empty:
        qualified = roster.copy()
    total_min = qualified["minutes_pg"].sum() if "minutes_pg" in qualified.columns else 0
    if total_min == 0:
        return {}

    qualified["min_share"] = qualified["minutes_pg"] / total_min
    team_avg_pie = (
        float((qualified["pie"] * qualified["min_share"]).sum())
        if "pie" in qualified.columns
        else 0.10
    )

    total_team_min = roster["minutes_pg"].sum() if "minutes_pg" in roster.columns else 0
    top3_min = (
        roster.nlargest(3, "minutes_pg")["minutes_pg"].sum()
        if "minutes_pg" in roster.columns
        else 0
    )
    top_3_minutes_share = float(top3_min / total_team_min) if total_team_min > 0 else 0.7

    new_min = (
        roster[~roster["player_id"].isin(original_ids)]["minutes_pg"].sum()
        if "player_id" in roster.columns and "minutes_pg" in roster.columns
        else 0
    )
    roster_turnover = float(new_min / total_team_min) if total_team_min > 0 else 0

    return {
        "team_avg_pie": team_avg_pie,
        "top_3_minutes_share": top_3_minutes_share,
        "roster_turnover_pct": roster_turnover,
    }


# ── Baseline simulation (cached per team/season) ───────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _baseline_sim(team_id, predict_season):
    from models.simulate_season import simulate_team

    try:
        result = simulate_team(team_id, predict_season, game_model, get_engine())
        return result["expected_wins"]
    except Exception:
        return None


baseline_wins = _baseline_sim(team_id, predict_season)

# ── Current roster display ─────────────────────────────────────────────────────
st.subheader("Current Roster")
display_cols = [
    c for c in ["player_name", "age", "games_played", "minutes_pg", "pie"] if c in roster_df.columns
]
st.dataframe(
    (
        roster_df[display_cols].sort_values("minutes_pg", ascending=False)
        if "minutes_pg" in roster_df.columns
        else roster_df[display_cols]
    ),
    use_container_width=True,
    hide_index=True,
)

# ── Swap expanders ─────────────────────────────────────────────────────────────
st.subheader("Swap Players")

all_players = (
    player_df[player_df["season_year"] == player_df["season_year"].max()]
    .copy()
    .sort_values("minutes_pg", ascending=False)
    if "minutes_pg" in player_df.columns
    else player_df.copy()
)

current_roster_ids = (
    set(roster_df["player_id"].tolist()) if "player_id" in roster_df.columns else set()
)
candidates = (
    all_players[~all_players["player_id"].isin(current_roster_ids)]
    if "player_id" in all_players.columns
    else all_players
)

for idx, row in roster_df.iterrows():
    pname = row.get("player_name", f"Player {idx}")
    pie_val = f"{row.get('pie', 0):.3f}" if "pie" in roster_df.columns else "N/A"
    with st.expander(f"Swap: {pname}  (PIE: {pie_val}, Min: {row.get('minutes_pg', '?'):.0f})"):
        if candidates.empty or "player_id" not in candidates.columns:
            st.caption("No candidates available.")
            continue

        swap_choice = st.selectbox(
            "Replace with:",
            options=["-- No Change --"] + candidates["player_id"].tolist(),
            format_func=lambda pid: (
                "-- No Change --"
                if pid == "-- No Change --"
                else (
                    candidates[candidates["player_id"] == pid]["player_name"].iloc[0]
                    if "player_name" in candidates.columns
                    else str(pid)
                )
            ),
            key=f"swap_{idx}_{team_id}_{selected_season}",
        )

        if swap_choice != "-- No Change --" and st.button("Apply Swap", key=f"apply_{idx}"):
            new_player = candidates[candidates["player_id"] == swap_choice].iloc[0]
            new_pie = adjusted_pie(new_player, row)
            log_key = f"swap_log_{team_id}_{selected_season}"
            st.session_state.setdefault(log_key, []).append(
                {
                    "out": pname,
                    "in": new_player.get("player_name", str(swap_choice)),
                    "old_pie": row.get("pie", 0),
                    "new_pie": new_pie,
                    "minutes": row.get("minutes_pg", 0),
                }
            )
            st.session_state[state_key].at[idx, "player_id"] = int(new_player["player_id"])
            st.session_state[state_key].at[idx, "player_name"] = new_player.get("player_name", "")
            st.session_state[state_key].at[idx, "pie"] = new_pie
            if "age" in new_player.index:
                st.session_state[state_key].at[idx, "age"] = new_player["age"]
            if "games_played" in new_player.index:
                st.session_state[state_key].at[idx, "games_played"] = new_player["games_played"]
            st.rerun()

log_key = f"swap_log_{team_id}_{selected_season}"
if st.button("Reset Roster"):
    for k in [state_key, orig_key, log_key]:
        st.session_state.pop(k, None)
    st.rerun()

swap_log = st.session_state.get(log_key, [])
if swap_log:
    st.subheader("Applied Swaps")
    for entry in swap_log:
        pie_dir = "+" if entry["new_pie"] >= entry["old_pie"] else ""
        st.info(
            f"**{entry['out']}** → **{entry['in']}** "
            f"| PIE: {entry['old_pie']:.3f} → {entry['new_pie']:.3f} "
            f"({pie_dir}{entry['new_pie'] - entry['old_pie']:.3f}) "
            f"| Minutes unchanged at {entry['minutes']:.0f} min/g"
        )

# ── Live prediction ────────────────────────────────────────────────────────────
st.divider()
_current_ids = (
    set(st.session_state[state_key]["player_id"].tolist())
    if "player_id" in st.session_state[state_key].columns
    else set()
)
_is_modified = _current_ids != st.session_state.get(orig_key, set())
st.subheader(
    "Predicted Wins (Modified Roster)" if _is_modified else "Predicted Wins (Original Roster)"
)

feats = recompute_team_features(st.session_state[state_key])
if feats:
    prior_row = ml_df[(ml_df["team_id"] == team_id) & (ml_df["season_year"] == selected_season)]
    if not prior_row.empty:
        base_pie = float(prior_row.iloc[0].get("team_avg_pie", 0.10))
        base_net = float(prior_row.iloc[0].get("net_rating", 0.0))
        delta_pie = feats["team_avg_pie"] - base_pie
        adj_net_rating = base_net + delta_pie * net_rating_scale

        with st.spinner("Simulating league…"):
            from models.simulate_season import simulate_team

            try:
                result = simulate_team(
                    team_id,
                    predict_season,
                    game_model,
                    get_engine(),
                    team_quality_override={
                        "prev_net_rating": adj_net_rating,
                        "prev_off_rating": float(prior_row.iloc[0].get("off_rating", 110))
                        + delta_pie * net_rating_scale * 0.5,
                        "prev_def_rating": float(prior_row.iloc[0].get("def_rating", 110))
                        - delta_pie * net_rating_scale * 0.5,
                    },
                )
                sim_wins = result["expected_wins"]
                delta = sim_wins - baseline_wins if baseline_wins is not None else None

                col_w, col_d = st.columns(2)
                col_w.metric("Expected Wins", f"{sim_wins:.1f}")
                if delta is not None:
                    col_d.metric(
                        "Change vs Original Roster",
                        f"{delta:+.1f} wins",
                        delta=round(delta, 1),
                        delta_color="normal",
                    )
            except Exception as e:
                st.warning(f"Simulation error: {e}")
