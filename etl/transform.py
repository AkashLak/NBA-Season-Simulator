import numpy as np
import pandas as pd


# ── Column normalization ──────────────────────────────────────────────────────

def normalize_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rename nba_api uppercase columns to clean snake_case."""
    rename_map = {
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbr",
        "TEAM_NAME": "team_name",
        "GP": "games_played",
        "W": "wins",
        "L": "losses",
        "WIN_PCT": "win_pct",
        "PTS": "pts_pg",
        "REB": "reb_pg",
        "AST": "ast_pg",
        "STL": "stl_pg",
        "BLK": "blk_pg",
        "TOV": "tov_pg",
        "OFF_RATING": "off_rating",
        "DEF_RATING": "def_rating",
        "NET_RATING": "net_rating",
        "PACE": "pace",
        "TS_PCT": "ts_pct",
        "EFG_PCT": "efg_pct",
        "OREB_PCT": "oreb_pct",
        "DREB_PCT": "dreb_pct",
        "TM_TOV_PCT": "tm_tov_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "ast_pg" in df.columns and "tov_pg" in df.columns:
        df["ast_to_ratio"] = df["ast_pg"] / df["tov_pg"].replace(0, np.nan)
    return df


def normalize_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rename nba_api uppercase player columns to clean snake_case."""
    rename_map = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "AGE": "age",
        "GP": "games_played",
        "MIN": "minutes_pg",
        "PTS": "pts_pg",
        "PIE": "pie",
        "NET_RATING": "net_rating",
        "USG_PCT": "usg_pct",
        "TS_PCT": "ts_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [v for v in rename_map.values() if v in df.columns] + ["season_year"]
    return df[[c for c in keep if c in df.columns]].copy()


# ── Player-level aggregation ──────────────────────────────────────────────────

# Minimum games threshold for PIE/age aggregation.
# Players with fewer games have volatile per-game stats that skew team averages.
# Verified via live API: e.g. 2024-25 Alondes Williams PIE=0.40 from 1 game, 3.7 min.
_MIN_GAMES_FOR_PIE = 10


def aggregate_player_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level stats to team-season level.

    Features produced:
        team_avg_pie        — minutes-weighted avg PIE (Player Impact Estimate)
        team_avg_age        — minutes-weighted average roster age
        avg_games_played    — avg games played for top-8-minute players (injury proxy)
        star_age_flag       — True if any player is 32+ and averages 30+ min/game

    Players with < _MIN_GAMES_FOR_PIE games are excluded from PIE and age
    aggregations to prevent small-sample noise from skewing team features.
    They remain in games_played and star_age_flag calculations.
    """
    rows = []
    for (team_id, season_year), group in player_df.groupby(["team_id", "season_year"]):
        group = group.copy()

        # Qualified players only: exclude fringe call-ups with tiny sample sizes
        qualified = group[group["games_played"] >= _MIN_GAMES_FOR_PIE]
        total_min = qualified["minutes_pg"].sum()
        if total_min == 0:
            continue

        qualified = qualified.copy()
        qualified["min_share"] = qualified["minutes_pg"] / total_min

        team_avg_pie = (
            (qualified["pie"] * qualified["min_share"]).sum()
            if "pie" in qualified.columns else np.nan
        )
        team_avg_age = (
            (qualified["age"] * qualified["min_share"]).sum()
            if "age" in qualified.columns else np.nan
        )

        # Use full group (not qualified) for injury proxy — a star playing < 10 games
        # IS the injury signal we want to capture.
        top8 = group.nlargest(8, "minutes_pg")
        avg_games_played = top8["games_played"].mean() if len(top8) > 0 else np.nan

        star_age_flag = bool(
            ((group["age"] >= 32) & (group["minutes_pg"] >= 30)).any()
        ) if "age" in group.columns else False

        # Roster quality variance — high std_dev_pie means star-dependent roster
        std_dev_pie = (
            float(qualified["pie"].std())
            if "pie" in qualified.columns and len(qualified) >= 2
            else 0.0
        )

        # Minutes concentration — fraction of team minutes from top-3 players
        total_team_min = group["minutes_pg"].sum()
        top_3_minutes_share = float(
            group.nlargest(3, "minutes_pg")["minutes_pg"].sum() / total_team_min
        ) if total_team_min > 0 else np.nan

        rows.append({
            "team_id": team_id,
            "season_year": season_year,
            "team_avg_pie": team_avg_pie,
            "team_avg_age": team_avg_age,
            "avg_games_played": avg_games_played,
            "star_age_flag": star_age_flag,
            "std_dev_pie": std_dev_pie,
            "top_3_minutes_share": top_3_minutes_share,
        })

    return pd.DataFrame(rows)


def compute_roster_turnover(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute roster_turnover_pct for each team-season.
    = fraction of minutes played by players new to the team vs the prior season.
    First season per team returns NaN (no prior to compare against).
    """
    rows = []
    grouped = {k: v for k, v in player_df.groupby(["team_id", "season_year"])}

    for (team_id, season_year), current in grouped.items():
        prior = grouped.get((team_id, season_year - 1))

        if prior is None or prior.empty:
            rows.append({"team_id": team_id, "season_year": season_year, "roster_turnover_pct": np.nan})
            continue

        total_min = current["minutes_pg"].sum()
        if total_min == 0:
            rows.append({"team_id": team_id, "season_year": season_year, "roster_turnover_pct": np.nan})
            continue

        prior_ids = set(prior["player_id"].unique())
        new_min = current[~current["player_id"].isin(prior_ids)]["minutes_pg"].sum()
        rows.append({
            "team_id": team_id,
            "season_year": season_year,
            "roster_turnover_pct": float(new_min / total_min),
        })

    return pd.DataFrame(rows)


# ── Lag features ──────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prior-season features for each team.

    Uses groupby(team_id).shift(1) — never a global shift — to prevent
    one team's stats from bleeding into another team's lag values.
    First season per team will have NaN lag features; these rows are
    dropped during model training but retained for inference.
    """
    df = df.sort_values(["team_id", "season_year"]).copy()
    lag_cols = [
        "wins", "wins_normalized",
        "off_rating", "def_rating", "net_rating",
        "pts_pg", "ts_pct", "ast_to_ratio", "pace",
        "oreb_pct", "dreb_pct", "tm_tov_pct",
        "team_avg_pie", "playoff_team",
        "std_dev_pie", "top_3_minutes_share",
    ]
    for col in lag_cols:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby("team_id")[col].shift(1)
    return df


# ── ML feature table assembly ─────────────────────────────────────────────────

def build_ml_features(
    team_df: pd.DataFrame,
    player_agg_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join team stats + player aggregates + roster turnover, then add lag features."""
    df = team_df.merge(player_agg_df, on=["team_id", "season_year"], how="left")
    df = df.merge(turnover_df, on=["team_id", "season_year"], how="left")

    # Normalize wins to 82-game pace to handle shortened seasons.
    # 2019-20: 65-72 games (COVID bubble). 2011-12: 66 games (lockout).
    # Raw wins kept for app display; wins_normalized is the model training target.
    df["wins_normalized"] = (df["wins"] / df["games_played"] * 82).round(1)

    # playoff_team threshold uses normalized wins so shortened seasons don't
    # artificially shift teams below the 44-win cutoff.
    df["playoff_team"] = df["wins_normalized"] >= 44

    df = add_lag_features(df)

    # prev_playoff_team arrives from add_lag_features as a float (0/1) — cast to bool.
    if "prev_playoff_team" in df.columns:
        df["prev_playoff_team"] = df["prev_playoff_team"].astype("boolean")

    return df


# ── Prediction mode & inference feature row ───────────────────────────────────

def detect_prediction_mode(games_played: int) -> str:
    """Return 'preseason' if fewer than 20 games played, 'mid-season' otherwise."""
    return "preseason" if games_played < 20 else "mid-season"


def build_feature_row(
    team_id: int,
    season_year: int,
    ml_features_df: pd.DataFrame,
    current_partial_stats: dict = None,
    games_played: int = 0,
) -> pd.DataFrame:
    """
    Build a single-row feature vector for inference (any team, any season).

    Preseason  (games_played < 20): use prior season's final stats as all features.
    Mid-season (games_played >= 20): blend current partial-season stats with prior
        season stats weighted by games_played/82. Injured players' reduced games_played
        and the team's lower net_rating automatically reflect their absence.
    """
    mode = detect_prediction_mode(games_played)

    prior_rows = ml_features_df[
        (ml_features_df["team_id"] == team_id) &
        (ml_features_df["season_year"] == season_year - 1)
    ]
    if prior_rows.empty:
        raise ValueError(f"No prior season data for team {team_id}, season {season_year - 1}")

    prior = prior_rows.iloc[0].to_dict()

    base = {
        "team_id": team_id,
        "season_year": season_year,
        "prediction_mode": mode,
        # lag features from prior season
        "prev_wins": prior.get("wins"),
        "prev_wins_normalized": prior.get("wins_normalized"),
        "prev_off_rating": prior.get("off_rating"),
        "prev_def_rating": prior.get("def_rating"),
        "prev_net_rating": prior.get("net_rating"),
        "prev_pts_pg": prior.get("pts_pg"),
        "prev_ts_pct": prior.get("ts_pct"),
        "prev_ast_to_ratio": prior.get("ast_to_ratio"),
        "prev_pace": prior.get("pace"),
        "prev_oreb_pct": prior.get("oreb_pct"),
        "prev_dreb_pct": prior.get("dreb_pct"),
        "prev_tm_tov_pct": prior.get("tm_tov_pct"),
        "prev_team_avg_pie": prior.get("team_avg_pie"),
        "prev_playoff_team": prior.get("playoff_team"),
        # roster features (from prior season for preseason; may be updated for mid-season)
        "team_avg_age": prior.get("team_avg_age"),
        "roster_turnover_pct": prior.get("roster_turnover_pct"),
        "avg_games_played": prior.get("avg_games_played"),
        "star_age_flag": prior.get("star_age_flag"),
    }

    # blendable current-season features
    blend_cols = [
        "off_rating", "def_rating", "net_rating",
        "pts_pg", "ts_pct", "ast_to_ratio", "pace",
        "oreb_pct", "dreb_pct", "tm_tov_pct",
        "team_avg_pie",
    ]

    if mode == "preseason" or not current_partial_stats:
        for col in blend_cols:
            base[col] = prior.get(col)
    else:
        w_curr = games_played / 82
        w_prior = 1.0 - w_curr
        for col in blend_cols:
            curr_val = current_partial_stats.get(col)
            prior_val = prior.get(col)
            if curr_val is not None and prior_val is not None:
                base[col] = curr_val * w_curr + prior_val * w_prior
            else:
                base[col] = prior_val
        base["avg_games_played"] = current_partial_stats.get("avg_games_played", prior.get("avg_games_played"))

    return pd.DataFrame([base])


# ── Top-level orchestrator ────────────────────────────────────────────────────

def transform_data(raw_data: dict) -> dict:
    """
    Full transform pipeline: normalize → aggregate players → compute turnover
    → join → add lag features → build ML feature table.

    Returns dict with keys: "team_stats", "player_stats", "ml_features"
    """
    team_df = normalize_team_stats(raw_data["team_stats"])
    player_df = normalize_player_stats(raw_data["player_stats"])

    player_agg_df = aggregate_player_features(player_df)
    turnover_df = compute_roster_turnover(player_df)
    ml_features_df = build_ml_features(team_df, player_agg_df, turnover_df)

    print(
        f"Transform complete: {len(team_df)} team rows, "
        f"{len(player_df)} player rows, {len(ml_features_df)} ML feature rows"
    )
    return {
        "team_stats": team_df,
        "player_stats": player_df,
        "ml_features": ml_features_df,
    }
