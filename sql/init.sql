-- NBA Win Predictor & Roster Simulator — PostgreSQL schema
-- Run automatically by Docker Compose on first startup via initdb.d

-- ── Raw team stats ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nba_team_season_stats (
    id            SERIAL PRIMARY KEY,
    team_id       INTEGER       NOT NULL,
    team_abbr     VARCHAR(5),
    team_name     VARCHAR(100),
    season        VARCHAR(10)   NOT NULL,   -- "2023-24"
    season_year   INTEGER       NOT NULL,   -- 2024
    games_played  INTEGER,
    wins          INTEGER,
    losses        INTEGER,
    win_pct       FLOAT,
    pts_pg        FLOAT,
    reb_pg        FLOAT,
    ast_pg        FLOAT,
    stl_pg        FLOAT,
    blk_pg        FLOAT,
    tov_pg        FLOAT,
    off_rating    FLOAT,
    def_rating    FLOAT,
    net_rating    FLOAT,
    pace          FLOAT,
    ts_pct        FLOAT,
    efg_pct       FLOAT,
    oreb_pct      FLOAT,
    dreb_pct      FLOAT,
    tm_tov_pct    FLOAT,
    ast_to_ratio  FLOAT,
    is_lakers     BOOLEAN,
    created_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (team_id, season)
);

-- ── Raw player stats ──────────────────────────────────────────────────────────
-- Uses PIE (Player Impact Estimate) — the NBA's official player impact metric,
-- available directly from nba_api. Analogous to Win Shares but sourced from
-- the official stats.nba.com API rather than Basketball Reference.
CREATE TABLE IF NOT EXISTS nba_player_season_stats (
    id           SERIAL PRIMARY KEY,
    player_id    INTEGER       NOT NULL,
    player_name  VARCHAR(100),
    team_id      INTEGER       NOT NULL,
    season_year  INTEGER       NOT NULL,
    games_played INTEGER,
    minutes_pg   FLOAT,
    pts_pg       FLOAT,
    pie          FLOAT,        -- Player Impact Estimate (NBA official metric)
    usg_pct      FLOAT,        -- Usage percentage
    ts_pct       FLOAT,
    net_rating   FLOAT,
    age          INTEGER,
    created_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE (player_id, team_id, season_year)
);

-- ── ML-ready feature table ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_training_features (
    id                   SERIAL PRIMARY KEY,
    team_id              INTEGER NOT NULL,
    season_year          INTEGER NOT NULL,
    -- team-level features
    pts_pg               FLOAT,
    off_rating           FLOAT,
    def_rating           FLOAT,
    net_rating           FLOAT,
    ts_pct               FLOAT,
    ast_to_ratio         FLOAT,
    pace                 FLOAT,
    oreb_pct             FLOAT,
    dreb_pct             FLOAT,
    tm_tov_pct           FLOAT,
    -- lag features (prior season) — computed with groupby(team_id).shift(1)
    prev_wins            FLOAT,
    prev_wins_normalized FLOAT,
    prev_off_rating      FLOAT,
    prev_def_rating      FLOAT,
    prev_net_rating      FLOAT,
    prev_pts_pg          FLOAT,
    prev_ts_pct          FLOAT,
    prev_ast_to_ratio    FLOAT,
    prev_pace            FLOAT,
    prev_oreb_pct        FLOAT,
    prev_dreb_pct        FLOAT,
    prev_tm_tov_pct      FLOAT,
    prev_team_avg_pie    FLOAT,
    prev_playoff_team    BOOLEAN,
    -- player-aggregated features
    team_avg_pie         FLOAT,  -- minutes-weighted avg PIE across roster
    team_avg_age         FLOAT,  -- minutes-weighted avg age
    roster_turnover_pct  FLOAT,  -- fraction of minutes from new players
    avg_games_played     FLOAT,  -- avg GP for top-8 min players (injury proxy)
    star_age_flag        BOOLEAN,
    std_dev_pie          FLOAT,  -- PIE variance across roster (balanced vs star-dependent)
    top_3_minutes_share  FLOAT,  -- fraction of team minutes from top-3 players
    prev_std_dev_pie     FLOAT,  -- lag version of std_dev_pie
    prev_top_3_minutes_share FLOAT, -- lag version of top_3_minutes_share
    -- metadata
    is_lakers            BOOLEAN,
    prediction_mode      VARCHAR(15),  -- "preseason" or "mid-season"
    -- targets: raw wins for app display, wins_normalized for model training
    wins                 INTEGER,      -- actual wins (may reflect shortened season)
    wins_normalized      FLOAT,        -- wins scaled to 82-game pace
    playoff_team         BOOLEAN,      -- based on wins_normalized >= 44
    created_at           TIMESTAMP DEFAULT NOW(),
    UNIQUE (team_id, season_year)
);

-- ── Game-level features ───────────────────────────────────────────────────────
-- One row per team per game (~2,460 rows per season × 25 seasons ≈ 61K rows).
-- Dynamic rolling features are computed from actual outcomes during training
-- and from expected probabilities (win_prob) during simulation.
CREATE TABLE IF NOT EXISTS nba_game_features (
    id                         SERIAL PRIMARY KEY,
    game_id                    VARCHAR(20)   NOT NULL,
    game_date                  DATE          NOT NULL,
    season_year                INTEGER       NOT NULL,
    team_id                    INTEGER       NOT NULL,
    opponent_id                INTEGER       NOT NULL,
    home_flag                  INTEGER       NOT NULL,  -- 1=home, 0=away
    win                        INTEGER,                 -- 1/0 (NULL for future games)
    -- dynamic rolling features (shift(1) applied during training)
    team_rolling_win_pct_5     FLOAT,
    team_rolling_win_pct_10    FLOAT,
    opp_rolling_win_pct_5      FLOAT,
    opp_rolling_win_pct_10     FLOAT,
    -- static prior-season quality baseline (from season_year - 1)
    team_prev_net_rating       FLOAT,
    team_prev_off_rating       FLOAT,
    team_prev_def_rating       FLOAT,
    opp_prev_net_rating        FLOAT,
    opp_prev_off_rating        FLOAT,
    opp_prev_def_rating        FLOAT,
    -- schedule-derived (always exact)
    rest_days                  INTEGER,
    opp_rest_days              INTEGER,
    days_into_season           INTEGER,
    created_at                 TIMESTAMP DEFAULT NOW(),
    UNIQUE (game_id, team_id)
);

-- ── Prediction history ────────────────────────────────────────────────────────
-- One row per weekly DAG run per team per model version. Allows Page 4 to show
-- how the projected win total evolved week by week throughout the season.
-- UNIQUE on (team_id, season_year, DATE(predicted_at), model_version) prevents
-- duplicate rows if the DAG runs more than once on the same day.
CREATE TABLE IF NOT EXISTS nba_predictions (
    id              SERIAL PRIMARY KEY,
    predicted_at    TIMESTAMP DEFAULT NOW(),
    team_id         INTEGER,
    season_year     INTEGER,
    predicted_wins  FLOAT,
    conf_interval   FLOAT,
    playoff_prob    FLOAT,
    prediction_mode VARCHAR(15),
    model_run_id    VARCHAR(100),
    model_version   VARCHAR(50)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_team_stats_team_season   ON nba_team_season_stats (team_id, season_year);
CREATE INDEX IF NOT EXISTS idx_player_stats_team_season ON nba_player_season_stats (team_id, season_year);
CREATE INDEX IF NOT EXISTS idx_ml_features_team_season  ON ml_training_features (team_id, season_year);
CREATE INDEX IF NOT EXISTS idx_predictions_team_season  ON nba_predictions (team_id, season_year);
-- Function-based unique index: one row per team per day per model version
CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique
    ON nba_predictions (team_id, season_year, model_version, DATE(predicted_at));
CREATE INDEX IF NOT EXISTS idx_game_features_team_date  ON nba_game_features (team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_game_features_season     ON nba_game_features (season_year);
CREATE INDEX IF NOT EXISTS idx_game_features_game       ON nba_game_features (game_id);
