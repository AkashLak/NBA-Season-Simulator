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

-- ── Prediction history ────────────────────────────────────────────────────────
-- One row per weekly DAG run per team. Allows Page 4 to show how the
-- projected win total evolved week by week throughout the season.
CREATE TABLE IF NOT EXISTS nba_predictions (
    id              SERIAL PRIMARY KEY,
    predicted_at    TIMESTAMP DEFAULT NOW(),
    team_id         INTEGER,
    season_year     INTEGER,
    predicted_wins  FLOAT,
    conf_interval   FLOAT,       -- ± RMSE of the winning model
    playoff_prob    FLOAT,
    prediction_mode VARCHAR(15),
    model_run_id    VARCHAR(100)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_team_stats_team_season ON nba_team_season_stats (team_id, season_year);
CREATE INDEX IF NOT EXISTS idx_player_stats_team_season ON nba_player_season_stats (team_id, season_year);
CREATE INDEX IF NOT EXISTS idx_ml_features_team_season ON ml_training_features (team_id, season_year);
CREATE INDEX IF NOT EXISTS idx_predictions_team_season ON nba_predictions (team_id, season_year);
