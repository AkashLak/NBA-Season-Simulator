"""
NBA Win Predictor & Roster Simulator — Airflow DAGs

Two DAGs with separate responsibilities:

DAG 1: nba_data_pipeline  (weekly, Monday 9am)
  Ingest current season → transform → load to PostgreSQL →
  update predictions for all 30 teams via league simulation.
  Fast: one LeagueGameLog API call per run.

DAG 2: nba_retrain_pipeline  (1st of each month + manual trigger)
  Check whether new season data exists → retrain season model →
  retrain game model → evaluate both → write last_trained_season.json.
  Skips via ShortCircuitOperator if no new seasons since last training.

All heavy imports happen inside task functions (never at module level) so
Airflow's DAG serializer does not attempt to import NBA project code on
the scheduler. The scheduler only sees the DAG structure; workers import
the actual modules at run time.
"""

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

# ── Shared defaults ────────────────────────────────────────────────────────────

_DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

_PROCESSED_DIR = "processed"
_LAST_TRAINED_PATH = os.path.join(_PROCESSED_DIR, "last_trained_season.json")


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 1 — nba_data_pipeline  (weekly)
# ═══════════════════════════════════════════════════════════════════════════════

def _ingest_season(**context):
    """Fetch current-season team + player stats from nba_api."""
    from dotenv import load_dotenv
    from etl.ingest import ingest_data, SEASONS
    from etl.load import save_to_parquet

    load_dotenv()
    os.makedirs(_PROCESSED_DIR, exist_ok=True)

    current_season = SEASONS[-1]
    print(f"Ingesting season stats for {current_season}...")
    raw = ingest_data([current_season])

    save_to_parquet(raw["team_stats"],   f"{_PROCESSED_DIR}/raw_team_current.parquet")
    save_to_parquet(raw["player_stats"], f"{_PROCESSED_DIR}/raw_player_current.parquet")
    print(f"Season ingest complete: {len(raw['team_stats'])} team rows, "
          f"{len(raw['player_stats'])} player rows")


def _ingest_games(**context):
    """Fetch current-season game logs from nba_api."""
    from dotenv import load_dotenv
    from etl.ingest_games import ingest_games, SEASONS
    from etl.load import save_to_parquet

    load_dotenv()
    current_season = SEASONS[-1]
    print(f"Ingesting game logs for {current_season}...")
    game_logs = ingest_games([current_season])

    save_to_parquet(game_logs, f"{_PROCESSED_DIR}/raw_game_logs_current.parquet")
    print(f"Game ingest complete: {len(game_logs)} rows")


def _transform_season(**context):
    """Transform season-level data: normalize, aggregate players, compute lags."""
    import pandas as pd
    from etl.transform import transform_data
    from etl.load import save_to_parquet

    team_df   = pd.read_parquet(f"{_PROCESSED_DIR}/raw_team_current.parquet")
    player_df = pd.read_parquet(f"{_PROCESSED_DIR}/raw_player_current.parquet")

    transformed = transform_data({"team_stats": team_df, "player_stats": player_df})
    save_to_parquet(transformed["team_stats"],   f"{_PROCESSED_DIR}/team_stats_current.parquet")
    save_to_parquet(transformed["player_stats"], f"{_PROCESSED_DIR}/player_stats_current.parquet")
    save_to_parquet(transformed["ml_features"],  f"{_PROCESSED_DIR}/ml_features_current.parquet")
    print("Season transform complete")


def _transform_games(**context):
    """Compute rolling game features for current-season game logs."""
    import pandas as pd
    from etl.transform_games import transform_games

    game_logs = pd.read_parquet(f"{_PROCESSED_DIR}/raw_game_logs_current.parquet")
    ml_df     = pd.read_parquet(f"{_PROCESSED_DIR}/ml_features.parquet")

    game_features = transform_games(game_logs, ml_df)
    game_features.to_parquet(f"{_PROCESSED_DIR}/game_features_current.parquet", index=False)
    print(f"Game transform complete: {len(game_features)} rows")


def _load_all(**context):
    """Upsert all transformed data to PostgreSQL and refresh Parquet snapshots."""
    import pandas as pd
    from dotenv import load_dotenv
    from etl.load import get_engine, upsert_to_postgres, save_to_parquet
    from etl.transform import normalize_team_stats, normalize_player_stats

    load_dotenv()
    engine = get_engine()

    # Season-level tables
    team_df   = normalize_team_stats(
        pd.read_parquet(f"{_PROCESSED_DIR}/raw_team_current.parquet"))
    player_df = normalize_player_stats(
        pd.read_parquet(f"{_PROCESSED_DIR}/raw_player_current.parquet"))
    ml_df     = pd.read_parquet(f"{_PROCESSED_DIR}/ml_features_current.parquet")

    upsert_to_postgres(team_df,   "nba_team_season_stats",  engine,
                       conflict_cols=["team_id", "season"])
    upsert_to_postgres(player_df, "nba_player_season_stats", engine,
                       conflict_cols=["player_id", "team_id", "season_year"])
    upsert_to_postgres(ml_df,     "ml_training_features",   engine,
                       conflict_cols=["team_id", "season_year"])

    # Game-level table
    game_df = pd.read_parquet(f"{_PROCESSED_DIR}/game_features_current.parquet")
    upsert_to_postgres(game_df, "nba_game_features", engine,
                       conflict_cols=["game_id", "team_id"])

    # Refresh Parquet fallback snapshots
    save_to_parquet(ml_df,   f"{_PROCESSED_DIR}/ml_features.parquet")
    save_to_parquet(game_df, f"{_PROCESSED_DIR}/game_features.parquet")

    print("All data loaded to PostgreSQL and Parquet snapshots refreshed")


def _update_predictions(**context):
    """
    Simulate the current season for all 30 teams and append results to nba_predictions.

    Calls simulate_league() once — full league, all teams simultaneously.
    Each run appends one row per team (UNIQUE on team_id, season_year, date, model_version).
    """
    import joblib
    import pandas as pd
    from dotenv import load_dotenv
    from sqlalchemy import text

    from etl.ingest_games import SEASONS, season_to_year
    from etl.load import get_engine
    from models.simulate_season import simulate_league

    load_dotenv()
    engine = get_engine()

    # Predict for the season currently being played
    current_season = SEASONS[-1]
    season_year    = season_to_year(current_season)

    game_model = joblib.load("models/game_win_model.pkl")

    # Read model version from report for tracking
    report_path = "models/game_model_report.json"
    model_version = "game_sim_v1"
    rmse_conf = 5.0
    if os.path.exists(report_path):
        with open(report_path) as f:
            rpt = json.load(f)
        model_version = f"game_{rpt.get('winner', 'unknown')}_{rpt.get('trained_at', '')[:10]}"
        rmse_conf = rpt.get("winner_metrics", {}).get("holdout_logloss", 5.0)

    print(f"Simulating season {season_year} for all teams (model: {model_version})...")
    results = simulate_league(season_year, game_model, engine)

    rows = []
    for team_id, res in results.items():
        rows.append({
            "team_id":         team_id,
            "season_year":     season_year,
            "predicted_wins":  res["expected_wins"],
            "conf_interval":   rmse_conf,
            "playoff_prob":    None,    # game model doesn't output this directly
            "prediction_mode": "simulation",
            "model_run_id":    None,
            "model_version":   model_version,
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_sql("nba_predictions", engine, if_exists="append", index=False)
    print(f"Predictions updated for {len(rows)} teams (season {season_year})")


with DAG(
    dag_id="nba_data_pipeline",
    description="Weekly NBA data ingestion and prediction update",
    default_args=_DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="0 9 * * MON",
    catchup=False,
    tags=["nba", "etl", "weekly"],
) as data_dag:

    t_ingest_season  = PythonOperator(task_id="ingest_season",  python_callable=_ingest_season)
    t_ingest_games   = PythonOperator(task_id="ingest_games",   python_callable=_ingest_games)
    t_transform_season = PythonOperator(task_id="transform_season", python_callable=_transform_season)
    t_transform_games  = PythonOperator(task_id="transform_games",  python_callable=_transform_games)
    t_load_all       = PythonOperator(task_id="load_all",       python_callable=_load_all)
    t_update_preds   = PythonOperator(task_id="update_predictions", python_callable=_update_predictions)

    # Ingest both streams in parallel, then transform, then load, then predict
    [t_ingest_season, t_ingest_games] >> t_transform_season
    t_transform_season >> t_transform_games
    t_transform_games  >> t_load_all
    t_load_all         >> t_update_preds


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 2 — nba_retrain_pipeline  (monthly + manual)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_should_retrain(**context) -> bool:
    """
    Return True if new season data exists since the last training run.

    Reads last_trained_season.json to find the season_year we trained on.
    Queries nba_game_features for the current max season_year.
    Skips retrain (returns False) if max_season <= last_trained.

    ShortCircuitOperator will skip all downstream tasks on False.
    """
    from dotenv import load_dotenv
    from sqlalchemy import text
    from etl.load import get_engine

    load_dotenv()

    last_trained_year = 0
    if os.path.exists(_LAST_TRAINED_PATH):
        with open(_LAST_TRAINED_PATH) as f:
            last_trained_year = json.load(f).get("season_year", 0)

    try:
        engine = get_engine()
        with engine.connect() as conn:
            max_season = conn.execute(
                text("SELECT MAX(season_year) FROM nba_game_features")
            ).scalar() or 0
    except Exception as e:
        print(f"Could not query DB ({e}). Proceeding with retrain.")
        return True

    should_retrain = int(max_season) > int(last_trained_year)
    if should_retrain:
        print(f"New seasons detected (max={max_season}, last_trained={last_trained_year}). Retraining.")
    else:
        print(f"No new seasons (max={max_season} == last_trained={last_trained_year}). Skipping retrain.")
    return should_retrain


def _retrain_season_model(**context):
    """Retrain the season-level wins regression model."""
    from dotenv import load_dotenv
    from models.train_model import run_training

    load_dotenv()
    print("Retraining season-level model...")
    report = run_training()
    context["ti"].xcom_push(key="season_report", value=report)
    print(f"Season model retrained. Winner: {report['winner']}")


def _retrain_game_model(**context):
    """Retrain the game-level win probability classifier."""
    from dotenv import load_dotenv
    from models.train_game_model import run_game_training

    load_dotenv()
    print("Retraining game-level model...")
    report = run_game_training()
    context["ti"].xcom_push(key="game_report", value=report)
    print(f"Game model retrained. Winner: {report['winner']}")


def _evaluate_and_gate(**context):
    """
    Gate: both models must pass quality thresholds.
    Season model: R² > 0.75 AND RMSE < 6.0
    Game model:   LogLoss < 0.65 AND AUC > 0.62

    On success, write last_trained_season.json so future runs skip retrain
    until new data arrives.
    """
    from dotenv import load_dotenv
    from sqlalchemy import text
    from etl.load import get_engine

    load_dotenv()

    ti = context["ti"]
    season_report = ti.xcom_pull(task_ids="retrain_season_model", key="season_report")
    game_report   = ti.xcom_pull(task_ids="retrain_game_model",   key="game_report")

    # Season model gate
    s_winner = season_report["winner"]
    s_metrics = season_report["results"][s_winner]
    season_passed = s_metrics["r2"] > 0.75 and s_metrics["rmse"] < 6.0
    print(
        f"Season model ({s_winner}): R²={s_metrics['r2']:.4f}, "
        f"RMSE={s_metrics['rmse']:.4f} → {'PASSED' if season_passed else 'FAILED'}"
    )

    # Game model gate
    g_winner  = game_report["winner"]
    g_metrics = game_report["winner_metrics"]
    game_passed = g_metrics["holdout_logloss"] < 0.65 and g_metrics["holdout_auc"] > 0.62
    print(
        f"Game model ({g_winner}): LogLoss={g_metrics['holdout_logloss']:.4f}, "
        f"AUC={g_metrics['holdout_auc']:.4f} → {'PASSED' if game_passed else 'FAILED'}"
    )

    if not season_passed or not game_passed:
        raise RuntimeError(
            "Quality gate FAILED — models not saved. "
            f"Season: R²={s_metrics['r2']:.4f} RMSE={s_metrics['rmse']:.4f} | "
            f"Game: LogLoss={g_metrics['holdout_logloss']:.4f} AUC={g_metrics['holdout_auc']:.4f}"
        )

    # Write last_trained_season.json so future check_should_retrain skips
    try:
        engine = get_engine()
        with engine.connect() as conn:
            max_season = conn.execute(
                text("SELECT MAX(season_year) FROM nba_game_features")
            ).scalar() or 0
    except Exception:
        max_season = 0

    os.makedirs(_PROCESSED_DIR, exist_ok=True)
    with open(_LAST_TRAINED_PATH, "w") as f:
        json.dump({"season_year": int(max_season), "trained_at": datetime.now().isoformat()}, f)

    print(f"Both models passed. last_trained_season.json updated to {max_season}.")


with DAG(
    dag_id="nba_retrain_pipeline",
    description="Monthly model retraining — skips if no new data",
    default_args=_DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="0 9 1 * *",      # 1st of each month at 9am
    catchup=False,
    tags=["nba", "ml", "monthly"],
) as retrain_dag:

    t_check = ShortCircuitOperator(
        task_id="check_should_retrain",
        python_callable=_check_should_retrain,
    )
    t_retrain_season = PythonOperator(
        task_id="retrain_season_model",
        python_callable=_retrain_season_model,
    )
    t_retrain_game = PythonOperator(
        task_id="retrain_game_model",
        python_callable=_retrain_game_model,
    )
    t_gate = PythonOperator(
        task_id="evaluate_and_gate",
        python_callable=_evaluate_and_gate,
    )

    # Retrain both models in parallel, then gate
    t_check >> [t_retrain_season, t_retrain_game] >> t_gate
