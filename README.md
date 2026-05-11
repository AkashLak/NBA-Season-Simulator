# NBA Win Predictor & Roster Simulator

A full-stack sports analytics platform that forecasts NBA season win totals, simulates the impact of roster moves, and explains model decisions using SHAP. Backed by 25 years of NBA data, two ML models, and a league-wide game simulation engine.

## What it does

Select any team and season. The app:

1. **Forecasts season wins** using a Random Forest trained on prior-season team and roster stats (R² = 0.70, RMSE = 6.3 wins vs. a naive baseline of 10.0)
2. **Simulates win probability for every game** using a Logistic Regression classifier trained on 60K+ game rows (AUC = 0.68), then sums expected wins across the full 82-game schedule
3. **Lets you swap players** on any roster and re-runs the full league simulation to show the win impact
4. **Explains predictions** with SHAP waterfall charts showing which features drove each game's win probability
5. **Visualizes 25 years of franchise history** with trend charts covering efficiency ratings, year-over-year win deltas, and roster balance over time
6. **Tracks model health** with CV overfitting gaps, learning curves, and MLflow experiment runs

## Stack

| Layer | Technology |
|---|---|
| App | Python + Streamlit (6 pages) |
| Season model | Random Forest (sklearn), wins regression |
| Game model | Logistic Regression (sklearn), win probability |
| Simulation | Custom league-wide engine (`simulate_season.py`) |
| Explainability | SHAP waterfall + global feature importance |
| ETL | nba_api, PostgreSQL, Parquet fallback |
| Orchestration | Apache Airflow (2 DAGs: weekly ingest + monthly retrain) |
| Experiment tracking | MLflow |
| Database | PostgreSQL 15 (Docker) |
| Infrastructure | Docker Compose |

## Pages

| Page | Description |
|---|---|
| Season Forecast | Preseason win total + playoff probability for any team/season |
| Roster Simulator | Swap players, see the win delta from a full league re-simulation |
| SHAP Explainer | Per-game SHAP waterfall with win probability breakdown |
| Historical Performance | 25-year trend charts: efficiency ratings, YoY deltas, roster balance |
| Team & Roster Dashboard | Season metrics, injury risk highlights, roster table |
| Model Performance | Both model cards, CV results, learning curve, MLflow runs |

## Model Performance

**Season model** (Random Forest, 22 features, 681 training rows):

| Metric | Value |
|---|---|
| Test R² | 0.70 |
| Test RMSE | 6.3 wins |
| Naive baseline RMSE | 10.0 wins |
| Improvement | +3.7 wins |
| Playoff classifier AUC | 0.86 |

**Game model** (Logistic Regression, 14 features, 54K training rows):

| Metric | Value |
|---|---|
| Holdout AUC | 0.68 |
| Holdout Log Loss | 0.644 |
| Holdout Accuracy | 62.5% |
| Home-court baseline AUC | 0.56 |

Features are a hybrid of **static** (prior-season net/off/def rating) and **dynamic** (5- and 10-game rolling win%) signals. Rolling features use `shift(1)` to prevent leakage; the simulation updates them after every game using expected win probabilities.

## Project Structure

```
LakersSeasonForecast/
├── app/
│   ├── app.py                    # Streamlit entry point + navigation
│   ├── shared.py                 # Cached data loaders, team selector, constants
│   └── pages/
│       ├── p1_forecast.py        # Season win forecast + playoff probability
│       ├── p2_simulator.py       # Roster swap simulator
│       ├── p3_shap.py            # SHAP waterfall explainer
│       ├── p4_history.py         # Historical trend charts
│       ├── p5_team_dashboard.py  # Team stats + roster table
│       └── p6_model_perf.py      # Model cards + MLflow runs
├── etl/
│   ├── ingest.py                 # Season-level team + player stats (nba_api)
│   ├── ingest_games.py           # Game logs via LeagueGameLog (nba_api)
│   ├── transform.py              # Normalize, aggregate, lag features, SOS
│   ├── transform_games.py        # Rolling win%, rest days, prior-season join
│   ├── load.py                   # PostgreSQL upserts + Parquet snapshots
│   └── run_etl.py                # CLI entry point (--fast flag for dev)
├── models/
│   ├── data_prep.py              # Feature definitions, chronological split
│   ├── train_model.py            # Season model: CV + winner selection + MLflow
│   ├── train_game_model.py       # Game model: CV + winner selection + MLflow
│   ├── simulate_season.py        # League-wide simulation engine
│   ├── evaluate.py               # Shared evaluation helpers + quality gate
│   └── shap_analysis.py          # SHAP artifacts for both models
├── dags/
│   └── etl_dag.py                # DAG 1: weekly ingest | DAG 2: monthly retrain
├── Testing/
│   ├── test_etl.py               # Lag feature correctness, leakage checks
│   ├── test_games.py             # Rolling feature shift(1), rest days
│   ├── test_simulation.py        # Win total consistency, override isolation
│   ├── test_model.py             # Quality gates, SHAP top features
│   └── test_integration.py       # End-to-end pipeline smoke test
├── sql/
│   └── init.sql                  # Schema for all 5 PostgreSQL tables
├── docker-compose.yml            # postgres + mlflow + airflow + streamlit
├── Dockerfile                    # Streamlit app image
└── Dockerfile.airflow            # Airflow scheduler + webserver image
```

## Architecture

```
nba_api
  └─► etl/ingest.py + ingest_games.py
        └─► etl/transform.py + transform_games.py
              └─► etl/load.py ──► PostgreSQL (primary)
                               └─► processed/*.parquet (Streamlit Cloud fallback)

processed/ml_features.parquet + game_features.parquet
  └─► models/train_model.py      ──► best_wins_model.pkl  (season wins regression)
  └─► models/train_game_model.py ──► game_win_model.pkl   (game win probability)

game_win_model.pkl
  └─► models/simulate_season.py
        └─► simulate_league(season_year)
              processes all ~1,230 games chronologically for all 30 teams
              updating rolling win% after every game
              └─► expected wins per team

app/shared.py ──► all 6 Streamlit pages
```

**Key design decisions:**

- **Lag-only season features:** all 22 inputs to the season model are knowable before a season tips off (prior-season stats + current roster composition), making forecasts genuinely pre-season.
- **Full league simulation:** game-level rolling features depend on cross-team games, so you cannot simulate one team in isolation. `simulate_league()` processes the complete schedule, updating all 30 teams simultaneously.
- **Expected-value simulation:** rather than Monte Carlo sampling, each game contributes `win_prob` to the home team's win total and `1 - win_prob` to the away team's. This gives a smooth, deterministic expected wins figure.
- **PostgreSQL to Parquet fallback:** all data loaders try the database first and fall back to Parquet snapshots, so the app runs on Streamlit Community Cloud without a database.

## Running Locally

**Prerequisites:** Docker Desktop, Python 3.11+

```bash
# 1. Clone and set up environment
git clone https://github.com/AkashLak/LakersSeasonForecast.git
cd LakersSeasonForecast
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in DATABASE_URL if needed

# 2. Start infrastructure
docker compose up postgres mlflow -d

# 3. Ingest data (use --fast for last 5 seasons, ~7 min; omit for all 25 seasons, ~20 min)
python -m etl.run_etl --fast
python -m etl.ingest_games --fast

# 4. Train models
python -m models.train_model
python -m models.train_game_model

# 5. Run the app
streamlit run app/app.py
```

Open http://localhost:8501. MLflow UI at http://localhost:5001.

## Running Tests

```bash
pytest Testing/ -v                    # all tests
pytest Testing/test_games.py -v       # rolling feature + leakage tests
pytest Testing/test_simulation.py -v  # simulation correctness
```

## Airflow DAGs

Start Airflow with:

```bash
docker compose up airflow-db airflow-init -d && sleep 10
docker compose up airflow-scheduler airflow-webserver -d
```

Open http://localhost:8080 (admin / admin).

| DAG | Schedule | What it does |
|---|---|---|
| `nba_data_pipeline` | Weekly, Monday 9am | Ingest current season, transform, load to PostgreSQL, update predictions for all 30 teams |
| `nba_retrain_pipeline` | Monthly, 1st of month | Check for new seasons, retrain both models, quality gate, write `last_trained_season.json` |

The retrain pipeline uses a `ShortCircuitOperator` to skip retraining if no new seasons have arrived since the last run.

## Environment Variables

Copy `.env.example` to `.env`:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (default: `postgresql://lakers:lakers@localhost:5433/lakersdb`) |
| `MLFLOW_TRACKING_URI` | MLflow server URL (default: `http://localhost:5001`) |
| `AIRFLOW_FERNET_KEY` | Fernet key for Airflow credential encryption |
| `AIRFLOW_ADMIN_PASSWORD` | Airflow UI admin password |

## Database Schema

| Table | Rows | Description |
|---|---|---|
| `nba_team_season_stats` | ~750 | Per-team season stats (ratings, pace, shooting) |
| `nba_player_season_stats` | ~18K | Per-player season stats (PIE, minutes, age) |
| `ml_training_features` | ~750 | Engineered lag + roster features used for training |
| `nba_game_features` | ~68K | Per-game rolling features for the game model |
| `nba_predictions` | growing | Historical win forecasts per team/season, appended weekly |

## License

MIT
