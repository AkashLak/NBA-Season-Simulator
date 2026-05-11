"""
Game-level win probability model.

Trains a binary classifier (win/loss) on per-game features. Winner is selected
by lowest CV log loss across three candidates using TimeSeriesSplit — no random
shuffling, chronological order is preserved throughout.

Quality gate: holdout log loss < 0.65 AND AUC > 0.62.
  - Random baseline log loss = 0.693 (ln 2)
  - Home-court-only AUC ≈ 0.58
  - Both thresholds require the model to be meaningfully better than trivial baselines.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from etl.transform_games import GAME_FEATURE_COLS, GAME_TARGET
from models.shap_analysis import save_shap_artifacts

load_dotenv()

MODEL_DIR = os.path.dirname(__file__)
PROCESSED_DIR = "processed"
GAME_MODEL_PATH = os.path.join(MODEL_DIR, "game_win_model.pkl")
GAME_REPORT_PATH = os.path.join(MODEL_DIR, "game_model_report.json")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_game_features() -> pd.DataFrame:
    """Load game features — PostgreSQL first, processed/game_features.parquet fallback."""
    try:
        from etl.load import get_engine
        engine = get_engine()
        df = pd.read_sql(
            "SELECT * FROM nba_game_features ORDER BY game_date",
            engine,
        )
        print(f"Loaded {len(df)} game rows from PostgreSQL")
        return df
    except Exception as e:
        print(f"PostgreSQL unavailable ({e}), falling back to Parquet")

    path = os.path.join(PROCESSED_DIR, "game_features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No game data: PostgreSQL failed and {path} not found. "
            "Run etl/ingest_games.py first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} game rows from {path}")
    return df


# ── Candidate definitions ─────────────────────────────────────────────────────

def _build_candidates() -> dict:
    """
    Three classification candidates trained on identical TimeSeriesSplit(n_splits=5).
    LogisticRegression uses lbfgs solver (calibrated probabilities, no log_loss → inf).
    XGBoost uses logloss eval metric directly.
    """
    return {
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────

def _cross_validate_candidates(candidates: dict, X_train, y_train) -> dict:
    """
    Run TimeSeriesSplit(n_splits=5) on each candidate.
    Primary metric: neg_log_loss (lower = better calibrated probabilities).
    Also records AUC, train log loss, and overfitting gap.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {}

    for name, model in candidates.items():
        cv_out = cross_validate(
            model, X_train, y_train,
            cv=tscv,
            scoring=["neg_log_loss", "roc_auc"],
            return_train_score=True,
            n_jobs=-1,
        )
        cv_logloss = -cv_out["test_neg_log_loss"].mean()
        cv_auc = cv_out["test_roc_auc"].mean()
        train_logloss = -cv_out["train_neg_log_loss"].mean()

        cv_results[name] = {
            "cv_logloss":       round(float(cv_logloss), 4),
            "cv_logloss_std":   round(float(cv_out["test_neg_log_loss"].std()), 4),
            "cv_auc":           round(float(cv_auc), 4),
            "cv_train_logloss": round(float(train_logloss), 4),
            "overfitting_gap":  round(float(train_logloss - cv_logloss), 4),
        }
        print(
            f"  {name:15s}  LogLoss={cv_logloss:.4f} ± {cv_out['test_neg_log_loss'].std():.4f}"
            f"  AUC={cv_auc:.4f}  gap={train_logloss - cv_logloss:.4f}"
        )

    return cv_results


# ── Winner selection ──────────────────────────────────────────────────────────

def _select_winner(
    cv_results: dict,
    X_train, y_train,
    X_test, y_test,
) -> tuple:
    """
    Fit all candidates on the full training set, evaluate on holdout.
    Winner = lowest CV log loss. Tie-break within 0.005 log loss: prefer
    Logistic > RF > XGBoost for interpretability (linear model is easiest to explain).

    Returns (winner_name, fitted_winner_model, holdout_metrics_dict, all_holdout_metrics).
    """
    TIEBREAK_THRESHOLD = 0.005
    TIEBREAK_ORDER = ["logistic", "random_forest", "xgboost"]

    # Sort by CV log loss ascending — lower is better
    sorted_names = sorted(cv_results, key=lambda n: cv_results[n]["cv_logloss"])
    best_name = sorted_names[0]
    for name in sorted_names[1:]:
        delta = cv_results[name]["cv_logloss"] - cv_results[best_name]["cv_logloss"]
        if delta <= TIEBREAK_THRESHOLD:
            if TIEBREAK_ORDER.index(name) < TIEBREAK_ORDER.index(best_name):
                print(
                    f"  Tie-break: {name} preferred over {best_name} "
                    f"(LogLoss delta={delta:.4f} ≤ {TIEBREAK_THRESHOLD})"
                )
                best_name = name

    print(f"\nFitting all candidates on full training set...")
    candidates = _build_candidates()  # fresh (unfitted) instances for final fit
    holdout_metrics = {}
    fitted_models = {}

    for name, model in candidates.items():
        print(f"  Fitting {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-15, 1 - 1e-15)
        holdout_metrics[name] = {
            "holdout_logloss":  round(float(log_loss(y_test, y_probs)), 4),
            "holdout_auc":      round(float(roc_auc_score(y_test, y_probs)), 4),
            "holdout_accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 4),
        }
        print(
            f"    LogLoss={holdout_metrics[name]['holdout_logloss']:.4f}"
            f"  AUC={holdout_metrics[name]['holdout_auc']:.4f}"
            f"  Acc={holdout_metrics[name]['holdout_accuracy']:.4f}"
        )

    winner_model = fitted_models[best_name]
    return best_name, winner_model, holdout_metrics[best_name], holdout_metrics


# ── Baseline comparison ───────────────────────────────────────────────────────

def _compute_home_court_baseline(X_test, y_test) -> dict:
    """
    Naive baseline: always predict the home team wins (home_flag = 1).
    Home teams win ~60% of games historically — this is the floor the model must beat.
    """
    home_probs = X_test["home_flag"].clip(0.01, 0.99).values
    baseline_logloss = float(log_loss(y_test, home_probs))
    baseline_auc = float(roc_auc_score(y_test, home_probs))
    baseline_acc = float(accuracy_score(y_test, (home_probs >= 0.5).astype(int)))
    return {
        "baseline_logloss":  round(baseline_logloss, 4),
        "baseline_auc":      round(baseline_auc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
    }


# ── Quality gate ──────────────────────────────────────────────────────────────

def _check_game_quality_gate(metrics: dict) -> bool:
    """
    Hard gate: log loss < 0.65 AND AUC > 0.62.
    Random baseline log loss = 0.693. Home-court-only AUC ≈ 0.58.
    Both thresholds ensure the model is meaningfully better than trivial baselines.
    """
    passed = metrics["holdout_logloss"] < 0.65 and metrics["holdout_auc"] > 0.62
    status = "PASSED" if passed else "FAILED"
    print(
        f"Game model quality gate {status}: "
        f"LogLoss={metrics['holdout_logloss']:.4f} (need <0.65), "
        f"AUC={metrics['holdout_auc']:.4f} (need >0.62)"
    )
    return passed


# ── MLflow logging ────────────────────────────────────────────────────────────

def _log_to_mlflow(
    winner_name: str,
    winner_model,
    cv_results: dict,
    winner_holdout: dict,
    baseline: dict,
    n_train: int,
    n_test: int,
):
    """Log training run to MLflow. Prints warning and continues if server is down."""
    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("nba-game-predictor")

        with mlflow.start_run(run_name=f"game_{winner_name}_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.set_tag("model_winner", winner_name)
            mlflow.log_param("n_training_samples", n_train)
            mlflow.log_param("n_test_samples", n_test)

            # Log hyperparameters — unwrap Pipeline if needed
            raw_model = (
                winner_model.named_steps["model"]
                if hasattr(winner_model, "named_steps")
                else winner_model
            )
            mlflow.log_params(raw_model.get_params())

            # CV metrics per candidate
            for name, metrics in cv_results.items():
                for k, v in metrics.items():
                    mlflow.log_metric(f"{name}_{k}", v)

            # Winner holdout metrics
            mlflow.log_metric("winner_logloss", winner_holdout["holdout_logloss"])
            mlflow.log_metric("winner_auc", winner_holdout["holdout_auc"])
            mlflow.log_metric("winner_accuracy", winner_holdout["holdout_accuracy"])

            # Baseline comparison
            mlflow.log_metric("baseline_logloss", baseline["baseline_logloss"])
            mlflow.log_metric("baseline_auc", baseline["baseline_auc"])
            mlflow.log_metric(
                "improvement_over_baseline",
                round(baseline["baseline_logloss"] - winner_holdout["holdout_logloss"], 4),
            )

            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run logged: {run_id}")
            return run_id

    except Exception as e:
        print(f"MLflow logging skipped ({e}) — continuing without tracking")
        return None


# ── Main training entry point ─────────────────────────────────────────────────

def run_game_training() -> dict:
    """
    Full game model training pipeline:
      1. Load game features → chronological train/holdout split (last 3 seasons)
      2. TimeSeriesSplit(n_splits=5) CV on training set for all 3 candidates
      3. Select winner by CV log loss (with tie-break for interpretability)
      4. Fit all candidates on full training set, evaluate on holdout
      5. Hard quality gate: LogLoss < 0.65 AND AUC > 0.62
      6. Generate SHAP artifacts
      7. Log to MLflow (graceful fallback)
      8. Save model + selection report

    Returns the game model report dict.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading game features...")
    df = load_game_features()

    # Sort chronologically — required for TimeSeriesSplit to be meaningful
    df = df.sort_values("game_date").reset_index(drop=True)

    available = [c for c in GAME_FEATURE_COLS if c in df.columns]
    missing = [c for c in GAME_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} feature columns missing from data: {missing}")

    X = df[available].copy()
    y = df[GAME_TARGET].copy()

    # Drop rows with NaN features or missing target
    valid_mask = X.notna().all(axis=1) & y.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN features (first season per team, expected)")
    X, y = X[valid_mask].reset_index(drop=True), y[valid_mask].reset_index(drop=True)
    df_valid = df[valid_mask].reset_index(drop=True)

    # Chronological split: hold out last 3 seasons (~7,380 games)
    season_years = df_valid["season_year"]
    cutoff = sorted(season_years.unique())[-3]
    train_mask = season_years < cutoff
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    print(f"Train: {len(X_train)} games ({season_years[train_mask].min()}–{cutoff - 1})")
    print(f"Holdout: {len(X_test)} games ({cutoff}–{season_years.max()})")

    # ── Step 1: Cross-validation ──────────────────────────────────────────────
    print("\nRunning TimeSeriesSplit(n_splits=5) cross-validation...")
    candidates = _build_candidates()
    cv_results = _cross_validate_candidates(candidates, X_train, y_train)

    # ── Step 2: Select winner + fit on full training set ──────────────────────
    print(f"\nFitting candidates and evaluating on holdout ({len(X_test)} games)...")
    winner_name, winner_model, winner_holdout, all_holdout = _select_winner(
        cv_results, X_train, y_train, X_test, y_test
    )
    print(f"\nWinner: {winner_name.upper()}")

    # ── Step 3: Baseline comparison ───────────────────────────────────────────
    baseline = _compute_home_court_baseline(X_test, y_test)
    improvement = baseline["baseline_logloss"] - winner_holdout["holdout_logloss"]
    print(
        f"\nBaseline (home court): LogLoss={baseline['baseline_logloss']:.4f} "
        f"AUC={baseline['baseline_auc']:.4f}"
    )
    print(
        f"Winner ({winner_name}): LogLoss={winner_holdout['holdout_logloss']:.4f} "
        f"AUC={winner_holdout['holdout_auc']:.4f}"
    )
    print(f"Improvement over baseline: {improvement:+.4f} log loss")

    baseline_warning = None
    if winner_holdout["holdout_logloss"] >= baseline["baseline_logloss"]:
        baseline_warning = (
            f"Game model ({winner_name}) does not improve on home-court baseline. "
            f"Consider adding more features before deploying."
        )
        print(f"WARNING: {baseline_warning}")

    # ── Step 4: Quality gate ──────────────────────────────────────────────────
    gate_passed = _check_game_quality_gate(winner_holdout)
    if not gate_passed:
        raise RuntimeError(
            f"Game model quality gate FAILED for {winner_name}: "
            f"LogLoss={winner_holdout['holdout_logloss']:.4f}, "
            f"AUC={winner_holdout['holdout_auc']:.4f}. "
            "Revisit feature engineering before proceeding."
        )

    # ── Step 5: SHAP artifacts ────────────────────────────────────────────────
    print("\nGenerating SHAP artifacts...")
    try:
        shap_summary = save_shap_artifacts(
            winner_model, X_train, X_test,
            summary_path=os.path.join(PROCESSED_DIR, "game_shap_summary.parquet"),
            values_path=os.path.join(PROCESSED_DIR, "game_shap_values.npy"),
        )
        print("Top 5 features by SHAP importance:")
        print(shap_summary.head(5).to_string(index=False))
    except Exception as e:
        print(f"SHAP artifacts skipped ({e})")
        shap_summary = None

    # ── Step 6: MLflow ────────────────────────────────────────────────────────
    run_id = _log_to_mlflow(
        winner_name, winner_model, cv_results, winner_holdout, baseline,
        n_train=len(X_train), n_test=len(X_test),
    )

    # ── Step 7: Save model ────────────────────────────────────────────────────
    joblib.dump(winner_model, GAME_MODEL_PATH)
    print(f"\nModel saved: {GAME_MODEL_PATH}")

    # ── Step 8: Save report ───────────────────────────────────────────────────
    report = {
        "winner": winner_name,
        "trained_at": datetime.now().isoformat(),
        "feature_cols": available,
        "n_training_games": int(len(X_train)),
        "n_test_games": int(len(X_test)),
        "holdout_seasons": int(3),
        "gate_passed": gate_passed,
        "winner_metrics": winner_holdout,
        "baseline_metrics": baseline,
        "improvement_over_baseline": round(improvement, 4),
        "baseline_warning": baseline_warning,
        "all_holdout_metrics": all_holdout,
        "cv_results": cv_results,
        "mlflow_run_id": run_id,
        "model_path": GAME_MODEL_PATH,
    }

    with open(GAME_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {GAME_REPORT_PATH}")

    print("\n" + "=" * 60)
    print(f"Game model training complete. Winner: {winner_name.upper()}")
    print(
        f"  LogLoss={winner_holdout['holdout_logloss']:.4f}  "
        f"AUC={winner_holdout['holdout_auc']:.4f}  "
        f"Acc={winner_holdout['holdout_accuracy']:.4f}"
    )

    return report


def load_game_model():
    """Load the trained game model from disk."""
    if not os.path.exists(GAME_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained game model at {GAME_MODEL_PATH}. "
            "Run run_game_training() first."
        )
    return joblib.load(GAME_MODEL_PATH)


if __name__ == "__main__":
    run_game_training()
