import json
import os
from datetime import datetime

import joblib
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBClassifier

from models.data_prep import load_data, prepare_features, chronological_split
from models.evaluate import evaluate_model, check_quality_gate, write_eval_report
from models.shap_analysis import save_shap_artifacts

load_dotenv()

MODEL_DIR = os.path.dirname(__file__)
PROCESSED_DIR = "processed"
REPORT_PATH = os.path.join(MODEL_DIR, "model_selection_report.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_wins_model.pkl")
PLAYOFF_MODEL_PATH = os.path.join(MODEL_DIR, "playoff_classifier.pkl")


# ── Candidate model definitions ───────────────────────────────────────────────

def _build_candidates() -> dict:
    """
    Three regression candidates trained on identical TimeSeriesSplit(n_splits=5).
    Ridge is wrapped in a Pipeline with StandardScaler — linear models require
    feature scaling, tree models do not.
    """
    return {
        "xgboost": xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────

def _cross_validate(candidates: dict, X_train, y_train) -> dict:
    """
    Run TimeSeriesSplit(n_splits=5) CV on each candidate.
    Returns dict of {name: {cv_r2_mean, cv_r2_std}}.
    Data is sorted by season_year before CV so folds respect time order.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2", n_jobs=-1)
        cv_results[name] = {
            "cv_r2_mean": round(float(scores.mean()), 4),
            "cv_r2_std": round(float(scores.std()), 4),
        }
        print(f"  {name:15s}  CV R²={scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results


# ── Model selection ───────────────────────────────────────────────────────────

def _select_winner(candidates: dict, X_train, y_train, X_test, y_test) -> tuple:
    """
    Fit all candidates on full training set, evaluate on holdout.
    Winner = lowest test RMSE.
    Tie-break (within 0.3 wins RMSE): prefer Ridge > RF > XGBoost for interpretability.
    Returns (winner_name, winner_model, results_dict).
    """
    TIEBREAK_THRESHOLD = 0.3
    TIEBREAK_ORDER = ["ridge", "random_forest", "xgboost"]

    fitted = {}
    results = {}
    for name, model in candidates.items():
        print(f"  Fitting {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"    Test R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")

    # Sort by RMSE ascending; apply tie-break within threshold
    sorted_by_rmse = sorted(results.items(), key=lambda x: x[1]["rmse"])
    best_name, best_metrics = sorted_by_rmse[0]

    for name, metrics in sorted_by_rmse[1:]:
        if metrics["rmse"] - best_metrics["rmse"] <= TIEBREAK_THRESHOLD:
            # Within tie-break window — prefer simpler model
            if TIEBREAK_ORDER.index(name) < TIEBREAK_ORDER.index(best_name):
                print(f"  Tie-break: {name} preferred over {best_name} (RMSE delta={metrics['rmse'] - best_metrics['rmse']:.3f} ≤ {TIEBREAK_THRESHOLD})")
                best_name, best_metrics = name, metrics

    reason = (
        f"{best_name} had lowest test RMSE ({best_metrics['rmse']:.3f} wins). "
        + ", ".join(
            f"{n}: RMSE={r['rmse']:.3f}"
            for n, r in results.items()
            if n != best_name
        )
    )
    return best_name, fitted[best_name], results, reason


# ── Playoff classifier ────────────────────────────────────────────────────────

def _train_playoff_classifier(X_train, y_playoff_train) -> XGBClassifier:
    """Train XGBoost classifier for playoff probability (0/1 target)."""
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_playoff_train)
    return clf


# ── MLflow logging (graceful fallback if server unavailable) ──────────────────

def _log_to_mlflow(winner_name, winner_model, all_results, cv_results, n_train, n_test):
    """Log training run to MLflow. Prints warning and continues if server is down."""
    try:
        import mlflow
        import mlflow.sklearn
        import mlflow.xgboost

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("nba-win-predictor")

        with mlflow.start_run(run_name=f"train_{winner_name}_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.set_tag("model_winner", winner_name)
            mlflow.log_param("n_training_samples", n_train)
            mlflow.log_param("n_test_samples", n_test)

            for name, metrics in all_results.items():
                mlflow.log_metric(f"{name}_test_r2", metrics["r2"])
                mlflow.log_metric(f"{name}_test_rmse", metrics["rmse"])

            for name, cv in cv_results.items():
                mlflow.log_metric(f"{name}_cv_r2_mean", cv["cv_r2_mean"])
                mlflow.log_metric(f"{name}_cv_r2_std", cv["cv_r2_std"])

            winner_metrics = all_results[winner_name]
            mlflow.log_metric("winner_r2", winner_metrics["r2"])
            mlflow.log_metric("winner_rmse", winner_metrics["rmse"])

            if winner_name == "xgboost":
                mlflow.xgboost.log_model(winner_model, "model")
            else:
                mlflow.sklearn.log_model(winner_model, "model")

            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run logged: {run_id}")
            return run_id

    except Exception as e:
        print(f"MLflow logging skipped ({e}) — continuing without tracking")
        return None


# ── Main training entry point ─────────────────────────────────────────────────

def run_training() -> dict:
    """
    Full training pipeline:
      1. Load data → prepare features → chronological split
      2. TimeSeriesSplit CV on training set for all 3 candidates
      3. Select winner by test RMSE
      4. Hard quality gate: R² > 0.75 AND RMSE < 6.0 — raises if not met
      5. Train playoff classifier
      6. Generate and save SHAP artifacts
      7. Log to MLflow (graceful fallback)
      8. Save model + selection report

    Returns the model selection report dict.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading and preparing data...")
    df = load_data()
    X, y_wins, y_playoff = prepare_features(df)

    # Sort by season_year so TimeSeriesSplit folds respect chronological order
    season_years = df.loc[X.index, "season_year"]
    sort_order = season_years.argsort()
    X = X.iloc[sort_order]
    y_wins = y_wins.iloc[sort_order]
    y_playoff = y_playoff.iloc[sort_order]

    X_train, X_test, y_train, y_test, y_playoff_train, y_playoff_test = chronological_split(
        X, y_wins, y_playoff, df
    )
    print(f"Train: {len(X_train)} rows  |  Holdout: {len(X_test)} rows")

    # ── Step 1: Cross-validation ──────────────────────────────────────────────
    print("\nRunning TimeSeriesSplit(n_splits=5) cross-validation...")
    candidates = _build_candidates()
    cv_results = _cross_validate(candidates, X_train, y_train)

    # ── Step 2: Select winner ─────────────────────────────────────────────────
    print("\nFitting candidates on full training set and evaluating on holdout...")
    candidates = _build_candidates()  # fresh instances for final fit
    winner_name, winner_model, all_results, reason = _select_winner(
        candidates, X_train, y_train, X_test, y_test
    )
    print(f"\nWinner: {winner_name.upper()}")
    print(f"Reason: {reason}")

    # ── Step 3: Quality gate ──────────────────────────────────────────────────
    winner_metrics = all_results[winner_name]
    gate_passed = check_quality_gate(winner_metrics)
    if not gate_passed:
        raise RuntimeError(
            f"Quality gate FAILED for best model ({winner_name}): "
            f"R²={winner_metrics['r2']:.4f}, RMSE={winner_metrics['rmse']:.4f}. "
            "Revisit feature engineering before proceeding to Phase 3."
        )

    # ── Step 4: Playoff classifier ────────────────────────────────────────────
    print("\nTraining playoff probability classifier...")
    playoff_model = _train_playoff_classifier(X_train, y_playoff_train)
    from sklearn.metrics import accuracy_score, roc_auc_score
    playoff_probs = playoff_model.predict_proba(X_test)[:, 1]
    playoff_auc = float(roc_auc_score(y_playoff_test, playoff_probs))
    print(f"  Playoff classifier AUC: {playoff_auc:.4f}")

    # ── Step 5: SHAP artifacts ────────────────────────────────────────────────
    print("\nGenerating SHAP artifacts...")
    shap_summary = save_shap_artifacts(winner_model, X_train, X_test)
    print("Top 5 features by SHAP importance:")
    print(shap_summary.head(5).to_string(index=False))

    # ── Step 6: MLflow ────────────────────────────────────────────────────────
    run_id = _log_to_mlflow(
        winner_name, winner_model, all_results, cv_results,
        n_train=len(X_train), n_test=len(X_test),
    )

    # ── Step 7: Save models ───────────────────────────────────────────────────
    joblib.dump(winner_model, MODEL_PATH)
    joblib.dump(playoff_model, PLAYOFF_MODEL_PATH)
    print(f"\nModels saved: {MODEL_PATH}, {PLAYOFF_MODEL_PATH}")

    # ── Step 8: Selection report ──────────────────────────────────────────────
    report = {
        "winner": winner_name,
        "trained_at": datetime.now().isoformat(),
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "feature_cols": list(X.columns),
        "results": {
            name: {**metrics, **cv_results.get(name, {})}
            for name, metrics in all_results.items()
        },
        "selection_reason": reason,
        "playoff_classifier_auc": round(playoff_auc, 4),
        "gate_passed": gate_passed,
        "mlflow_run_id": run_id,
        "model_path": MODEL_PATH,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Model selection report saved to {REPORT_PATH}")

    write_eval_report(report)

    print("\n" + "=" * 60)
    print(f"Training complete. Winner: {winner_name.upper()}")
    print(f"  R²={winner_metrics['r2']:.4f}  RMSE={winner_metrics['rmse']:.4f}")

    return report


def load_models() -> tuple:
    """Load the saved wins regressor and playoff classifier from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run run_training() first.")
    wins_model = joblib.load(MODEL_PATH)
    playoff_model = joblib.load(PLAYOFF_MODEL_PATH) if os.path.exists(PLAYOFF_MODEL_PATH) else None
    return wins_model, playoff_model


if __name__ == "__main__":
    run_training()
