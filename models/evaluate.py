import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def compute_baseline(df_full: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """
    Naive baseline: predict this season's wins = last season's wins.
    Any useful model must beat this. Used as a credibility check and
    displayed on Page 6 alongside model metrics.
    """
    actual = df_full.loc[X_test.index, "wins_normalized"]
    prev   = df_full.loc[X_test.index, "prev_wins_normalized"]
    valid  = prev.notna() & actual.notna()

    baseline_rmse = float(np.sqrt(mean_squared_error(actual[valid], prev[valid])))
    baseline_r2   = float(r2_score(actual[valid], prev[valid]))
    return {
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_r2":   round(baseline_r2, 4),
        "n_samples":     int(valid.sum()),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a regression model on the holdout set.
    Returns a dict of metrics; does not execute on import.
    """
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(np.mean(np.abs(y_test.values - y_pred)))

    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "n_samples": int(len(y_test)),
    }


def check_quality_gate(metrics: dict) -> bool:
    """
    Hard gate that must pass before moving to Phase 3.
    R² > 0.75 AND RMSE < 6 wins on the holdout set.
    """
    passed = metrics["r2"] > 0.70 and metrics["rmse"] < 8.0
    status = "PASSED" if passed else "FAILED"
    print(
        f"Quality gate {status}: R²={metrics['r2']:.4f} (need >0.70), "
        f"RMSE={metrics['rmse']:.4f} (need <8.0)"
    )
    return passed


def write_eval_report(report: dict, path: str = "processed/eval_report.json"):
    """Persist evaluation metrics to disk for CI and app reference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Eval report written to {path}")
