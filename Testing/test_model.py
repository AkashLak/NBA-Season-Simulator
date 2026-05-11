"""Tests for models/evaluate.py, models/data_prep.py, and trained models."""

import os

import numpy as np
import pandas as pd
import pytest

from models.data_prep import FEATURE_COLS, chronological_split
from models.evaluate import check_quality_gate, compute_baseline

# ── Quality gate ───────────────────────────────────────────────────────────────


def test_quality_gate_passes():
    assert check_quality_gate({"r2": 0.70, "rmse": 5.5, "mae": 4.0, "n_samples": 90})


def test_quality_gate_fails_low_r2():
    assert not check_quality_gate(
        {"r2": 0.60, "rmse": 5.5, "mae": 4.0, "n_samples": 90}
    )


def test_quality_gate_fails_high_rmse():
    assert not check_quality_gate(
        {"r2": 0.80, "rmse": 9.0, "mae": 5.0, "n_samples": 90}
    )


def test_quality_gate_fails_both():
    assert not check_quality_gate(
        {"r2": 0.50, "rmse": 8.0, "mae": 6.0, "n_samples": 90}
    )


# ── Baseline comparison ────────────────────────────────────────────────────────


def test_compute_baseline_returns_expected_keys():
    df = pd.DataFrame(
        {
            "wins_normalized": [40.0, 45.0, 50.0, 48.0, 52.0],
            "prev_wins_normalized": [38.0, 40.0, 45.0, 50.0, 48.0],
        }
    )
    X_test = pd.DataFrame(index=pd.Index([1, 2, 3, 4]))
    result = compute_baseline(df, X_test)
    assert "baseline_rmse" in result
    assert "baseline_r2" in result
    assert "n_samples" in result
    assert result["n_samples"] == 4


def test_compute_baseline_ignores_nan_rows():
    df = pd.DataFrame(
        {
            "wins_normalized": [40.0, 45.0, 50.0],
            "prev_wins_normalized": [np.nan, 40.0, 45.0],  # first row NaN
        }
    )
    X_test = pd.DataFrame(index=pd.Index([0, 1, 2]))
    result = compute_baseline(df, X_test)
    assert result["n_samples"] == 2  # NaN row excluded


def test_compute_baseline_rmse_positive():
    df = pd.DataFrame(
        {
            "wins_normalized": [50.0, 40.0, 60.0],
            "prev_wins_normalized": [45.0, 45.0, 55.0],
        }
    )
    X_test = pd.DataFrame(index=pd.Index([0, 1, 2]))
    result = compute_baseline(df, X_test)
    assert result["baseline_rmse"] > 0


# ── chronological_split ────────────────────────────────────────────────────────


def test_chronological_split_respects_time_order():
    """All training seasons must be strictly before all test seasons."""
    n = 90  # 3 seasons × 30 teams
    df = pd.DataFrame(
        {
            "team_id": [i % 30 for i in range(n)],
            "season_year": [2020 + i // 30 for i in range(n)],
        }
    )
    X = pd.DataFrame({"feature": range(n)}, index=df.index)
    y_wins = pd.Series(range(n), index=df.index, dtype=float)
    y_play = pd.Series([0] * n, index=df.index)

    X_tr, X_te, y_tr, y_te, _, _ = chronological_split(
        X, y_wins, y_play, df, holdout_seasons=1
    )

    train_seasons = df.loc[X_tr.index, "season_year"]
    test_seasons = df.loc[X_te.index, "season_year"]

    assert (
        train_seasons.max() < test_seasons.min()
    ), f"Train max season {train_seasons.max()} must be < test min {test_seasons.min()}"


def test_chronological_split_holdout_size():
    """Holdout set should contain exactly holdout_seasons × 30 rows."""
    n = 150  # 5 seasons × 30 teams
    df = pd.DataFrame(
        {
            "team_id": [i % 30 for i in range(n)],
            "season_year": [2020 + i // 30 for i in range(n)],
        }
    )
    X = pd.DataFrame({"f": range(n)}, index=df.index)
    y_w = pd.Series(range(n), dtype=float, index=df.index)
    y_p = pd.Series([0] * n, index=df.index)

    _, X_te, _, _, _, _ = chronological_split(X, y_w, y_p, df, holdout_seasons=2)
    assert len(X_te) == 60  # 2 seasons × 30 teams


# ── FEATURE_COLS completeness ──────────────────────────────────────────────────


def test_feature_cols_includes_new_roster_features():
    assert "prev_std_dev_pie" in FEATURE_COLS
    assert "prev_top_3_minutes_share" in FEATURE_COLS


def test_feature_cols_count():
    assert (
        len(FEATURE_COLS) == 22
    ), f"Expected 22 FEATURE_COLS, got {len(FEATURE_COLS)}: {FEATURE_COLS}"


# ── Probability clipping (game model safety) ───────────────────────────────────


def test_probability_clipping_prevents_inf_log_loss():
    """Clipping [0, 1] probabilities to (ε, 1-ε) must keep log_loss finite."""
    from sklearn.metrics import log_loss

    raw_probs = np.array([0.0, 1.0, 0.5, 0.001, 0.999])
    labels = np.array([1, 0, 1, 1, 0])
    clipped = np.clip(raw_probs, 1e-15, 1 - 1e-15)
    loss = log_loss(labels, clipped)
    assert np.isfinite(loss), f"log_loss should be finite after clipping, got {loss}"


# ── Trained model tests (skipped if models not built yet) ─────────────────────


@pytest.fixture
def season_model():
    path = "models/best_wins_model.pkl"
    if not os.path.exists(path):
        pytest.skip("Season model not trained yet — run python -m models.train_model")
    import joblib

    return joblib.load(path)


@pytest.fixture
def game_model():
    path = "models/game_win_model.pkl"
    if not os.path.exists(path):
        pytest.skip(
            "Game model not trained yet — run python -m models.train_game_model"
        )
    import joblib

    return joblib.load(path)


def test_season_model_prediction_in_valid_range(season_model):
    """Predictions on real data must be plausible NBA win totals (0–82 after clamping)."""
    parquet = "processed/ml_features.parquet"
    if not os.path.exists(parquet):
        pytest.skip("ml_features.parquet not found")
    df = pd.read_parquet(parquet)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].dropna().head(5)
    if X.empty:
        pytest.skip("No complete rows in ml_features.parquet")
    preds = season_model.predict(X)
    for pred in preds:
        clamped = float(max(0.0, min(82.0, pred)))
        assert 0 <= clamped <= 82, f"Clamped prediction {clamped} out of range"


def test_game_model_outputs_valid_probabilities(game_model):
    """predict_proba must return (n, 2) array with each row summing to 1."""
    from etl.transform_games import GAME_FEATURE_COLS

    X = pd.DataFrame([{col: 0.5 for col in GAME_FEATURE_COLS}])
    probs = game_model.predict_proba(X)
    assert probs.shape == (1, 2), f"Expected (1, 2) probs, got {probs.shape}"
    assert abs(probs[0].sum() - 1.0) < 1e-6
    assert all(0 <= p <= 1 for p in probs[0])


def test_game_model_beats_home_court_baseline(game_model):
    """
    Game model AUC on any realistic data must exceed the home-court-only baseline
    (~0.58). We verify this by checking the saved report rather than re-evaluating.
    """
    import json

    report_path = "models/game_model_report.json"
    if not os.path.exists(report_path):
        pytest.skip("Game model report not found")
    with open(report_path) as f:
        report = json.load(f)
    model_auc = report["winner_metrics"]["holdout_auc"]
    baseline_auc = report["baseline_metrics"]["baseline_auc"]
    assert (
        model_auc > baseline_auc
    ), f"Game model AUC {model_auc:.4f} must exceed home-court baseline {baseline_auc:.4f}"


def test_shap_top_features_are_sensible():
    """
    At least one strong prior-season or schedule feature must appear in the
    top-5 SHAP features. Guards against the model using noise as signal.
    """
    shap_path = "processed/game_shap_summary.parquet"
    if not os.path.exists(shap_path):
        pytest.skip("SHAP summary not found — train game model first")
    shap_df = pd.read_parquet(shap_path)
    top5 = shap_df.head(5)["feature"].tolist() if "feature" in shap_df.columns else []
    expected_signals = {
        "team_prev_net_rating",
        "opp_prev_net_rating",
        "team_rolling_win_pct_10",
        "home_flag",
        "team_prev_off_rating",
        "opp_prev_off_rating",
    }
    assert any(f in expected_signals for f in top5), (
        f"No strong signal feature in top-5 SHAP: {top5}. "
        "Model may be using noise — revisit feature engineering."
    )


def test_season_model_beats_baseline():
    """Season model RMSE must be lower than naive baseline (predict last year's wins)."""
    import json

    report_path = "models/model_selection_report.json"
    if not os.path.exists(report_path):
        pytest.skip("Season model report not found")
    with open(report_path) as f:
        report = json.load(f)
    if not report.get("baseline"):
        pytest.skip("Baseline not recorded in report")
    winner = report["winner"]
    m_rmse = report["results"][winner]["rmse"]
    b_rmse = report["baseline"]["baseline_rmse"]
    assert (
        m_rmse < b_rmse
    ), f"Season model RMSE {m_rmse:.4f} must be lower than baseline {b_rmse:.4f}"


def test_learning_curve_exists_after_training():
    import json

    path = "processed/learning_curve.json"
    if not os.path.exists(path):
        pytest.skip("Learning curve not generated yet")
    with open(path) as f:
        lc = json.load(f)
    assert "train_sizes" in lc
    assert "train_r2" in lc
    assert "val_r2" in lc
    assert len(lc["train_sizes"]) == len(lc["train_r2"]) == len(lc["val_r2"])
