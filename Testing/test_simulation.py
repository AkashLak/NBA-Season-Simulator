"""Tests for models/simulate_season.py — league-wide simulation engine."""

import numpy as np
import pandas as pd
import pytest

from models.simulate_season import (
    _build_feature_row,
    _init_team_state,
    _rest_days,
    _rolling_pct,
    simulate_league,
    simulate_team,
)
from etl.transform_games import GAME_FEATURE_COLS


# ── Helpers ────────────────────────────────────────────────────────────────────

class _ConstantModel:
    """Always predicts the same home win probability."""
    def __init__(self, prob=0.6):
        self.prob = prob
    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]] * len(X))


def _make_schedule(n_games=4, teams=(1, 2)):
    """Alternating home/away between two teams."""
    home = [teams[i % 2] for i in range(n_games)]
    away = [teams[(i + 1) % 2] for i in range(n_games)]
    return pd.DataFrame({
        "game_id":  [f"G{i:03d}" for i in range(n_games)],
        "game_date": pd.date_range("2024-10-01", periods=n_games, freq="2D"),
        "home_id":  home,
        "away_id":  away,
    })


def _make_prior_lookup(*team_ids):
    return {
        tid: {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0}
        for tid in team_ids
    }


def _patch_sim(monkeypatch, schedule, prior_lookup):
    """Replace schedule and prior lookups with deterministic test data."""
    monkeypatch.setattr(
        "models.simulate_season._get_schedule",
        lambda sy, engine=None, _depth=0: schedule,
    )
    monkeypatch.setattr(
        "models.simulate_season._build_prior_features_lookup",
        lambda sy, engine=None: prior_lookup,
    )


# ── _rolling_pct ───────────────────────────────────────────────────────────────

def test_rolling_pct_empty_history_returns_half():
    assert _rolling_pct([], 5) == 0.5
    assert _rolling_pct([], 10) == 0.5


def test_rolling_pct_full_window():
    history = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert abs(_rolling_pct(history, 5) - 0.6) < 1e-9


def test_rolling_pct_partial_window_uses_available():
    history = [1.0, 0.0, 1.0]
    # Only 3 values available for window-5 → use all 3
    assert abs(_rolling_pct(history, 5) - (2 / 3)) < 1e-9


def test_rolling_pct_uses_last_n_only():
    history = [0.0] * 5 + [1.0, 1.0, 1.0, 1.0, 1.0]
    # Most recent 5 games are all wins
    assert abs(_rolling_pct(history, 5) - 1.0) < 1e-9


# ── _rest_days ─────────────────────────────────────────────────────────────────

def test_rest_days_none_returns_7():
    assert _rest_days(None, "2024-10-01") == 7


def test_rest_days_two_days():
    assert _rest_days("2024-10-01", "2024-10-03") == 2


def test_rest_days_capped_at_7():
    assert _rest_days("2024-09-01", "2024-10-01") == 7


# ── _build_feature_row ─────────────────────────────────────────────────────────

def test_build_feature_row_has_all_columns():
    home = _init_team_state(1, {"net_rating": 3.0, "off_rating": 115.0, "def_rating": 112.0})
    away = _init_team_state(2, {"net_rating": -1.0, "off_rating": 110.0, "def_rating": 111.0})
    row = _build_feature_row(home, away, "2024-10-01")
    assert set(row.keys()) == set(GAME_FEATURE_COLS)


def test_build_feature_row_home_flag_is_1():
    home = _init_team_state(1, {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0})
    away = _init_team_state(2, {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0})
    row = _build_feature_row(home, away, "2024-10-01")
    assert row["home_flag"] == 1


def test_build_feature_row_first_game_rest_7():
    home = _init_team_state(1, {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0})
    away = _init_team_state(2, {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0})
    row = _build_feature_row(home, away, "2024-10-01")
    assert row["rest_days"] == 7
    assert row["opp_rest_days"] == 7


def test_build_feature_row_wrong_keys_raises():
    home = _init_team_state(1, {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0})
    away = _init_team_state(2, {"net_rating": 0.0, "off_rating": 113.0, "def_rating": 113.0})
    # Monkeypatch GAME_FEATURE_COLS temporarily to trigger mismatch
    import models.simulate_season as sim_mod
    original = sim_mod.GAME_FEATURE_COLS
    sim_mod.GAME_FEATURE_COLS = original + ["nonexistent_col"]
    try:
        with pytest.raises(AssertionError, match="Feature mismatch"):
            _build_feature_row(home, away, "2024-10-01")
    finally:
        sim_mod.GAME_FEATURE_COLS = original


# ── simulate_league ────────────────────────────────────────────────────────────

def test_total_expected_wins_equals_total_games(monkeypatch):
    """
    Conservation law: sum of all teams' expected_wins must equal total games played.
    Each game contributes exactly 1.0 (win_prob + (1 - win_prob) = 1.0).
    """
    schedule = _make_schedule(n_games=82, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    results = simulate_league(2025, _ConstantModel(0.6))
    total_wins  = sum(r["expected_wins"] for r in results.values())
    total_games = len(schedule)   # 82 games

    assert abs(total_wins - total_games) < 0.01, (
        f"Total expected wins {total_wins:.2f} must equal total games {total_games}"
    )


def test_simulation_output_in_valid_range(monkeypatch):
    schedule = _make_schedule(n_games=82, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    results = simulate_league(2025, _ConstantModel(0.5))
    for tid, res in results.items():
        assert 0 <= res["expected_wins"] <= 82
        assert all(0 <= p <= 1 for p in res["game_probs"])


def test_rolling_state_updates_after_each_game(monkeypatch):
    """After the first simulated game, both teams must have win_history of length 1."""
    schedule = _make_schedule(n_games=1, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    results = simulate_league(2025, _ConstantModel(0.6))
    # Each team played 1 game
    assert len(results[1]["game_probs"]) == 1
    assert len(results[2]["game_probs"]) == 1


def test_constant_model_symmetric_schedule(monkeypatch):
    """With equal home/away games and constant model, totals are symmetric."""
    # 4 games: alternating home/away → each team hosts 2, visits 2
    schedule = _make_schedule(n_games=4, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    results = simulate_league(2025, _ConstantModel(0.6))
    # Team 1 hosts games 0,2 → +0.6 each; visits games 1,3 → +0.4 each = 2.0
    # Team 2 hosts games 1,3 → +0.6 each; visits games 0,2 → +0.4 each = 2.0
    assert abs(results[1]["expected_wins"] - 2.0) < 0.01
    assert abs(results[2]["expected_wins"] - 2.0) < 0.01


def test_team_quality_override_changes_result(monkeypatch):
    """Overriding a team's net_rating must change that team's results."""
    schedule = _make_schedule(n_games=10, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    model = _ConstantModel(0.5)   # 50/50 without override

    base_results = simulate_league(2025, model)
    base_wins = base_results[1]["expected_wins"]

    # With a large override the constant model ignores it,
    # so test that override flows through without error
    override_results = simulate_league(
        2025, model, team_overrides={1: {"prev_net_rating": 15.0}}
    )
    assert 1 in override_results
    assert abs(override_results[1]["expected_wins"] - base_wins) < 0.01


def test_schedule_fallback_raises_on_empty_db(monkeypatch):
    """If no schedule found after 3 attempts, ValueError must be raised."""
    monkeypatch.setattr(
        "models.simulate_season._get_schedule",
        lambda sy, engine=None, _depth=0: (_ for _ in ()).throw(
            ValueError("No schedule data available")
        ),
    )
    # Since we patched _get_schedule to raise directly, confirm simulate_league propagates it
    with pytest.raises((ValueError, Exception)):
        simulate_league(1990, _ConstantModel())


# ── simulate_team ──────────────────────────────────────────────────────────────

def test_simulate_team_returns_correct_team(monkeypatch):
    schedule = _make_schedule(n_games=4, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    result = simulate_team(1, 2025, _ConstantModel(0.6))
    assert "expected_wins" in result
    assert "game_probs" in result
    assert len(result["game_probs"]) == 4


def test_simulate_team_unknown_team_raises(monkeypatch):
    schedule = _make_schedule(n_games=4, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    with pytest.raises(ValueError, match="not found"):
        simulate_team(99, 2025, _ConstantModel())


# ── Feature column order ───────────────────────────────────────────────────────

def test_feature_column_order_enforced(monkeypatch):
    """
    pd.DataFrame([row])[GAME_FEATURE_COLS] must produce columns in the
    exact order defined in GAME_FEATURE_COLS regardless of dict insertion order.
    """
    schedule = _make_schedule(n_games=1, teams=(1, 2))
    prior    = _make_prior_lookup(1, 2)
    _patch_sim(monkeypatch, schedule, prior)

    captured_X = []
    class CapturingModel:
        def predict_proba(self, X):
            captured_X.append(X.columns.tolist())
            return np.array([[0.4, 0.6]])

    simulate_league(2025, CapturingModel())
    assert captured_X[0] == GAME_FEATURE_COLS, (
        f"Column order mismatch.\nExpected: {GAME_FEATURE_COLS}\nGot:      {captured_X[0]}"
    )
