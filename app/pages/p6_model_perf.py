import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.shared import load_season_report, load_game_report, load_learning_curve

st.header("Model Performance")

season_report = load_season_report()
game_report   = load_game_report()
lc            = load_learning_curve()

# ── Season model card ──────────────────────────────────────────────────────────
st.subheader("Season Model  (wins regression)")

if season_report:
    winner = season_report.get("winner", "unknown")
    res    = season_report.get("results", {}).get(winner, {})
    base   = season_report.get("baseline", {})
    improv = season_report.get("improvement_over_baseline", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Winner",    winner.upper())
    col2.metric("Test R²",   f"{res.get('r2', 0):.4f}")
    col3.metric("Test RMSE", f"{res.get('rmse', 0):.2f} wins")
    col4.metric("vs Baseline", f"{improv:+.2f} wins", delta=improv, delta_color="normal")

    if season_report.get("baseline_warning"):
        st.warning(season_report["baseline_warning"])

    # CV results table
    with st.expander("Cross-validation results (all candidates)"):
        rows = []
        for name, m in season_report.get("results", {}).items():
            rows.append({
                "Model":          name,
                "CV R²":          m.get("cv_r2_mean"),
                "CV train R²":    m.get("cv_train_r2_mean"),
                "Overfitting gap":m.get("overfitting_gap"),
                "Test R²":        m.get("r2"),
                "Test RMSE":      m.get("rmse"),
            })
        if rows:
            cv_df = pd.DataFrame(rows)
            st.dataframe(cv_df, hide_index=True, use_container_width=True)
            for _, r in cv_df.iterrows():
                if r.get("Overfitting gap") and r["Overfitting gap"] > 0.15:
                    st.warning(f"{r['Model']}: overfitting gap {r['Overfitting gap']:.3f} > 0.15")

    # Baseline breakdown
    if base:
        st.caption(
            f"Baseline (predict last season's wins): "
            f"RMSE={base.get('baseline_rmse', '?'):.2f}  "
            f"R²={base.get('baseline_r2', '?'):.4f}"
        )
else:
    st.info("Season model not trained yet. Run `python -m models.train_model`.")

st.divider()

# ── Game model card ────────────────────────────────────────────────────────────
st.subheader("Game Model  (win probability classifier)")

if game_report:
    g_winner  = game_report.get("winner", "unknown")
    g_metrics = game_report.get("winner_metrics", {})
    g_base    = game_report.get("baseline_metrics", {})
    g_improv  = game_report.get("improvement_over_baseline", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Winner",    g_winner.upper())
    col2.metric("AUC",       f"{g_metrics.get('holdout_auc', 0):.4f}")
    col3.metric("Log Loss",  f"{g_metrics.get('holdout_logloss', 0):.4f}")
    col4.metric("Accuracy",  f"{g_metrics.get('holdout_accuracy', 0):.3f}")

    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Baseline Log Loss",
                  f"{g_base.get('baseline_logloss', 0):.4f}",
                  help="Home-court-always baseline")
    col_b2.metric("Baseline AUC",  f"{g_base.get('baseline_auc', 0):.4f}")
    col_b3.metric("Log Loss Improvement", f"{g_improv:+.4f}",
                  delta=g_improv, delta_color="normal")

    if game_report.get("baseline_warning"):
        st.warning(game_report["baseline_warning"])

    with st.expander("Cross-validation results (all candidates)"):
        rows = []
        for name, m in game_report.get("cv_results", {}).items():
            rows.append({
                "Model":           name,
                "CV LogLoss":      m.get("cv_logloss"),
                "CV AUC":          m.get("cv_auc"),
                "CV train LogLoss":m.get("cv_train_logloss"),
                "Overfitting gap": m.get("overfitting_gap"),
            })
        if rows:
            cv_df = pd.DataFrame(rows)
            st.dataframe(cv_df, hide_index=True, use_container_width=True)
else:
    st.info("Game model not trained yet. Run `python -m models.train_game_model`.")

st.divider()

# ── Learning curve ─────────────────────────────────────────────────────────────
st.subheader("Season Model Learning Curve")
st.caption(
    "Converging train and validation R² lines indicate the model is not overfitting. "
    "A persistent gap suggests more data or regularisation would help."
)

if lc and "train_sizes" in lc:
    lc_df = pd.DataFrame({
        "Training samples": lc["train_sizes"],
        "Train R²":         lc["train_r2"],
        "Validation R²":    lc["val_r2"],
    })
    lc_melted = lc_df.melt("Training samples", var_name="Set", value_name="R²")
    fig = px.line(
        lc_melted, x="Training samples", y="R²", color="Set",
        markers=True,
        color_discrete_map={"Train R²": "#3498db", "Validation R²": "#e74c3c"},
    )
    fig.update_layout(height=380, yaxis=dict(range=[-0.1, 1.05]))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Learning curve not available. Train the season model to generate it.")

# ── MLflow runs ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("MLflow Experiment Runs")

@st.cache_data(ttl=120)
def _load_mlflow_runs(experiment_name: str) -> pd.DataFrame:
    try:
        import mlflow
        import os
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
        records = []
        for r in runs[:20]:
            records.append({
                "Date":    pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                "Model":   r.data.tags.get("model_winner", "?"),
                "R²/AUC":  r.data.metrics.get("winner_r2") or r.data.metrics.get("winner_auc"),
                "RMSE/LogLoss": r.data.metrics.get("winner_rmse") or r.data.metrics.get("winner_logloss"),
                "N train": r.data.params.get("n_training_samples"),
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.caption("**nba-win-predictor** (season model)")
    runs_s = _load_mlflow_runs("nba-win-predictor")
    if not runs_s.empty:
        st.dataframe(runs_s, hide_index=True, use_container_width=True)
    else:
        st.info("MLflow server not reachable or no runs logged yet.")

with col_e2:
    st.caption("**nba-game-predictor** (game model)")
    runs_g = _load_mlflow_runs("nba-game-predictor")
    if not runs_g.empty:
        st.dataframe(runs_g, hide_index=True, use_container_width=True)
    else:
        st.info("MLflow server not reachable or no runs logged yet.")
