import numpy as np
import pandas as pd
import shap


def _get_explainer(model, X_background: pd.DataFrame):
    """Return the appropriate SHAP explainer for the model type."""
    model_type = type(model).__name__
    if model_type in ("XGBRegressor", "XGBClassifier", "RandomForestRegressor", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    # Ridge and other linear models
    return shap.LinearExplainer(model, X_background)


def get_prediction_explanation(model, X_row: pd.DataFrame, X_background: pd.DataFrame) -> dict:
    """
    Explain a single prediction row using SHAP.

    Returns:
        base_value      — model's average prediction across training data
        contributions   — dict of {feature_name: shap_value} for this row
        predicted_wins  — base_value + sum(contributions), matches model.predict()
    """
    explainer = _get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_row)

    # shap_values shape: (1, n_features) for single-row input
    if isinstance(shap_values, list):
        # Some explainers return list for multi-output — take first output
        shap_values = shap_values[0]
    values = np.array(shap_values).flatten()

    base_value = float(
        explainer.expected_value[0]
        if hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value
    )

    contributions = {
        col: round(float(val), 4)
        for col, val in zip(X_row.columns, values)
    }

    return {
        "base_value": round(base_value, 4),
        "contributions": contributions,
        "predicted_wins": round(base_value + values.sum(), 4),
    }


def generate_shap_summary(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature across the dataset.
    Returns a DataFrame sorted by importance descending.
    Saved to processed/shap_summary.parquet for app use.
    """
    explainer = _get_explainer(model, X)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(np.array(shap_values)).mean(axis=0)
    summary = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def save_shap_artifacts(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        summary_path: str = "processed/shap_summary.parquet",
                        values_path: str = "processed/shap_values.npy"):
    """Compute and persist SHAP summary + raw values for the test set."""
    import os
    os.makedirs("processed", exist_ok=True)

    summary = generate_shap_summary(model, X_train)
    summary.to_parquet(summary_path, index=False)
    print(f"SHAP summary saved to {summary_path}")

    explainer = _get_explainer(model, X_train)
    test_shap = explainer.shap_values(X_test)
    if isinstance(test_shap, list):
        test_shap = test_shap[0]
    np.save(values_path, np.array(test_shap))
    print(f"SHAP values saved to {values_path}")

    return summary
