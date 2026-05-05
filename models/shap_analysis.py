import numpy as np
import pandas as pd
import shap


def _unwrap_pipeline(model):
    """
    If model is a sklearn Pipeline, return the final estimator step.
    LinearExplainer and TreeExplainer need the raw model, not the Pipeline wrapper.
    For Pipelines, also return a transformed background dataset.
    """
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model.named_steps[list(model.named_steps.keys())[-1]], model
    return model, None


def _get_explainer(model, X_background: pd.DataFrame):
    """Return the appropriate SHAP explainer for the model type."""
    raw_model, pipeline = _unwrap_pipeline(model)
    model_type = type(raw_model).__name__

    if model_type in ("XGBRegressor", "XGBClassifier",
                      "RandomForestRegressor", "RandomForestClassifier"):
        return shap.TreeExplainer(raw_model)

    # Linear models — need to transform background through any preceding steps
    if pipeline is not None:
        # Apply all steps except the final estimator to get scaled background
        steps_except_last = list(pipeline.named_steps.keys())[:-1]
        X_transformed = X_background.copy()
        for step_name in steps_except_last:
            X_transformed = pipeline.named_steps[step_name].transform(X_transformed)
        return shap.LinearExplainer(raw_model, X_transformed)

    return shap.LinearExplainer(raw_model, X_background)


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


def _transform_for_linear(model, X: pd.DataFrame):
    """Apply pipeline pre-processing steps (e.g. StandardScaler) before SHAP."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        steps_except_last = list(model.named_steps.keys())[:-1]
        X_t = X.copy()
        for step_name in steps_except_last:
            X_t = model.named_steps[step_name].transform(X_t)
        return X_t
    return X


def generate_shap_summary(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature across the dataset.
    Returns a DataFrame sorted by importance descending.
    """
    explainer  = _get_explainer(model, X)
    X_input    = _transform_for_linear(model, X)
    shap_values = explainer.shap_values(X_input)

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

    explainer  = _get_explainer(model, X_train)
    X_test_in  = _transform_for_linear(model, X_test)
    test_shap  = explainer.shap_values(X_test_in)
    if isinstance(test_shap, list):
        test_shap = test_shap[0]
    np.save(values_path, np.array(test_shap))
    print(f"SHAP values saved to {values_path}")

    return summary
