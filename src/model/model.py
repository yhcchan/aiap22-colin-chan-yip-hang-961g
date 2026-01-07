import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from src.features.feature_engineering import feature_pipeline
from src.utils import load_yaml_config

MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hgboost": HistGradientBoostingClassifier
}

def _build_estimator(
    model_name: str,
    model_cfg: dict,
    ) -> BaseEstimator:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered. "
            "Add it to MODEL_REGISTRY in src/model/model.py."
        )

    estimator_cls = MODEL_REGISTRY[model_name]
    estimator_params = dict(model_cfg.get("estimator_params", {}))
    return estimator_cls(**estimator_params)

# Evaluation Metrics
def _evaluate_model(y_true, y_pred, metrics, average="macro"):
    results = {}
    for metric in metrics:
        if metric == "accuracy":
            results["accuracy"] = accuracy_score(y_true, y_pred)
        elif metric == "macro_f1":
            results["macro_f1"] = f1_score(y_true, y_pred, average=average)
        elif metric == "cohen_kappa":
            results["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
        elif metric == "qwk":
            # Simplified QWK
            results["qwk"] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return results

def _save_model_results(model_name, best_params, metrics, feature_importances_df=None, output_dir="outputs"):
    """
    Saves the results of a model pipeline run into a single CSV/JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / f"{model_name}_results.csv"

    # Flatten results into a dict
    flat_results = {
        'model': model_name,
        'best_params': json.dumps(best_params),  # stringify dict
    }
    # Add metrics to flat_results
    flat_results.update({k: float(v) for k, v in metrics.items()})

    # Start DataFrame with metrics and params
    results_df = pd.DataFrame([flat_results])

    # If feature importances exist, merge them as separate rows with a marker column
    if feature_importances_df is not None:
        feature_importances_df = feature_importances_df.copy()
        feature_importances_df['model'] = model_name
        feature_importances_df['type'] = 'feature_importance'

        # Add model level metrics row
        model_summary_df = results_df.copy()
        model_summary_df['type'] = 'summary'

        # Align columns
        combined_df = pd.concat([model_summary_df, feature_importances_df], ignore_index=True, sort=False)

        combined_df.to_csv(output_file, index=False)
    else:
        # Save only metrics if no feature importances
        results_df['type'] = 'summary'
        results_df.to_csv(output_file, index=False)

    print(f"Saved model results to {output_file}")

def _extract_feature_importances(
    pipeline: Pipeline,
    feature_names: pd.Index,
    model_name: str,
    ) -> pd.DataFrame | None:
    names = feature_names
    if "feature_selector" in pipeline.named_steps:
        selector = pipeline.named_steps["feature_selector"]
        support = selector.get_support()
        names = names[support]

    estimator = pipeline.named_steps["classifier"]
    importances = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coefs = estimator.coef_
        importances = coefs[0] if coefs.ndim > 1 else coefs

    if importances is None:
        print(f"{model_name} does not provide feature importances.")
        return None

    importance_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"Feature Importances for {model_name}:")
    print(importance_df.head())
    return importance_df

# Main Pipeline Workflow
def run_model_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    ):
    """
    Train/tune models defined in config/model_params.yaml.
    Workflow:
        1. Feature-engineer training data (and validation/test when provided).
        2. Fit models on training data using cross-validation + GridSearch.
        3. Evaluate best estimator on the validation split (acts as held-out set).

    Note: Testing on the real test split should occur after selecting the best
    model; this function reports validation metrics only.
    """

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "model_params.yaml"
    config = load_yaml_config(config_path)
    if not isinstance(config, dict):
        raise ValueError("model_params.yaml must define a mapping of model configurations.")

    results = []

    for model_name, model_cfg in config.items():
        if not model_cfg.get("enabled", False):
            continue

        print(f"\nRunning pipeline for: {model_name}")

        fe_config = model_cfg.get("feature_engineering")
        if fe_config is None:
            raise ValueError(f"feature_engineering block missing for model '{model_name}'.")

        (
            X_train_fe,
            X_val_fe,
            X_test_fe,
            y_train_fe,
            y_val_fe,
            y_test_fe,
        ) = feature_pipeline(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            config=fe_config,
        )

        estimator = _build_estimator(model_name, model_cfg)

        steps = []
        threshold = model_cfg.get("feature_selection")
        if threshold not in (None, "", False):
            selector = SelectFromModel(estimator=clone(estimator), threshold=threshold)
            steps.append(("feature_selector", selector))

        steps.append(("classifier", estimator))
        pipeline = Pipeline(steps)

        param_grid = {
            f"classifier__{param}": values
            for param, values in model_cfg.get("param_grid", {}).items()
        }

        eval_cfg = model_cfg.get("evaluation", {})
        primary_metric = eval_cfg.get("primary_metric", "cohen_kappa")
        additional_metrics = eval_cfg.get("additional_metrics", [])
        scoring_method = eval_cfg.get("scoring_method", primary_metric)
        average_method = eval_cfg.get("average", "macro")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_method,
            verbose=2,
            error_score = 'raise'
        )
        grid.fit(X_train_fe, y_train_fe)

        print(f"Best Params for {model_name}: {grid.best_params_}")

        y_pred = grid.best_estimator_.predict(X_val_fe)
        metrics = _evaluate_model(
            y_val_fe,
            y_pred,
            [primary_metric] + [m for m in additional_metrics if m != primary_metric],
            average_method,
        )

        print(f"{model_name} Validation Results: {metrics}")

        best_pipeline = grid.best_estimator_
        importance_df = _extract_feature_importances(
            best_pipeline,
            X_train_fe.columns,
            model_name,
        )

        outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
        _save_model_results(
            model_name,
            grid.best_params_,
            metrics,
            importance_df,
            outputs_dir,
        )

        results.append(
            {
                "model": model_name,
                "best_params": grid.best_params_,
                "metrics": metrics,
            }
        )

    return results

