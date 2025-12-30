import yaml
import json
import numpy as np  
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
from pathlib import Path

# Custom function imports
from src.utils import load_yaml_config

# Compute Class Weights
def get_class_weights(y, mode='balanced'):
    if mode == 'balanced':
        classes = np.array(sorted(set(y)))  # Ensure it's a NumPy array
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    else:
        raise ValueError("Only 'balanced' mode is implemented currently.")


# Evaluation Metrics
def evaluate_model(y_true, y_pred, metrics, average="macro"):
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

def save_model_results(model_name, best_params, metrics, feature_importances_df=None, output_dir="outputs"):
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


# Model Mapping
MODEL_MAPPING = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "lightgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
    "xgboost": XGBClassifier

}

# Main Pipeline Workflow
def run_model_pipeline(X_train, X_test, y_train, y_test):

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "model.yaml"
    config = load_yaml_config(config_path)

    results = []

    class_weights = None
    if config.get("class_weights", {}).get("enabled", False):
        class_weights = get_class_weights(y_train, config["class_weights"].get("mode", "balanced"))
        print(f"Computed class weights: {class_weights}")

    for model_name, model_conf in config["models"].items():
        if not model_conf.get("enabled", False):
            continue

        print(f"\nRunning pipeline for: {model_name}")
        ModelClass = MODEL_MAPPING[model_name]

        # Base model kwargs
        base_model_kwargs = {}
        if model_name in ["random_forest", "logistic_regression"] and class_weights:
            base_model_kwargs['class_weight'] = class_weights
        if model_name == "catboost":
            base_model_kwargs['loss_function'] = 'MultiClass'
            base_model_kwargs['auto_class_weights'] = model_conf['param_grid'].get('auto_class_weights', ["Balanced"])[0]
            base_model_kwargs['verbose'] = False
        if model_name == "xgboost":
            base_model_kwargs['objective'] = 'multi:softprob'
            base_model_kwargs['verbosity'] = 0
            base_model_kwargs['tree_method'] = 'hist'
        
        base_model = ModelClass(**base_model_kwargs)

        # Determine threshold per model from YAML
        model_thresholds = config.get("feature_selection", {}).get("thresholds", {})
        threshold = model_thresholds.get(model_name, "mean")  # default fallback to 'mean'

        # Feature Selector with per-model threshold
        selector = SelectFromModel(estimator=clone(base_model), threshold=threshold)

        # Pipeline
        pipeline = Pipeline([
            ('feature_selector', selector),
            ('classifier', base_model)
        ])

        # GridSearchCV Param Grid
        param_grid = {f"classifier__{k}": v for k, v in model_conf["param_grid"].items() if k != 'auto_class_weights'}

        # Stratified CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=config["evaluation"]["scoring_method"], verbose=2)
        grid.fit(X_train, y_train)

        print(f"Best Params for {model_name}: {grid.best_params_}")

        # Evaluate on Test Set
        y_pred = grid.best_estimator_.predict(X_test)
        eval_results = evaluate_model(
            y_test, y_pred,
            [config["evaluation"]["primary_metric"]] + config["evaluation"].get("additional_metrics", []),
            config["evaluation"].get("average", "macro")
        )

        print(f"{model_name} Evaluation Results: {eval_results}")

        # Retrieve Feature Importances
        selector = grid.best_estimator_.named_steps['feature_selector']
        estimator = selector.estimator_

        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            importances = estimator.coef_[0]
        else:
            importances = None

        if importances is not None:
            selected_features = X_train.columns[selector.get_support()]
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': importances[selector.get_support()]
            }).sort_values(by='importance', ascending=False)

            print(f"Feature Importances for {model_name}:")
            print(importance_df)
        else:
            print(f"{model_name} does not provide feature importances.")

        results.append({
            "model": model_name,
            "best_params": grid.best_params_,
            "metrics": eval_results
        })

        outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
        save_model_results(model_name, grid.best_params_, eval_results, importance_df, outputs_dir)

    return results