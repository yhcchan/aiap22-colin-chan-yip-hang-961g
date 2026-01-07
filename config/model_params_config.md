# Model Parameters Configuration Guide

This guide describes the structure of `config/model_params.yaml`, which controls **per-model** settings (feature engineering, training, and evaluation) used by `run_model_pipeline` in `src/model/model.py`.

---

## File Layout
Each top-level key corresponds to a single model (e.g., `logistic_regression`). The block contains:

```yaml
model_name:
  enabled: bool
  feature_selection: str | float | null
  estimator_params: {...}
  class_weights: {...}
  feature_engineering: {...}
  param_grid: {...}
  evaluation: {...}
```

Models set `enabled: false` are skipped.

---

## Feature Selection
Controls optional `SelectFromModel` usage before the classifier step.

```yaml
feature_selection: "mean"  # or numeric threshold (e.g., 0.02). Omit/null to disable.
```

`"mean"` uses the average absolute coefficient/importance; numeric values act as cutoffs.

---

## Estimator Parameters
Set estimator-level kwargs (e.g., `max_iter`) before tuning. Use this to override sklearn defaults directly from YAML.

```yaml
estimator_params:
  max_iter: 1000
```

Leave empty (`{}`) to rely on sklearn defaults. The pipeline automatically injects computed class weights when enabled

---

## Class Weights
Enable automatic class-weight computation for imbalanced data.

```yaml
class_weights:
  enabled: true
  mode: balanced   # currently only 'balanced' supported
```

When enabled, `compute_class_weight` (sklearn) runs on the training labels and feeds the result into the estimator via its `class_weight` parameter.

---

## Feature Engineering
Embed the **same structure** used in `config/global_feature_engineering.yaml` under the `feature_engineering` key. Steps run in the listed order.

Example:

```yaml
feature_engineering:
  columns_to_impute:
    feature_a:
      dtype: numeric
      strategy: impute
      impute_method: median
      add_flag_column: true
  onehot_features:
    categorical_col:
      drop: BaseCategory
  log_transform_features:
    - skewed_feature
  scale_features:
    - numeric_feature
```

Any omitted step (empty dict/list or not defined) is skipped for that model.

---

## Hyperparameter Grid
Defines the GridSearchCV space for the model. Keys map directly to estimator parameters (prefixes are applied automatically inside `run_model_pipeline`).

```yaml
param_grid:
  C: [10, 50, 100]
  penalty: ["l1", "l2"]
  solver: ["liblinear"]
```

Leave empty (`{}`) to train without tuning.

---

## Evaluation Settings
Controls the scoring used during cross-validation and what metrics are reported on the validation split.

```yaml
evaluation:
  primary_metric: cohen_kappa          # metric used for final reporting
  scoring_method: balanced_accuracy    # metric used by GridSearchCV (default: primary_metric)
  additional_metrics: [accuracy, macro_f1, qwk]
  average: macro                       # averaging method for multi-class metrics
```

- `primary_metric` names the first metric returned in the summary.
- `scoring_method` overrides the GridSearch scoring function if desired.
- `additional_metrics` lists extra validation metrics (exclude duplicates of `primary_metric`).
- `average` affects metrics such as F1; options mirror sklearn (macro/micro/weighted/etc.).

---

## Adding New Models
1. Duplicate the block and change the top-level key.
2. Set `enabled: true`.
3. Provide a `feature_engineering` block (even if minimal) and the desired `param_grid`.
4. Adjust `feature_selection`, `estimator_params`, `class_weights`, and `evaluation` per model specifics.

`run_model_pipeline` automatically iterates through each enabled model, applies its feature engineering pipeline, performs cross-validation/grid search, and records metrics/results under `outputs/`.
