# Feature Engineering Configuration Guide

This guide describes how to specify feature engineering configurations that feed into both global and model-specific feature engineering pipelines. These are used to set configurations in `global_feature_engineering.yaml` and `model_params.yaml`.

---


## Overall Structure

Global and per-model configs expect a top-level `feature_engineering` mapping:

```yaml
feature_engineering:
  columns_to_impute:
    ...
  onehot_features:
    ...
  log_transform_features:
    ...
  target_features:
    label:
      dtype: categorical
      ordinal: false
```

Steps are executed in the order they appear under `feature_engineering`.

---

## One-Hot Encoding for Categorical Features

Convert categorical columns into one-hot/binary indicator columns.  
Useful for nominal (unordered) categorical variables.  
Optionally, you can drop one specific category to avoid collinearity.

```yaml
onehot_features:
  example_col:
    drop: category_to_drop                # Optional. Name of category to drop, or null to keep all.
```

---

## Threshold-based Category Encoding

Convert numeric columns into categorical indicators based on a specified threshold.  
Creates new columns to reflect binary splits (above/below or equal to threshold).

```yaml
categorize_features:
  example_col:
    threshold: 3.14                      # Numeric threshold for splitting
    labels:
      above: "High" or 1                 # Label for value > threshold
      below_or_equal: "Low" or 0         # Label for value <= threshold
    new_column: "example_col_cat"        # Name for the new categorical column
    drop_original: false                 # Optional; drop source column if true
```

---

## Frequency / Count Encoding for Categorical Features

Augment categorical columns with frequency (count) encodings.  
This adds a new feature giving the count/frequency of each value occurrence.

```yaml
frequency_features:
  - example_col_1
  - example_col_2
```

---

## Ordinal Encoding for Ordinal Categorical Features

Map ordinal categorical features to integer rankings by specifying an order.  
Used for features where category order matters.

```yaml
ordinal_features:
  example_col_1:
    order: ["low", "medium", "high"]      # From lowest to highest
```

---

## Scaling for Numerical Features

Scale numeric features to zero mean and unit variance using StandardScaler.  
List all columns that require scaling.

```yaml
scale_features:
  - example_col_1
  - example_col_2
```

---

## Log-Transform for Numerical Features

Log-transform features to zero mean and unit variance.  
List all columns that require scaling.

```yaml
log_transform_features:
  - example_col_1
  - example_col_2
```

---

## Handling Missing or Invalid Values (Imputation)

Configure how missing or invalid values are handled for each column.  
Supports strategies like mean/median/mode/sentinel imputation or dropping rows.

```yaml
columns_to_impute:
  example_col_1:
    dtype: numeric               # 'numeric' or 'categorical'
    strategy: impute             # 'impute' to fill, 'drop' to remove
    impute_method: mean          # mean | median | mode | sentinel
    data_issue: missing          # 'missing' or 'invalid'
    sentinel_value: -999         # Used if impute_method: sentinel
    add_flag_column: true        # Add indicator column if true
    groupby_cols: [group_col]    # Group-wise imputation
    ordinal: false               # If categorical, set true for ordinal
    categories:                  # If ordinal, specify explicit order
      - cat1
      - cat2
      - cat3
```

---

## Target Feature Encoding

Configure the target variable properties, including whether it is ordinal and (if so) its value order.  
This is used for transformations and for ensuring correct processing in ML pipelines.

```yaml
target_features:
  label:
    dtype: categorical           # 'numeric' or 'categorical'
    ordinal: true                # true if target is ordinal
    order: [0, 1, 2]             # List of target classes in order (if applicable)
```
