# General imports
from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.utils import load_yaml_config

"""
feature_engineering.py

This module contains general-purpose feature engineering transformations to be used in machine learning pipelines. 
The functionality implemented here is not specific to any one model, but intended to create, transform, or preprocess
features in a standardized manner that can be re-used across different models and experiments. This may include 
missing value imputation, encoding, scaling, and other configurable transformations that prepare data for modeling.
"""

class MissingValueImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that imputes values using configuration-driven strategies.
    """

    def __init__(self, columns_config: dict | None = None):
        self.columns_config = columns_config or {}
        self.impute_values_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.impute_values_ = {}
        df = X

        for col, opts in self.columns_config.items():
            if col not in df.columns:
                continue

            strategy = opts.get("strategy", "impute")
            dtype = opts.get("dtype")
            impute_method = opts.get("impute_method")
            groupby_cols = opts.get("groupby_cols")
            ordinal = opts.get("ordinal", False)
            categories = opts.get("categories")

            if strategy == "drop":
                continue

            if dtype is None:
                dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

            if dtype == "categorical" and ordinal and impute_method == "median":
                if categories is None:
                    raise ValueError(f"Ordinal variable '{col}' must specify 'categories'.")

                cat_to_num = {cat: i for i, cat in enumerate(categories)}
                codes = df[col].map(cat_to_num)

                if groupby_cols:
                    medians = codes.groupby(df[groupby_cols].apply(tuple, axis=1)).median()
                    self.impute_values_[col] = {"group_medians": medians.to_dict(), "categories": categories}
                else:
                    self.impute_values_[col] = {"global_median": codes.median(), "categories": categories}

            elif dtype == "numeric":
                if groupby_cols:
                    if impute_method == "median":
                        medians = df.groupby(groupby_cols)[col].median().to_dict()
                        self.impute_values_[col] = {"group_medians": medians}
                    elif impute_method == "mean":
                        means = df.groupby(groupby_cols)[col].mean().to_dict()
                        self.impute_values_[col] = {"group_means": means}
                else:
                    if impute_method == "median":
                        self.impute_values_[col] = {"global_median": df[col].median()}
                    elif impute_method == "mean":
                        self.impute_values_[col] = {"global_mean": df[col].mean()}

            elif dtype == "categorical":
                if groupby_cols:
                    modes = df.groupby(groupby_cols)[col].agg(
                        lambda x: x.mode().iloc[0] if not x.mode().empty else None
                    ).to_dict()
                    self.impute_values_[col] = {"group_modes": modes}
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        self.impute_values_[col] = {"global_mode": mode_val.iloc[0]}

            if impute_method == "sentinel":
                sentinel_value = opts.get("sentinel_value")
                self.impute_values_[col] = {"sentinel": sentinel_value}

        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        for col, opts in self.columns_config.items():
            if col not in df.columns:
                continue

            strategy = opts.get("strategy", "impute")
            dtype = opts.get("dtype")
            data_issue = opts.get("data_issue")
            add_flag_column = opts.get("add_missing_indicator", False)
            groupby_cols = opts.get("groupby_cols")
            ordinal = opts.get("ordinal", False)

            if add_flag_column:
                df[f"{col}_{data_issue}"] = df[col].isna().astype(int)

            if strategy == "drop":
                df = df.drop(columns=[col])
                continue

            if dtype is None:
                dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

            impute_data = self.impute_values_.get(col, {})

            if dtype == "numeric":
                if groupby_cols and "group_medians" in impute_data:
                    medians = impute_data["group_medians"]
                    df[col] = df.apply(
                        lambda row: medians.get(tuple(row[groupby_cols]), row[col]) if pd.isnull(row[col]) else row[col],
                        axis=1,
                    )
                elif groupby_cols and "group_means" in impute_data:
                    means = impute_data["group_means"]
                    df[col] = df.apply(
                        lambda row: means.get(tuple(row[groupby_cols]), row[col]) if pd.isnull(row[col]) else row[col],
                        axis=1,
                    )
                elif "global_median" in impute_data:
                    df[col] = df[col].fillna(impute_data["global_median"])
                elif "global_mean" in impute_data:
                    df[col] = df[col].fillna(impute_data["global_mean"])
                elif "sentinel" in impute_data:
                    df[col] = df[col].fillna(impute_data["sentinel"])

            elif dtype == "categorical":
                if ordinal and "group_medians" in impute_data:
                    categories = impute_data["categories"]
                    cat_to_num = {cat: i for i, cat in enumerate(categories)}
                    num_to_cat = {i: cat for i, cat in enumerate(categories)}
                    codes = df[col].map(cat_to_num)
                    df[col] = df.apply(
                        lambda row: num_to_cat[
                            round(impute_data["group_medians"].get(tuple(row[groupby_cols]), codes[row.name]))
                        ]
                        if pd.isnull(row[col])
                        else row[col],
                        axis=1,
                    )
                elif ordinal and "global_median" in impute_data:
                    categories = impute_data["categories"]
                    cat_to_num = {cat: i for i, cat in enumerate(categories)}
                    num_to_cat = {i: cat for i, cat in enumerate(categories)}
                    codes = df[col].map(cat_to_num)
                    codes = codes.fillna(impute_data["global_median"])
                    df[col] = codes.round().astype(int).map(num_to_cat)
                elif "group_modes" in impute_data:
                    modes = impute_data["group_modes"]
                    df[col] = df.apply(
                        lambda row: modes.get(tuple(row[groupby_cols]), row[col]) if pd.isnull(row[col]) else row[col],
                        axis=1,
                    )
                elif "global_mode" in impute_data:
                    df[col] = df[col].fillna(impute_data["global_mode"])
                elif "sentinel" in impute_data:
                    df[col] = df[col].fillna(impute_data["sentinel"])

        return df


class ColumnScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies StandardScaler to specific columns while leaving others unchanged.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []
        self.scaler_: StandardScaler | None = None

    def fit(self, X: pd.DataFrame, y=None):
        if not self.columns:
            return self
        existing_cols = [col for col in self.columns if col in X.columns]
        if not existing_cols:
            return self
        self.columns = existing_cols
        self.scaler_ = StandardScaler().fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame):
        if not self.columns or self.scaler_ is None:
            return X
        df = X.copy()
        df[self.columns] = self.scaler_.transform(df[self.columns])
        return df


class ThresholdCategorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Creates categorical indicators based on thresholds defined for numeric columns.
    """

    def __init__(self, categorize_config: dict | None = None):
        self.categorize_config = categorize_config or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if not self.categorize_config:
            return X
        df = X.copy()
        for col, opts in self.categorize_config.items():
            if col not in df.columns:
                continue
            threshold = opts.get("threshold")
            if threshold is None:
                raise ValueError(f"Categorization threshold missing for column '{col}'.")
            labels = opts.get("labels", {})
            above_label = labels.get("above", 1)
            below_label = labels.get("below_or_equal", 0)
            new_column = opts.get("new_column", f"{col}_categorized")
            mask = df[col] > threshold
            df[new_column] = mask.replace({True: above_label, False: below_label})
        return df


class OrdinalMapperTransformer(BaseEstimator, TransformerMixin):
    """
    Maps categorical columns to ordinal integer codes using configuration-defined order.
    """

    def __init__(self, ordinal_config: dict | None = None):
        self.ordinal_config = ordinal_config or {}
        self.mappings_: dict[str, dict] = {}

    def fit(self, X, y=None):
        self.mappings_ = {}
        for col, opts in self.ordinal_config.items():
            order = opts.get("order")
            if not order:
                raise ValueError(f"Order not specified for ordinal column '{col}'.")
            self.mappings_[col] = {cat: idx for idx, cat in enumerate(order)}
        return self

    def transform(self, X: pd.DataFrame):
        if not self.mappings_:
            return X
        df = X.copy()
        for col, mapping in self.mappings_.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        return df


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Applies OneHotEncoder to specified columns and returns a DataFrame with expanded columns.
    """

    def __init__(self, onehot_config: dict | None = None):
        self.onehot_config = onehot_config or {}
        self.encoder_: OneHotEncoder | None = None
        self.columns: list[str] = []
        self.feature_names_: list[str] = []
        self.feature_to_columns_: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        if not self.onehot_config:
            return self
        self.columns = [col for col in self.onehot_config.keys() if col in X.columns]
        if not self.columns:
            return self
        drop_values = [self.onehot_config[col].get("drop") for col in self.columns]
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=drop_values,
        ).fit(X[self.columns])
        raw_feature_names = self.encoder_.get_feature_names_out(self.columns)

        def clean_columns(cols):
            return [name.split("__", 1)[-1] if "__" in name else name for name in cols]

        cleaned_feature_names = clean_columns(raw_feature_names)
        self.feature_names_ = cleaned_feature_names
        self.feature_to_columns_ = {}
        for raw, clean in zip(raw_feature_names, cleaned_feature_names):
            source_col = raw.split("__", 1)[0] if "__" in raw else raw
            self.feature_to_columns_.setdefault(source_col, []).append(clean)
        return self

    def transform(self, X: pd.DataFrame):
        if not self.columns or self.encoder_ is None:
            return X
        df = X.copy()
        encoded = pd.DataFrame(
            self.encoder_.transform(df[self.columns]),
            columns=self.feature_names_,
            index=df.index,
        )
        ordered_parts = []
        for col in df.columns:
            if col in self.feature_to_columns_:
                ordered_parts.append(encoded[self.feature_to_columns_[col]])
            else:
                ordered_parts.append(df[[col]])
        return pd.concat(ordered_parts, axis=1)


class TargetTransformer(BaseEstimator, TransformerMixin):
    """
    Handles target-specific scaling or encoding based on YAML configuration.
    """

    def __init__(self, target_config: dict | None = None):
        self.target_config = target_config or {}
        self.target_col: str | None = None
        self.dtype: str | None = None
        self.ordinal: bool = False
        self.categories: list | None = None
        self.scaler_: StandardScaler | None = None
        self.cat_to_int_: dict | None = None

    def fit(self, y: pd.Series, X=None):
        if y is None:
            return self
        target_series = y if isinstance(y, pd.Series) else y.squeeze()
        self.target_col = target_series.name or "target"
        col_config = self.target_config.get(self.target_col)
        if not col_config:
            raise ValueError(f"Target column '{self.target_col}' not found in target_features config.")
        self.dtype = col_config.get("dtype")
        self.ordinal = col_config.get("ordinal", False)
        self.categories = col_config.get("order")

        if self.dtype == "numeric":
            self.scaler_ = StandardScaler().fit(target_series.to_frame())
        elif self.dtype == "categorical" and self.ordinal:
            if not self.categories:
                raise ValueError("Ordinal target specified but 'order' missing in config.")
            self.cat_to_int_ = {cat: idx for idx, cat in enumerate(self.categories)}
        return self

    def transform(self, y: pd.Series):
        if y is None or self.dtype is None:
            return y
        target_series = y.copy()
        if self.dtype == "numeric" and self.scaler_ is not None:
            transformed = self.scaler_.transform(target_series.to_frame())
            return pd.Series(transformed.ravel(), index=target_series.index, name=target_series.name)
        if self.dtype == "categorical" and self.ordinal and self.cat_to_int_:
            return target_series.map(self.cat_to_int_)
        return target_series


def build_global_feature_pipeline(config: dict) -> Pipeline:
    """
    Builds an sklearn Pipeline composed of configuration-driven transformers.
    """
    steps = []

    impute_cfg = config.get("columns_to_impute", {})
    if impute_cfg:
        steps.append(("imputer", MissingValueImputerTransformer(impute_cfg)))

    categorize_cfg = config.get("categorize_features", {})
    if categorize_cfg:
        steps.append(("categorizer", ThresholdCategorizerTransformer(categorize_cfg)))

    if not steps:
        steps.append(("identity", FunctionTransformer(lambda X: X, validate=False)))

    return Pipeline(steps)


def global_feature_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_val: pd.Series | None = None,
    y_test: pd.Series | None = None,
):
    """
    Runs the feature engineering pipeline on the supplied train/val/test splits
    and returns processed feature and target splits.
    """

    if X_train is None:
        raise ValueError("Training features must be provided.")

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "feature_engineering.yaml"
    config = load_yaml_config(config_path)

    global_features_pipe = build_global_feature_pipeline(config)
    X_train_processed = global_features_pipe.fit_transform(X_train)
    X_val_processed = global_features_pipe.transform(X_val) if X_val is not None else None
    X_test_processed = global_features_pipe.transform(X_test) if X_test is not None else None

    target_features = config.get("target_features", {})
    target_transformer = TargetTransformer(target_features) if target_features else None

    y_train_processed = y_val_processed = y_test_processed = None
    if y_train is not None and target_transformer is not None:
        target_transformer.fit(y_train)
        y_train_processed = target_transformer.transform(y_train)
        if y_val is not None:
            y_val_processed = target_transformer.transform(y_val)
        if y_test is not None:
            y_test_processed = target_transformer.transform(y_test)
    else:
        y_train_processed = y_train
        y_val_processed = y_val
        y_test_processed = y_test

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train_processed,
        y_val_processed,
        y_test_processed,
    )
