import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import load_yaml_config


def train_val_test_split_data(df: pd.DataFrame, config: dict):
    """
    Split a DataFrame into training, validation, and testing sets before preprocessing.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        config (dict): Configuration dictionary containing split parameters. Expected keys in
            config['train_val_test_split']:
                - target_col (str): Name of the target column.
                - test_size (float): Proportion of full data used for the test split.
                - val_size (float): Proportion of full data used for the validation split.
                - stratify (bool): Whether to stratify splits using the target column.
                - random_state (int): PRNG seed for reproducibility.

    Returns:
        dict: Nested dictionary keyed by split name with "features" and "target" entries.
    """
    split_cfg = config.get("train_val_test_split", {})
    target_col = split_cfg["target_col"]
    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.1)
    stratify_flag = split_cfg.get("stratify", True)
    random_state = split_cfg.get("random_state", 42)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("The sum of test_size and val_size must be less than 1.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    stratify_vals = y if stratify_flag else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_vals, random_state=random_state
    )

    val_relative = val_size / (1 - test_size)
    stratify_train_val = y_train_val if stratify_flag else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        stratify=stratify_train_val,
        random_state=random_state,
    )

    def _reset_pair(features: pd.DataFrame, target: pd.Series) -> dict:
        return {
            "features": features.reset_index(drop=True),
            "target": target.reset_index(drop=True),
        }

    return {
        "train": _reset_pair(X_train, y_train),
        "val": _reset_pair(X_val, y_val),
        "test": _reset_pair(X_test, y_test),
    } 


def run_train_val_test_split(df: pd.DataFrame):
    """
    Load split configuration and run the full train/validation/test split pipeline.
    """
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "data_split.yaml"
    config = load_yaml_config(config_path)
    return train_val_test_split_data(df, config)
