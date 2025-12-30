# General imports
from collections.abc import Mapping
from pathlib import Path
import pandas as pd


# Custom function imports
from src.utils import to_snake, load_yaml_config

### PRE-SPLIT INTEGRITY FUNCTIONS

# Function to rename columns to snake_case
def rename_columns_to_snake(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename all column names in a pandas DataFrame to snake_case.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame whose columns you want to rename.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with renamed columns in snake_case format.

    """
    df = df.copy()
    df.columns = [to_snake(col) for col in df.columns]

    return df

# Function to drop duplicate rows from the df
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate rows from the DataFrame and resets the index.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    df = df.copy()
    n_duplicates = df.duplicated().sum()
    df = df.drop_duplicates(ignore_index=True)
    print(f"[drop_duplicates] dropped {n_duplicates} duplicate row(s).")
    return df

# Function to drop unnecessary columns from the df
def drop_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    columns_to_drop = config.get("columns_to_drop", [])
    df = df.drop(columns = columns_to_drop, axis = 1, errors = 'ignore')
    return df

# Function to standardise all inconsistent schema naming to snake_case
def rename_variables_to_snake(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    For columns listed in config['columns_to_change'],
    apply to_snake_case to all values in those columns.
    """
    df = df.copy()
    cols = config.get("columns_to_change", [])
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(to_snake)
    return df

# Function to strip whitespace
def strip_whitespace(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    cols = config.get("columns_to_strip", [])
    for col in cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df

# Function to recode variables
def recode_variables(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    recode_config = config.get("columns_to_recode", {})

    if not isinstance(recode_config, dict):
        return df

    for col, replacements in recode_config.items():
        if col not in df.columns or replacements is None:
            continue

        mapping = {}
        if isinstance(replacements, dict):
            mapping = replacements
        elif isinstance(replacements, list):
            for pair in replacements:
                if isinstance(pair, dict) and "from" in pair and "to" in pair:
                    mapping[pair["from"]] = pair["to"]

        if mapping:
            df[col] = df[col].replace(mapping)

    return df

# Function to replace impossible values with Nan
def replace_impossible_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    replace_config = config.get("columns_to_replace_impossible_values", {})
    if not isinstance(replace_config, Mapping):
        return df

    for col, bounds in replace_config.items():
        if col not in df.columns or not isinstance(bounds, Mapping):
            continue

        lower = bounds.get("lower_bound")
        upper = bounds.get("upper_bound")

        if lower is None and upper is None:
            continue

        col_data = df[col]
        mask = None

        if lower is not None:
            mask = (col_data < lower) if mask is None else (mask | (col_data < lower))
        if upper is not None:
            mask = (col_data > upper) if mask is None else (mask | (col_data > upper))

        if mask is not None:
            df.loc[mask.fillna(False), col] = pd.NA

    return df

def run_pre_split_integrity_pipe(df: pd.DataFrame) -> pd.DataFrame:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "pre_split_integrity.yaml"
    config = load_yaml_config(config_path)
    return (
        df
        .pipe(drop_duplicates)
        .pipe(drop_columns, config) 
        .pipe(replace_impossible_values, config)
        .pipe(recode_variables, config)
        .pipe(strip_whitespace, config)
    )
