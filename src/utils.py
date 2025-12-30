import re
import yaml 
import pandas as pd

def to_snake(text: str) -> str:
    
    """
    Convert a string to snake_case.

    - Strips leading/trailing whitespace
    - Replaces spaces and hyphens with underscores
    - Inserts underscores before CamelCase transitions
    - Converts to lowercase

    Parameters:
    -----------
    text : str
        The input string to convert.

    Returns:
    --------
    str
        The snake_case version of the input.
    """
    if pd.isnull(text):
        return text
    
    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s\-]+', '_', text)

    # Add underscore before camelCase or PascalCase transitions (e.g., "SensorValue" → "sensor_value")
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)

    # Convert to lowercase
    text = text.lower()

    return text
    
def load_yaml_config(config_path: str) -> dict:
    """
    Load and parse a YAML configuration file into a Python dictionary.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML content as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f) 

