import pandas as pd
from sqlalchemy import create_engine, MetaData
import requests
from pathlib import Path
from urllib.parse import urlparse

def download_database(DATA_URL, DATA_DIR, force_refresh=False):
    """
    Download the phishing SQLite DB unless a cached copy exists (or refresh requested).
    Determines DB_PATH from DATA_URL and DATA_DIR.
    """
    data_url = str(DATA_URL).strip()
    if not data_url:
        raise ValueError("DATA_URL must be a non-empty string.")

    data_url_path = Path(urlparse(data_url).path)
    if not data_url_path.name:
        raise ValueError("DATA_URL must include a file name.")
    db_path = (DATA_DIR / data_url_path.name).expanduser()

    if db_path.exists() and not force_refresh:
        print(f"Using cached database at {db_path.resolve()}")
        return db_path

    print("Downloading phishing.db ...")
    response = requests.get(data_url, timeout=30)
    response.raise_for_status()
    db_path.write_bytes(response.content)
    print(f"Saved database to {db_path.resolve()}")
    return db_path

def load_data(path: str, table_name: str) -> pd.DataFrame:
    """
    Loads a table from a SQL database into a Pandas DataFrame.

    Args:
        path (str): SQLAlchemy connection string (e.g. 'sqlite:///mydb.db')
        table_name (str): Name of the table to load

    Returns:
        pd.DataFrame: DataFrame containing the table data
    """
    
    connection_target = str(path).strip()
    if not connection_target:
        raise ValueError("path must be a non-empty connection string.")

    if "://" in connection_target:
        connection_url = connection_target
    else:
        db_path = Path(connection_target).expanduser()
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found at '{db_path}'.")
        connection_url = f"sqlite:///{db_path.resolve().as_posix()}"

    table = str(table_name).strip()
    if not table:
        raise ValueError("table_name must be a non-empty string.")

    engine = create_engine(connection_url)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    if table not in metadata.tables:
        raise ValueError(f"Table '{table}' not found in the database.")

    df = pd.read_sql_table(table, con=engine)
    return df


