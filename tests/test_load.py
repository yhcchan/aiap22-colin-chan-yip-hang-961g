import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.prep import load


class DummyResponse:
    """Minimal requests.Response stand-in for download tests."""

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _create_sqlite_db(tmp_path: Path, table_name: str = "websites") -> tuple[Path, pd.DataFrame]:
    """Create a temporary SQLite database for load_data tests."""
    db_path = tmp_path / "sample.db"
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "domain": ["foo.com", "bar.com"],
            "label": [0, 1],
        }
    )
    conn = sqlite3.connect(db_path)
    try:
        data.to_sql(table_name, conn, index=False, if_exists="replace")
    finally:
        conn.close()
    return db_path, data


def test_download_database_uses_cached_file(tmp_path, monkeypatch):
    cached = tmp_path / "phishing.db"
    cached.write_bytes(b"cached-bytes")

    def fake_get(*args, **kwargs):
        raise AssertionError("download_database should not hit the network when cache exists")

    monkeypatch.setattr(load.requests, "get", fake_get)
    result = load.download_database("https://example.com/phishing.db", tmp_path)
    assert result == cached


def test_download_database_fetches_and_saves(tmp_path, monkeypatch):
    url = "https://example.com/phishing.db"
    response = DummyResponse(b"db-bytes")

    def fake_get(requested_url, timeout):
        assert requested_url == url
        assert timeout == 30
        return response

    monkeypatch.setattr(load.requests, "get", fake_get)
    db_path = load.download_database(url, tmp_path, force_refresh=True)
    assert db_path.exists()
    assert db_path.read_bytes() == b"db-bytes"


def test_download_database_requires_filename(tmp_path):
    with pytest.raises(ValueError):
        load.download_database("https://example.com/", tmp_path)


def test_load_data_from_file_path(tmp_path):
    db_path, expected = _create_sqlite_db(tmp_path)
    loaded = load.load_data(str(db_path), "websites")
    pd.testing.assert_frame_equal(loaded, expected)


def test_load_data_from_connection_url(tmp_path):
    db_path, expected = _create_sqlite_db(tmp_path, table_name="pages")
    url = f"sqlite:///{db_path.as_posix()}"
    loaded = load.load_data(url, "pages")
    pd.testing.assert_frame_equal(loaded, expected)


def test_load_data_missing_table(tmp_path):
    db_path, _ = _create_sqlite_db(tmp_path)
    with pytest.raises(ValueError, match="Table 'missing' not found"):
        load.load_data(str(db_path), "missing")


def test_load_data_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.db"
    with pytest.raises(FileNotFoundError):
        load.load_data(str(missing_path), "websites")
