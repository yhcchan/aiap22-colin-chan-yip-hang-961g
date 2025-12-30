import pandas as pd

from src.prep import preprocess


def test_rename_columns_to_snake():
    df = pd.DataFrame({"No Of Image": [1], "DomainURL": [2]})
    renamed = preprocess.rename_columns_to_snake(df)
    assert renamed.columns.tolist() == ["no_of_image", "domain_url"]


def test_drop_duplicates_removes_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned = preprocess.drop_duplicates(df)
    assert len(cleaned) == 2
    assert cleaned.index.tolist() == [0, 1]


def test_drop_columns_uses_config():
    df = pd.DataFrame({"keep": [1], "drop_me": [2]})
    config = {"columns_to_drop": ["drop_me"]}
    result = preprocess.drop_columns(df, config)
    assert "drop_me" not in result.columns
    assert "keep" in result.columns


def test_rename_variables_to_snake():
    df = pd.DataFrame({"Industry": ["Financial Services"], "Other": ["MixedCase"]})
    config = {"columns_to_change": ["Industry"]}
    renamed = preprocess.rename_variables_to_snake(df, config)
    assert renamed.loc[0, "Industry"] == "financial_services"
    assert renamed.loc[0, "Other"] == "MixedCase"


def test_strip_whitespace():
    df = pd.DataFrame({"Industry": ["  Finance  "], "HostingProvider": [" AWS "]})
    config = {"columns_to_strip": ["Industry", "HostingProvider"]}
    stripped = preprocess.strip_whitespace(df, config)
    assert stripped.loc[0, "Industry"] == "Finance"
    assert stripped.loc[0, "HostingProvider"] == "AWS"


def test_run_preprocessing_pipeline(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "Unnamed: 0": [1, 1],
            "Industry": ["  Finance  ", "  Finance  "],
            "HostingProvider": [" AWS  ", " AWS  "],
            "Other": ["x", "x"],
        }
    )

 