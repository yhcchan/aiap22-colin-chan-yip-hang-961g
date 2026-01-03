from src.prep.load import download_database, load_data
from src.prep.data_split import run_train_val_test_split
from src.prep.pre_split_integrity import run_pre_split_integrity_pipe
from src.features.feature_engineering import (
    feature_pipeline,
    load_feature_engineering_config,
)
from src.model.model import run_model_pipeline
from src.config import DATA_URL, DATA_DIR, TABLE_NAME


def main():
    # Load raw data
    db_path = download_database(DATA_URL, DATA_DIR, force_refresh=False)
    df_raw = load_data(db_path, TABLE_NAME)

    # Clean integrities before splitting
    df_clean = run_pre_split_integrity_pipe(df_raw)

    # Create train/val/test splits
    splits = run_train_val_test_split(df_clean)
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    # Run feature engineering pipeline on splits
    fe_config = load_feature_engineering_config("config/global_feature_engineering.yaml")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = feature_pipeline(
        train_split["features"],
        val_split["features"],
        test_split["features"],
        train_split["target"],
        val_split["target"],
        test_split["target"],
        config=fe_config,
    )

    # Example placeholder for downstream modeling step

    engineered_splits= [X_train, X_val, X_test, y_train,  y_val, y_test]

    results = run_model_pipeline(
        X_train,
        X_val,
        y_train,
        y_val,
        X_test,
        y_test,
    )
    

if __name__ == "__main__":
    main()  
