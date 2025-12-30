from src.prep.load import download_database, load_data
from src.prep.data_split import run_train_val_test_split
from src.prep.pre_split_integrity import run_pre_split_integrity_pipe
#from src.features.feature_engineering import feature_pipeline
#from src.model.model import run_model_pipeline
from src.config import DATA_URL, DATA_DIR, TABLE_NAME

# main.py

def main():
    
    # Load data and immediately create train/val/test splits
    db_path = download_database(DATA_URL, DATA_DIR, force_refresh=False)
    df_raw = load_data(db_path, TABLE_NAME)      
    
    print(df_raw.head())
    
    df_clean = run_pre_split_integrity_pipe(df_raw)
    
    print(df_clean.head())
    
    #splits = run_train_val_test_split(df_clean)
  

    #val_split = processed_splits["val"]
    #test_split = processed_splits["test"]
    #X_val, y_val = val_split["features"], val_split["target"]
    #X_test, y_test = test_split["features"], test_split["target"]
    # Run feature engineering pipeline on data
    #X_train, X_val, X_test, y_train, y_val, y_test = feature_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)
    # Continue with modeling or saving
    #results = run_model_pipeline(X_train, X_test, y_train, y_test) # Export results to root/output/
    #print(f"Test Model Results: {results}")
    

if __name__ == "__main__":
     main()
