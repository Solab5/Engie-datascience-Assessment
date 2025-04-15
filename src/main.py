import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.features.build_features import load_data, build_features
from src.models.train import train_models

def main():
    # Load and process data
    print("Loading data...")
    df = load_data(RAW_DATA_PATH)
    
    print("Building features...")
    processed_df = build_features(df)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print("Training models...")
    metrics = train_models(processed_df)
    
    print("\nModel Performance:")
    print("Completion Model:")
    print(f"Accuracy: {metrics['completion_metrics']['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['completion_metrics']['roc_auc']:.4f}")
    
    print("\nTiming Model:")
    print(f"MAE: {metrics['timing_metrics']['mae']:.2f} days")
    print(f"R²: {metrics['timing_metrics']['r2']:.4f}")

if __name__ == "__main__":
    main() 