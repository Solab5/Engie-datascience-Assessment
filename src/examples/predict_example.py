import pandas as pd
from ..models.inference import LoanPredictor
from ..features.build_features import load_data

def main():
    # Initialize predictor
    predictor = LoanPredictor()
    
    # Load some data (you can replace this with your own data)
    df = load_data('../data/repayment_data.csv')
    
    # Make predictions for all accounts
    predictions = predictor.predict(df)
    print("\nPredictions for all accounts:")
    print(predictions.head())
    
    # Example: Get predictions for a single account
    account_id = df['Account ID'].iloc[0]
    account_data = df[df['Account ID'] == account_id]
    single_prediction = predictor.predict_single_account(account_data)
    
    print(f"\nPrediction for account {account_id}:")
    print(f"Completion probability: {single_prediction['completion_probability']:.2%}")
    print(f"Expected time to payoff: {single_prediction['expected_time_to_payoff_days']:.1f} days")
    print(f"Likely to complete: {'Yes' if single_prediction['likely_to_complete'] else 'No'}")

if __name__ == "__main__":
    main() 