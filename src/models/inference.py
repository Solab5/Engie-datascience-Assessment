import pandas as pd
import numpy as np
import joblib
from ..config import COMPLETION_MODEL_PATH, TIMING_MODEL_PATH
from ..features.build_features import create_date_features, create_account_features, create_payment_features, create_early_payment_features, create_derived_features, create_target_variables, handle_missing_values

class LoanPredictor:
    def __init__(self):
        self.completion_model = joblib.load(COMPLETION_MODEL_PATH)
        self.timing_model = joblib.load(TIMING_MODEL_PATH)
        self.required_columns = [
            'Account ID', 'Account Kit Type', 'Date of Transaction',
            'Type of Transaction', 'Value of Transaction', 'Number of Days Purchased',
            'Total Amount of Loan Received', 'Payment Plan Total Loan Value',
            'Payment Plan Deposit', 'Payment Plan Daily Rate', 'Payment Plan Loan Duration'
        ]

    def validate_input_data(self, df):
        """Validate that all required columns are present in the input data."""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in the input data: {', '.join(missing_cols)}. "
                "Please ensure your CSV file contains all required columns: "
                f"{', '.join(self.required_columns)}"
            )
        return True

    def _prepare_features(self, df):
        """
        Prepare features for prediction using the same pipeline as training.
        """
        # Validate input data first
        self.validate_input_data(df)
        
        # Apply the preprocessing pipeline
        df = create_date_features(df)
        account_df = create_account_features(df)
        account_df = create_payment_features(df, account_df)
        account_df = create_early_payment_features(df, account_df)
        account_df = create_derived_features(account_df)
        account_df = create_target_variables(account_df)
        account_df = handle_missing_values(account_df)
        
        return account_df

    def predict(self, df):
        """
        Make predictions for both loan completion and time to payoff
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Predictions with account ID, completion probability, 
                         and expected time to payoff
        """
        # Prepare features
        processed_df = self._prepare_features(df)
        
        # Drop non-feature columns
        drop_cols = ['Account ID', 'Account Kit Type', 'first_transaction_date', 
                    'last_transaction_date', 'final_status', 'loan_cancelled', 
                    'Payoff', 'Cancellation', 'account_lifetime_days',
                    'loan_completed', 'time_to_payoff']
        
        features = processed_df.drop(drop_cols, axis=1, errors='ignore')
        
        # Make completion predictions
        completion_proba = self.completion_model.predict_proba(features)[:, 1]
        
        # Make timing predictions only for accounts likely to complete
        timing_predictions = np.zeros(len(features))
        likely_complete = completion_proba > 0.5
        if any(likely_complete):
            timing_predictions[likely_complete] = self.timing_model.predict(
                features[likely_complete]
            )
        
        # Create results dataframe
        results = pd.DataFrame({
            'Account ID': processed_df['Account ID'],
            'completion_probability': completion_proba,
            'expected_time_to_payoff_days': timing_predictions,
            'likely_to_complete': likely_complete
        })
        
        return results

    def predict_single_account(self, account_df):
        """
        Make predictions for a single account
        
        Args:
            account_df (pd.DataFrame): Transaction data for a single account
            
        Returns:
            dict: Prediction results
        """
        results = self.predict(account_df)
        return results.iloc[0].to_dict()