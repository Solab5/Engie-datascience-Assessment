import pandas as pd
import numpy as np
import joblib
from ..config import COMPLETION_MODEL_PATH, TIMING_MODEL_PATH
from ..features.build_features import build_features

class LoanPredictor:
    def __init__(self):
        self.completion_model = joblib.load(COMPLETION_MODEL_PATH)
        self.timing_model = joblib.load(TIMING_MODEL_PATH)
        self.feature_columns = None  # Will be set during first prediction

    def _prepare_features(self, df):
        # Build features using the same pipeline as training
        processed_df = build_features(df)
        
        # Store feature columns for future reference
        if self.feature_columns is None:
            self.feature_columns = processed_df.columns.tolist()
        
        # Ensure all required columns are present
        required_cols = ['Account ID', 'Account Kit Type', 'first_transaction_date', 
                        'last_transaction_date', 'final_status', 'loan_cancelled', 
                        'Payoff', 'Cancellation', 'account_lifetime_days', 
                        'loan_completed', 'time_to_payoff']
        
        features = processed_df.drop(required_cols, axis=1, errors='ignore')
        return features

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
        features = self._prepare_features(df)
        
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
            'Account ID': df['Account ID'].unique(),
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