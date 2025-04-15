import pandas as pd
import numpy as np
from ..config import EARLY_PAYMENT_WINDOW

def load_data(file_path):
    return pd.read_csv(file_path)

def create_date_features(df):
    df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'])
    df['Year'] = df['Date of Transaction'].dt.year
    df['Month'] = df['Date of Transaction'].dt.month
    df['Day'] = df['Date of Transaction'].dt.day
    df['Weekday'] = df['Date of Transaction'].dt.day_name()
    return df

def create_account_features(df):
    account_df = df.sort_values('Date of Transaction').groupby('Account ID').first()[
        ['Account Kit Type', 'Payment Plan Deposit', 'Payment Plan Daily Rate', 
         'Payment Plan Loan Duration', 'Payment Plan Total Loan Value']
    ].reset_index()

    account_date_ranges = df.groupby('Account ID').agg({
        'Date of Transaction': ['min', 'max']
    })
    account_date_ranges.columns = ['first_transaction_date', 'last_transaction_date']
    account_date_ranges['account_lifetime_days'] = (account_date_ranges['last_transaction_date'] - 
                                                   account_date_ranges['first_transaction_date']).dt.days
    account_date_ranges = account_date_ranges.reset_index()

    account_final_status = df.sort_values('Date of Transaction').groupby('Account ID').last()['Type of Transaction']
    account_final_status = account_final_status.reset_index()
    account_final_status.columns = ['Account ID', 'final_status']

    account_df = account_df.merge(account_date_ranges, on='Account ID')
    account_df = account_df.merge(account_final_status, on='Account ID')
    return account_df

def create_payment_features(df, account_df):
    payment_df = df[df['Type of Transaction'] == 'Payment']
    payment_features = payment_df.groupby('Account ID').agg({
        'Value of Transaction': ['count', 'mean', 'std', 'sum', 'min', 'max'],
        'Number of Days Purchased': ['mean', 'sum'],
        'Date of Transaction': lambda x: x.diff().dt.days.mean()  
    })
    payment_features.columns = ['_'.join(col).strip() for col in payment_features.columns.values]
    payment_features.rename(columns={'Date of Transaction_<lambda>': 'avg_days_between_payments'}, inplace=True)
    payment_features = payment_features.reset_index()
    
    # Calculate loan paid percentage
    last_transactions = df.sort_values('Date of Transaction').groupby('Account ID').last()[['Total Amount of Loan Received', 'Payment Plan Total Loan Value']]
    last_transactions['loan_paid_percentage'] = (last_transactions['Total Amount of Loan Received'] / 
                                                last_transactions['Payment Plan Total Loan Value'] * 100)
    last_transactions = last_transactions.reset_index()[['Account ID', 'loan_paid_percentage']]
    
    # Add count of different transaction types
    transaction_type_counts = pd.crosstab(df['Account ID'], df['Type of Transaction']).reset_index()
    
    # Merge all features
    account_df = account_df.merge(payment_features, on='Account ID', how='left')
    account_df = account_df.merge(last_transactions, on='Account ID', how='left')
    account_df = account_df.merge(transaction_type_counts, on='Account ID', how='left')
    return account_df

def create_early_payment_features(df, account_df):
    early_df = df[(df['Date of Transaction'] - df.groupby('Account ID')['Date of Transaction'].transform('min')).dt.days <= EARLY_PAYMENT_WINDOW]
    early_payment_features = early_df[early_df['Type of Transaction'] == 'Payment'].groupby('Account ID').agg({
        'Value of Transaction': ['count', 'sum'],
        'Number of Days Purchased': ['sum']
    })
    early_payment_features.columns = ['early_payment_count', 'early_payment_amount', 'early_days_purchased']
    early_payment_features = early_payment_features.reset_index()
    return account_df.merge(early_payment_features, on='Account ID', how='left')

def create_derived_features(account_df):
    # Handle division by zero cases
    account_df['payment_frequency'] = np.where(
        account_df['account_lifetime_days'] > 0,
        account_df['Value of Transaction_count'] / account_df['account_lifetime_days'],
        0
    )
    
    account_df['avg_payment_size'] = np.where(
        account_df['Value of Transaction_count'] > 0,
        account_df['Value of Transaction_sum'] / account_df['Value of Transaction_count'],
        0
    )
    
    account_df['payment_to_loan_ratio'] = np.where(
        account_df['Payment Plan Total Loan Value'] > 0,
        account_df['Value of Transaction_sum'] / account_df['Payment Plan Total Loan Value'],
        0
    )
    
    account_df['days_purchased_ratio'] = np.where(
        account_df['account_lifetime_days'] > 0,
        account_df['Number of Days Purchased_sum'] / account_df['account_lifetime_days'],
        0
    )
    
    account_df['payment_amount_cv'] = np.where(
        (account_df['Value of Transaction_mean'] > 0) & (account_df['Value of Transaction_std'] > 0),
        account_df['Value of Transaction_std'] / account_df['Value of Transaction_mean'],
        0
    )
    
    account_df['repayment_pace'] = np.where(
        account_df['account_lifetime_days'] > 0,
        account_df['loan_paid_percentage'] / account_df['account_lifetime_days'],
        0
    )
    
    return account_df

def create_target_variables(account_df):
    account_df['loan_completed'] = account_df['final_status'].apply(lambda x: 1 if x == 'Payoff' else 0)
    account_df['loan_cancelled'] = account_df['final_status'].apply(lambda x: 1 if x == 'Cancellation' else 0)
    account_df['time_to_payoff'] = (account_df['last_transaction_date'] - account_df['first_transaction_date']).dt.days
    return account_df

def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    return df

def build_features(df):
    df = create_date_features(df)
    account_df = create_account_features(df)
    account_df = create_payment_features(df, account_df)
    account_df = create_early_payment_features(df, account_df)
    account_df = create_derived_features(account_df)
    account_df = create_target_variables(account_df)
    account_df = handle_missing_values(account_df)
    return account_df 