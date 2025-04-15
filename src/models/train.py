import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
import joblib
from src.config import RANDOM_STATE, TEST_SIZE, COMPLETION_MODEL_PATH, TIMING_MODEL_PATH

def prepare_features(df, target_type):
    drop_cols = ['Account ID', 'Account Kit Type', 'first_transaction_date', 
                'last_transaction_date', 'final_status', 'loan_cancelled', 
                'Payoff', 'Cancellation', 'account_lifetime_days']
    
    if target_type == 'completion':
        X = df.drop(drop_cols + ['loan_completed', 'time_to_payoff'], axis=1)
        y = df['loan_completed']
    else:  # timing
        completed_loans = df[df['loan_completed'] == 1]
        X = completed_loans.drop(drop_cols + ['loan_completed', 'time_to_payoff'], axis=1)
        y = completed_loans['time_to_payoff']
    
    return X, y

def train_completion_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

def train_timing_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, metrics

def save_model(model, path):
    joblib.dump(model, path)

def train_models(df):
    # Train completion model
    X_completion, y_completion = prepare_features(df, 'completion')
    completion_model, completion_metrics = train_completion_model(X_completion, y_completion)
    save_model(completion_model, COMPLETION_MODEL_PATH)
    
    # Train timing model
    X_timing, y_timing = prepare_features(df, 'timing')
    timing_model, timing_metrics = train_timing_model(X_timing, y_timing)
    save_model(timing_model, TIMING_MODEL_PATH)
    
    return {
        'completion_metrics': completion_metrics,
        'timing_metrics': timing_metrics
    } 