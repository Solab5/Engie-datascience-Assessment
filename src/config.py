import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Data paths
RAW_DATA_PATH = DATA_DIR / "repayment_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"

# Model paths
COMPLETION_MODEL_PATH = MODELS_DIR / "completion_model.pkl"
TIMING_MODEL_PATH = MODELS_DIR / "timing_model.pkl"

# Feature settings
EARLY_PAYMENT_WINDOW = 30  # days
TARGET_COLUMNS = {
    'completion': 'loan_completed',
    'timing': 'time_to_payoff'
}

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True) 