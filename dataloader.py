import pandas as pd
import numpy as np
from .config import TRAIN_PATH, TEST_PATH, NONE_COLUMNS, ZERO_COLUMNS

def load_data():
    """Load training and test data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test

def handle_missing_values(df, is_train=True):
    """Handle missing values in the dataset."""
    df = df.copy()
    
    # Fill columns with 'None'
    for col in NONE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # Fill columns with 0
    for col in ZERO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Fill LotFrontage with median
    if 'LotFrontage' in df.columns:
        if is_train:
            median_lot_frontage = df['LotFrontage'].median()
        else:
            median_lot_frontage = df['LotFrontage'].median()
        df['LotFrontage'] = df['LotFrontage'].fillna(median_lot_frontage)
    
    # Fill Electrical with mode
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    return df

def get_missing_info(df):
    """Get information about missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"Columns with missing values: {len(missing)}")
    if len(missing) > 0:
        print(missing)
    return missing