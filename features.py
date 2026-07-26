import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def create_features(df, is_train=True):
    """Create engineered features."""
    df = df.copy()
    
    # Create total square feet features
    if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Create total bathrooms feature
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
    
    # Create age features
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    return df

def encode_categorical_features(train_df, test_df, categorical_cols):
    """Encode categorical features using Label Encoding."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    label_encoders = {}
    
    for col in categorical_cols:
        if col in train_df.columns and col in test_df.columns:
            le = LabelEncoder()
            # Fit on combined data to handle all categories
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            le.fit(combined.astype(str))
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            label_encoders[col] = le
    
    return train_df, test_df, label_encoders

def scale_numeric_features(train_df, test_df, numeric_cols):
    """Scale numeric features using StandardScaler."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    scaler = StandardScaler()
    
    # Get available numeric columns
    available_cols = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
    
    if available_cols:
        train_df[available_cols] = scaler.fit_transform(train_df[available_cols])
        test_df[available_cols] = scaler.transform(test_df[available_cols])
    
    return train_df, test_df, scaler

def prepare_features(train_df, test_df, target_col='SalePrice'):
    """Prepare features for modeling."""
    # Create features
    train_df = create_features(train_df, is_train=True)
    test_df = create_features(test_df, is_train=False)
    
    # Get feature columns (exclude target and ID)
    feature_cols = [col for col in train_df.columns if col not in ['Id', target_col] and train_df[col].dtype in ['int64', 'float64', 'object']]
    
    # Separate numeric and categorical
    numeric_cols = [col for col in feature_cols if train_df[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in feature_cols if train_df[col].dtype == 'object']
    
    # Encode categorical features
    train_df, test_df, label_encoders = encode_categorical_features(
        train_df, test_df, categorical_cols
    )
    
    # Scale numeric features
    train_df, test_df, scaler = scale_numeric_features(
        train_df, test_df, numeric_cols
    )
    
    # Prepare X and y
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].copy() if target_col in train_df.columns else None
    
    return X_train, X_test, y_train, feature_cols, label_encoders, scaler