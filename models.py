import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def train_linear_regression(X_train, y_train, X_val=None, y_val=None):
    """Train Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': model,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'name': 'Linear Regression'
    }
    
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        results.update({
            'val_rmse': val_rmse,
            'val_r2': val_r2
        })
    
    return results

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost model."""
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': model,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'name': 'XGBoost'
    }
    
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        results.update({
            'val_rmse': val_rmse,
            'val_r2': val_r2
        })
    
    return results

def train_random_forest(X_train, y_train, X_val=None, y_val=None):
    """Train Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': model,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'name': 'Random Forest'
    }
    
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        results.update({
            'val_rmse': val_rmse,
            'val_r2': val_r2
        })
    
    return results

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation on a model."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return -scores.mean(), scores.std()

def evaluate_models(X_train, y_train, X_val, y_val):
    """Train and evaluate all models."""
    models = {}
    
    # Linear Regression
    print("Training Linear Regression...")
    lr_results = train_linear_regression(X_train, y_train, X_val, y_val)
    models['linear_regression'] = lr_results
    
    # XGBoost
    print("Training XGBoost...")
    xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
    models['xgboost'] = xgb_results
    
    # Random Forest
    print("Training Random Forest...")
    rf_results = train_random_forest(X_train, y_train, X_val, y_val)
    models['random_forest'] = rf_results
    
    return models

def get_best_model(models, metric='val_rmse'):
    """Get the best model based on validation metric."""
    best_model_name = None
    best_score = float('inf')
    
    for name, results in models.items():
        if metric in results:
            if results[metric] < best_score:
                best_score = results[metric]
                best_model_name = name
    
    return best_model_name, models[best_model_name] if best_model_name else None