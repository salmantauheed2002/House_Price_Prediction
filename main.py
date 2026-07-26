import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from .dataloader import load_data, handle_missing_values, get_missing_info
from .features import prepare_features
from .models import evaluate_models, get_best_model
from .utils import (
    create_output_dir, plot_distribution, plot_correlation_with_target,
    print_model_summary, save_predictions, print_project_summary
)

def main():
    """Main execution function."""
    print("Starting House Price Prediction Pipeline...")
    print("=" * 70)
    
    # Create output directory
    create_output_dir()
    
    # Load data
    print("\n1. Loading data...")
    train, test = load_data()
    
    # Handle missing values
    print("\n2. Handling missing values...")
    print("Training set:")
    get_missing_info(train)
    train = handle_missing_values(train, is_train=True)
    
    print("\nTest set:")
    get_missing_info(test)
    test = handle_missing_values(test, is_train=False)
    
    # Exploratory analysis
    print("\n3. Exploratory Data Analysis...")
    plot_distribution(train, 'SalePrice', log_transform=False, 
                     save_path='../outputs/saleprice_dist.png')
    plot_distribution(train, 'SalePrice', log_transform=True,
                     save_path='../outputs/log_saleprice_dist.png')
    
    # Show correlation
    print("\nTop correlations with SalePrice:")
    correlations = plot_correlation_with_target(train, 'SalePrice', top_n=10,
                                                save_path='../outputs/correlations.png')
    
    # Prepare features
    print("\n4. Feature Engineering...")
    X_train_full, X_test, y_train_full, feature_cols, label_encoders, scaler = prepare_features(
        train, test, target_col='SalePrice'
    )
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Split training data
    print("\n5. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train models
    print("\n6. Training models...")
    models = evaluate_models(X_train, y_train, X_val, y_val)
    
    # Print model summary
    print_model_summary(models)
    
    # Get best model
    best_model_name, best_model_results = get_best_model(models)
    print(f"\nBest model: {best_model_results['name']}")
    
    # Retrain best model on full training data
    print("\n7. Retraining best model on full data...")
    best_model = best_model_results['model']
    best_model.fit(X_train_full, y_train_full)
    
    # Make predictions
    print("\n8. Making predictions...")
    predictions = best_model.predict(X_test)
    
    # Convert from log scale if needed
    if y_train_full.max() > 1000000:  # If data was not log-transformed
        pass
    else:
        predictions = np.expm1(predictions)
    
    # Save predictions
    save_predictions(test['Id'], predictions, 'submission.csv')
    
    # Print final summary
    print_project_summary(train.shape, test.shape, models, best_model_name)
    
    return models, best_model_name, predictions

if __name__ == "__main__":
    models, best_model, predictions = main()