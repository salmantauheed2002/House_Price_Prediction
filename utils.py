import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_output_dir(output_dir='../outputs'):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def plot_distribution(data, column, log_transform=False, save_path=None):
    """Plot distribution of a column."""
    plt.figure(figsize=(10, 6))
    
    if log_transform:
        sns.histplot(np.log1p(data[column]), kde=True)
        plt.title(f'Log({column}) Distribution')
    else:
        sns.histplot(data[column], kde=True)
        plt.title(f'{column} Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_with_target(data, target_col, top_n=10, save_path=None):
    """Plot top correlations with target variable."""
    numeric_data = data.select_dtypes(include=[np.number])
    correlations = numeric_data[target_col].corr().abs().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    correlations.head(top_n + 1).plot(kind='bar')
    plt.title(f'Top {top_n} Features Correlated with {target_col}')
    plt.ylabel('Correlation Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return correlations

def print_model_summary(models):
    """Print summary of all trained models."""
    print("=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    for name, results in models.items():
        print(f"\n{results['name']}:")
        print(f"  Train RMSE: {results['train_rmse']:.4f}")
        print(f"  Train R²:   {results['train_r2']:.4f}")
        
        if 'val_rmse' in results:
            print(f"  Val RMSE:   {results['val_rmse']:.4f}")
            print(f"  Val R²:     {results['val_r2']:.4f}")
    
    print("=" * 70)

def save_predictions(test_ids, predictions, filename='submission.csv'):
    """Save predictions to CSV file."""
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    return submission

def print_project_summary(train_shape, test_shape, models, best_model_name):
    """Print project summary."""
    print("=" * 70)
    print("   HOUSE PRICE PREDICTION — PROJECT SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {train_shape[0]} houses, {train_shape[1]} features")
    print(f"Test set: {test_shape[0]} houses")
    print(f"\nModels Trained:")
    
    for name, results in models.items():
        emoji = "🥇" if name == best_model_name else ""
        print(f"   {results['name']} RMSE: {results.get('val_rmse', results['train_rmse']):.4f} {emoji}")
    
    print(f"\nBest Model: {models[best_model_name]['name']}")
    print(f"\nSubmission file saved as submission.csv ✅")
    print("=" * 70)