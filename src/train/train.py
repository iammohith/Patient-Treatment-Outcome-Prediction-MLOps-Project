import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
import argparse
import numpy as np

def train(data_dir, model_dir, metrics_file):
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }
    
    print(f"Model Metrics: {metrics}")
    
    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save model
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    print(f"Model saved to {os.path.join(model_dir, 'model.joblib')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory to save the model")
    parser.add_argument('--metrics', type=str, required=True, help="Path to save metrics JSON")
    args = parser.parse_args()
    
    train(args.data, args.model_dir, args.metrics)
