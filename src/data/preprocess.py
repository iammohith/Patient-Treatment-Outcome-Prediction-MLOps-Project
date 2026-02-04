import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import argparse

def preprocess(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Drop Patient_ID as it's not a predictive feature
    if 'Patient_ID' in df.columns:
        df = df.drop(columns=['Patient_ID'])
        
    # Categorical columns to encode
    cat_cols = ['Gender', 'Condition', 'Drug_Name', 'Side_Effects']
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # Save encoders for inference usage
    joblib.dump(encoders, os.path.join(output_dir, 'encoders.pkl'))
    
    # Split features and target
    target = 'Improvement_Score'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features (optional but good practice)
    # Identifying numerical columns that are not the encoded ones
    # In this dataset: Age, Dosage_mg, Treatment_Duration_days
    num_cols = ['Age', 'Dosage_mg', 'Treatment_Duration_days']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Save processed data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Preprocessing completed. Artifacts saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to raw input CSV")
    parser.add_argument('--output', type=str, required=True, help="Directory to save processed files")
    args = parser.parse_args()
    
    preprocess(args.input, args.output)
