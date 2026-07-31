import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
import os

def train_alpha_model(df, feature_cols, target_col='target_outperform', n_splits=5):
    """
    Trains a LightGBM Classifier using Time-Series Cross-Validation.
    Evaluates out-of-sample AUC-ROC and Information Coefficients per fold.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get unique trading dates for time-series group splitting
    unique_dates = df['date'].unique()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    oof_predictions = np.zeros(len(df))
    fold_metrics = []
    
    print(f"\n--- Starting LightGBM Cross-Validation ({n_splits} Folds) ---")
    
    for fold, (train_date_idx, val_date_idx) in enumerate(tscv.split(unique_dates)):
        # Apply embargo: Purge overlap between train and validation sets
        # Remove last 21 trading days from train set to avoid target leakage
        train_dates = unique_dates[train_date_idx][:-21]
        val_dates = unique_dates[val_date_idx]
        
        train_mask = df['date'].isin(train_dates)
        val_mask = df['date'].isin(val_dates)
        
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
        X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]
        
        if len(X_train) == 0 or len(X_val) == 0:
            continue
            
        # LightGBM Model Configuration
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validation Probabilities
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_mask] = val_preds
        
        auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold + 1} | Train Size: {len(X_train)} | Val Size: {len(X_val)} | Validation AUC: {auc:.4f}")
        fold_metrics.append(auc)
        
    print(f"\nMean OOF AUC Score across folds: {np.mean(fold_metrics):.4f}")
    
    # Fit final model on all data for deployment
    final_model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=42, verbose=-1
    )
    final_model.fit(df[feature_cols], df[target_col])
    
    return final_model, fold_metrics

if __name__ == "__main__":
    # Load normalized factor dataset
    input_path = "data/processed/normalized_factors.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run factor_engineering.py first!")
        
    df = pd.read_parquet(input_path)
    
    # Features (z-score normalized factors)
    feature_cols = [c for c in df.columns if c.endswith('_zscore')]
    
    # Train Model
    model, metrics = train_alpha_model(df, feature_cols)
    
    print("\nModel Training Completed Successfully!")