import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

def purged_time_splits(dates_sorted, n_splits=5, embargo=21):
    """
    Splits dates sequentially without shuffling, enforcing an embargo period
    between training and validation to prevent future leakage.
    """
    unique_dates = np.array(sorted(pd.unique(dates_sorted)))
    fold_size = len(unique_dates) // (n_splits + 1)
    
    splits = []
    for k in range(1, n_splits + 1):
        tr_end = fold_size * k
        va_start = tr_end + embargo
        va_end = va_start + fold_size
        
        train_dates = set(unique_dates[:tr_end])
        valid_dates = set(unique_dates[va_start:va_end])
        
        if len(valid_dates) > 0:
            splits.append((train_dates, valid_dates))
            
    return splits

def compute_daily_ic(df, score_col='score', target_col='fwd_ret_21d'):
    """Calculates Spearman Rank Information Coefficient (IC) per date."""
    def _ic(g):
        g = g.dropna(subset=[score_col, target_col])
        if len(g) < 5:
            return np.nan
        return spearmanr(g[score_col], g[target_col]).correlation

    ic_series = df.groupby('date').apply(_ic).dropna()
    mean_ic = ic_series.mean()
    ic_std = ic_series.std(ddof=1)
    ic_ir = mean_ic / (ic_std + 1e-12)
    t_stat = ic_ir * np.sqrt(len(ic_series))
    
    return mean_ic, ic_ir, t_stat

def compute_ece(p_cal, y_cal, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_cal > lo) & (p_cal <= hi)
        if mask.sum() > 0:
            acc = y_cal[mask].mean()
            conf = p_cal[mask].mean()
            ece += (mask.sum() / len(p_cal)) * np.abs(acc - conf)
    return ece

def train_lambdarank_engine(data_path="data/processed/processed_factors.parquet"):
    print(f"Loading factor data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Ensure factor_engineering.py was run.")

    df = pd.read_parquet(data_path)
    df = df.sort_values('date').reset_index(drop=True)

    target_col = 'fwd_ret_21d'
    
    # Strictly exclude non-predictive/target columns
    exclude_cols = {
        'date', 'Ticker', 'symbol', 'close', 'open', 'high', 'low', 'volume', 
        'dividends', 'stock splits', 'fwd_ret_21d', 'target_rel_ret', 
        'target_outperform', 'label', 'relevance'
    }

    candidate_features = [c for c in df.columns if c not in exclude_cols]

    # Z-score standardization across cross-section
    z_feature_cols = []
    for c in candidate_features:
        z_col = f"{c}_z" if not c.endswith('_z') else c
        if z_col not in df.columns:
            df[z_col] = df.groupby('date')[c].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9)
            ).fillna(0.0)
        z_feature_cols.append(z_col)

    z_feature_cols = list(dict.fromkeys(z_feature_cols))

    # Define binary target for propensity calibration (1 if return > daily median return)
    df['binary_outperform'] = (
        df[target_col] > df.groupby('date')[target_col].transform('median')
    ).astype(int)

    # Convert continuous target into 5 quintile relevance grades for LambdaRank
    df['relevance'] = df.groupby('date')[target_col].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)
    )

    splits = purged_time_splits(df['date'].values, n_splits=5, embargo=21)
    
    fold_ics = []
    oof_predictions = []

    for fold, (train_dates, valid_dates) in enumerate(splits, 1):
        tr_df = df[df['date'].isin(train_dates)].copy()
        va_df = df[df['date'].isin(valid_dates)].copy()

        grp_tr = tr_df.groupby('date', sort=False).size().to_numpy()
        grp_va = va_df.groupby('date', sort=False).size().to_numpy()

        X_tr, y_tr = tr_df[z_feature_cols].to_numpy(), tr_df['relevance'].to_numpy()
        X_va, y_va = va_df[z_feature_cols].to_numpy(), va_df['relevance'].to_numpy()

        dtr = lgb.Dataset(X_tr, label=y_tr, group=grp_tr)
        dva = lgb.Dataset(X_va, label=y_va, group=grp_va, reference=dtr)

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'learning_rate': 0.03,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'seed': 42,
            'bagging_seed': 42,
            'feature_fraction_seed': 42,
            'verbose': -1
        }

        model = lgb.train(
            params,
            dtr,
            num_boost_round=500,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        va_df['raw_score'] = model.predict(X_va)
        mean_ic, ic_ir, t_stat = compute_daily_ic(va_df, score_col='raw_score', target_col=target_col)
        fold_ics.append(mean_ic)

        oof_predictions.append(va_df[['date', 'Ticker', 'raw_score', 'binary_outperform', target_col]])
        print(f"Fold {fold} | Mean Rank IC: {mean_ic:.4f} | IC-IR: {ic_ir:.4f}")

    # Combine out-of-fold validation predictions for probability calibration
    oof_df = pd.concat(oof_predictions, ignore_index=True)

    # Fit Isotonic Calibration on OOF raw scores
    calibrator = IsotonicRegression(out_of_bounds='clip')
    
    # Normalize raw scores to 0-1 range prior to Isotonic fitting
    raw_min, raw_max = oof_df['raw_score'].min(), oof_df['raw_score'].max()
    norm_scores = (oof_df['raw_score'] - raw_min) / (raw_max - raw_min + 1e-9)
    
    calibrator.fit(norm_scores, oof_df['binary_outperform'])
    oof_df['calibrated_propensity'] = calibrator.predict(norm_scores)

    ece_val = compute_ece(oof_df['calibrated_propensity'].values, oof_df['binary_outperform'].values)

    # Save models and calibration objects
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, "data/models/lambdarank_model.pkl")
    joblib.dump(calibrator, "data/models/isotonic_calibrator.pkl")
    joblib.dump({'raw_min': raw_min, 'raw_max': raw_max}, "data/models/score_scaler.pkl")

    print("--------------------------------------------------")
    print(f"Overall Cross-Validated Out-of-Sample Rank IC: {np.nanmean(fold_ics):.4f}")
    print(f"Isotonic Calibration Completed. Out-of-Sample ECE: {ece_val:.4f}")
    print(f"Saved artifacts to 'data/models/' directory.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    train_lambdarank_engine()