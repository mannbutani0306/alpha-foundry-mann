import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import optuna
from optuna.samplers import TPESampler
import joblib

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def purged_time_splits(dates_sorted, n_splits=5, embargo=21):
    """Splits dates sequentially with embargo to eliminate look-ahead bias."""
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
    """Calculates Spearman Rank IC per date."""
    def _ic(g):
        g = g.dropna(subset=[score_col, target_col])
        if len(g) < 5:
            return np.nan
        return spearmanr(g[score_col], g[target_col]).correlation

    ic_series = df.groupby('date').apply(_ic).dropna()
    mean_ic = ic_series.mean()
    ic_std = ic_series.std(ddof=1)
    ic_ir = mean_ic / (ic_std + 1e-12)
    return mean_ic, ic_ir

def optimize_lambdarank(df, z_feature_cols, target_col='fwd_ret_21d', n_trials=15):
    """Runs Optuna HPO maximizing out-of-sample Rank IC under Purged CV."""
    print(f"Starting Optuna HPO ({n_trials} trials)...")
    splits = purged_time_splits(df['date'].values, n_splits=5, embargo=21)

    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': 1,
            'verbose': -1,
            'seed': 42,
            'bagging_seed': 42,
            'feature_fraction_seed': 42
        }
        
        fold_ics = []
        for train_dates, valid_dates in splits:
            tr_df = df[df['date'].isin(train_dates)]
            va_df = df[df['date'].isin(valid_dates)]

            grp_tr = tr_df.groupby('date', sort=False).size().to_numpy()
            grp_va = va_df.groupby('date', sort=False).size().to_numpy()

            dtr = lgb.Dataset(tr_df[z_feature_cols].to_numpy(), label=tr_df['relevance'].to_numpy(), group=grp_tr)
            dva = lgb.Dataset(va_df[z_feature_cols].to_numpy(), label=va_df['relevance'].to_numpy(), group=grp_va, reference=dtr)

            model = lgb.train(params, dtr, num_boost_round=200, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(30, verbose=False)])

            scores = model.predict(va_df[z_feature_cols].to_numpy())
            tmp_df = va_df[['date', target_col]].copy()
            tmp_df['score'] = scores
            mean_ic, _ = compute_daily_ic(tmp_df, score_col='score', target_col=target_col)
            fold_ics.append(mean_ic)

        return float(np.nanmean(fold_ics))

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"Optuna Best Trial OOS Rank IC: {study.best_value:.4f}")
    print(f"Optuna Best Params: {study.best_params}")
    return study.best_params

def run_ensemble_stacking_pipeline(data_path="data/processed/processed_factors.parquet"):
    print(f"Loading factor data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    df = pd.read_parquet(data_path).sort_values('date').reset_index(drop=True)
    target_col = 'fwd_ret_21d'

    # Feature setup
    exclude_cols = {'date', 'Ticker', 'symbol', 'close', 'open', 'high', 'low', 'volume', 
                    'dividends', 'stock splits', 'fwd_ret_21d', 'target_rel_ret', 
                    'target_outperform', 'label', 'relevance'}
    candidate_features = [c for c in df.columns if c not in exclude_cols]

    z_feature_cols = []
    for c in candidate_features:
        z_col = f"{c}_z" if not c.endswith('_z') else c
        if z_col not in df.columns:
            df[z_col] = df.groupby('date')[c].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9)
            ).fillna(0.0)
        z_feature_cols.append(z_col)
    z_feature_cols = list(dict.fromkeys(z_feature_cols))

    df['relevance'] = df.groupby('date')[target_col].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)
    )

    # 1. Run Optuna HPO to get best LightGBM parameters
    best_lgb_params = optimize_lambdarank(df, z_feature_cols, target_col=target_col, n_trials=10)
    best_lgb_params.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbose': -1,
        'seed': 42,
        'bagging_seed': 42,
        'feature_fraction_seed': 42
    })

    splits = purged_time_splits(df['date'].values, n_splits=5, embargo=21)

    oof_predictions = []

    # 2. Out-of-fold predictions across 3 diverse base learners
    for fold, (train_dates, valid_dates) in enumerate(splits, 1):
        tr_df = df[df['date'].isin(train_dates)].copy()
        va_df = df[df['date'].isin(valid_dates)].copy()

        X_tr, y_tr = tr_df[z_feature_cols].to_numpy(), tr_df['relevance'].to_numpy()
        X_va, y_va = va_df[z_feature_cols].to_numpy(), va_df['relevance'].to_numpy()

        # Learner 1: Optimized LightGBM LambdaRank
        grp_tr = tr_df.groupby('date', sort=False).size().to_numpy()
        grp_va = va_df.groupby('date', sort=False).size().to_numpy()
        dtr = lgb.Dataset(X_tr, label=y_tr, group=grp_tr)
        dva = lgb.Dataset(X_va, label=y_va, group=grp_va, reference=dtr)
        m_lgb = lgb.train(best_lgb_params, dtr, num_boost_round=300, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        lgb_scores = m_lgb.predict(X_va)

        # Learner 2: Random Forest Regressor
        m_rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        m_rf.fit(X_tr, tr_df[target_col].to_numpy())
        rf_scores = m_rf.predict(X_va)

        # Learner 3: Ridge Linear Factor Model
        m_ridge = Ridge(alpha=100.0)
        m_ridge.fit(X_tr, tr_df[target_col].to_numpy())
        ridge_scores = m_ridge.predict(X_va)

        va_df['score_lgb'] = lgb_scores
        va_df['score_rf'] = rf_scores
        va_df['score_ridge'] = ridge_scores

        oof_predictions.append(va_df)

    oof_df = pd.concat(oof_predictions, ignore_index=True)

    # 3. Compute Rank-Averaged Ensemble
    oof_df['rank_lgb'] = oof_df.groupby('date')['score_lgb'].rank(pct=True)
    oof_df['rank_rf'] = oof_df.groupby('date')['score_rf'].rank(pct=True)
    oof_df['rank_ridge'] = oof_df.groupby('date')['score_ridge'].rank(pct=True)

    oof_df['ensemble_rank_avg'] = (
        0.50 * oof_df['rank_lgb'] + 0.30 * oof_df['rank_rf'] + 0.20 * oof_df['rank_ridge']
    )

    # Evaluate individual base models vs Ensemble
    ic_lgb, ir_lgb = compute_daily_ic(oof_df, score_col='score_lgb', target_col=target_col)
    ic_rf, ir_rf = compute_daily_ic(oof_df, score_col='score_rf', target_col=target_col)
    ic_ridge, ir_ridge = compute_daily_ic(oof_df, score_col='score_ridge', target_col=target_col)
    ic_ens, ir_ens = compute_daily_ic(oof_df, score_col='ensemble_rank_avg', target_col=target_col)

    print("--------------------------------------------------")
    print("Ensemble & Stacking Model Comparison (Out-of-Sample):")
    print(f"  Base Learner 1 (LightGBM LambdaRank) | Mean Rank IC: {ic_lgb:.4f} | IC-IR: {ir_lgb:.4f}")
    print(f"  Base Learner 2 (Random Forest)        | Mean Rank IC: {ic_rf:.4f} | IC-IR: {ir_rf:.4f}")
    print(f"  Base Learner 3 (Ridge Linear Factor)  | Mean Rank IC: {ic_ridge:.4f} | IC-IR: {ir_ridge:.4f}")
    print(f"  --> Rank-Averaged Ensemble           | Mean Rank IC: {ic_ens:.4f} | IC-IR: {ir_ens:.4f}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_ensemble_stacking_pipeline()