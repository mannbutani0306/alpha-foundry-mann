import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from src.models.ensemble import purged_time_splits, compute_daily_ic, optimize_lambdarank

def run_seed_stability_check(data_path="data/processed/processed_factors.parquet", seeds=(42, 7, 123)):
    print(f"Loading factor data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    df = pd.read_parquet(data_path).sort_values('date').reset_index(drop=True)
    target_col = 'fwd_ret_21d'

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

    # Run HPO once (seeded) to get stable hyperparameters
    best_lgb_params = optimize_lambdarank(df, z_feature_cols, target_col=target_col, n_trials=10)
    base_params = {'objective': 'lambdarank', 'metric': 'ndcg', 'verbose': -1}
    base_params.update(best_lgb_params)

    splits = purged_time_splits(df['date'].values, n_splits=5, embargo=21)

    print("--------------------------------------------------")
    print("Seed Stability Check: LightGBM LambdaRank Base Learner")
    print("--------------------------------------------------")

    seed_ics = []
    for seed in seeds:
        params = dict(base_params)
        params.update({'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed})

        oof_predictions = []
        for train_dates, valid_dates in splits:
            tr_df = df[df['date'].isin(train_dates)].copy()
            va_df = df[df['date'].isin(valid_dates)].copy()

            X_tr, y_tr = tr_df[z_feature_cols].to_numpy(), tr_df['relevance'].to_numpy()
            X_va = va_df[z_feature_cols].to_numpy()

            grp_tr = tr_df.groupby('date', sort=False).size().to_numpy()
            grp_va = va_df.groupby('date', sort=False).size().to_numpy()

            dtr = lgb.Dataset(X_tr, label=y_tr, group=grp_tr)
            dva = lgb.Dataset(X_va, label=va_df['relevance'].to_numpy(), group=grp_va, reference=dtr)

            model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(30, verbose=False)])

            va_df['score'] = model.predict(X_va)
            oof_predictions.append(va_df)

        oof_df = pd.concat(oof_predictions, ignore_index=True)
        mean_ic, ic_ir = compute_daily_ic(oof_df, score_col='score', target_col=target_col)
        seed_ics.append(mean_ic)
        print(f"  Seed {seed:>4} | Mean Rank IC: {mean_ic:.4f} | IC-IR: {ic_ir:.4f}")

    seed_ics = np.array(seed_ics)
    print("--------------------------------------------------")
    print(f"Across {len(seeds)} seeds -> Mean IC: {seed_ics.mean():.4f} | Std IC: {seed_ics.std(ddof=1):.4f} | Min: {seed_ics.min():.4f} | Max: {seed_ics.max():.4f}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_seed_stability_check()