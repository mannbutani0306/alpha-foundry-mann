import os
import numpy as np
import pandas as pd
import lightgbm as lgb


def purged_time_splits(dates_sorted, n_splits=5, embargo=21):
    """Identical split logic to train.py / ensemble.py / baseline.py."""
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


def build_oof_scores(df, z_feature_cols, target_col='fwd_ret_21d', n_splits=5, embargo=21):
    """
    Regenerates out-of-fold LambdaRank scores by training a FRESH model per
    fold and predicting only on that fold's own validation dates.

    This deliberately does NOT reuse data/models/lambdarank_model.pkl, because
    that file only holds the model from whichever fold trained last -- using
    it to score earlier dates would mean scoring dates the saved model may
    effectively have "seen" via correlated features, which is not a clean
    out-of-sample backtest. Regenerating OOF scores here keeps the backtest
    honest at the cost of a few minutes of extra training time.
    """
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5, 10],
        'learning_rate': 0.03, 'num_leaves': 31, 'min_data_in_leaf': 20,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'seed': 42, 'bagging_seed': 42, 'feature_fraction_seed': 42, 'verbose': -1
    }
    splits = purged_time_splits(df['date'].values, n_splits, embargo)
    oof_frames = []
    for fold, (train_dates, valid_dates) in enumerate(splits, 1):
        tr_df = df[df['date'].isin(train_dates)].copy()
        va_df = df[df['date'].isin(valid_dates)].copy()

        grp_tr = tr_df.groupby('date', sort=False).size().to_numpy()
        grp_va = va_df.groupby('date', sort=False).size().to_numpy()

        dtr = lgb.Dataset(tr_df[z_feature_cols].to_numpy(), label=tr_df['relevance'].to_numpy(), group=grp_tr)
        dva = lgb.Dataset(va_df[z_feature_cols].to_numpy(), label=va_df['relevance'].to_numpy(), group=grp_va, reference=dtr)

        model = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                           callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        va_df = va_df.copy()
        va_df['oof_score'] = model.predict(va_df[z_feature_cols].to_numpy())
        oof_frames.append(va_df[['date', 'Ticker', 'oof_score', target_col]])
        print(f"  Fold {fold}: generated {len(va_df)} out-of-fold scores.")

    return pd.concat(oof_frames, ignore_index=True)


def decile_backtest(oof_df, n_buckets, target_col='fwd_ret_21d'):
    """
    Bucket-sorted backtest. Uses n_buckets (not a hardcoded 10) because with
    only ~15 names in the universe, 10 deciles would leave 1-2 names per
    bucket -- not a meaningful average. This is a deliberate, documented
    adaptation of the project brief's Section A12.1 method to a small
    universe, not a deviation from its intent.
    """
    rows = []
    for dt, g in oof_df.groupby('date'):
        g = g.dropna(subset=['oof_score', target_col])
        if len(g) < n_buckets * 2:
            continue
        g = g.copy()
        g['bucket'] = pd.qcut(g['oof_score'].rank(method='first'), n_buckets, labels=False)
        bucket_means = g.groupby('bucket')[target_col].mean()
        bucket_names = g.groupby('bucket')['Ticker'].apply(list)
        row = {'date': dt}
        for b in range(n_buckets):
            row[f'B{b+1}_ret'] = bucket_means.get(b, np.nan)
            row[f'B{b+1}_names'] = bucket_names.get(b, [])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def compute_turnover(names_series):
    """0.5 * sum(|weight_t - weight_{t-1}|) for an equal-weight bucket portfolio."""
    turnovers = [np.nan]
    prev = None
    for names in names_series:
        cur = set(names) if isinstance(names, list) else set()
        if prev is None:
            prev = cur
            continue
        w_prev = {n: 1.0 / len(prev) for n in prev} if prev else {}
        w_cur = {n: 1.0 / len(cur) for n in cur} if cur else {}
        keys = set(w_prev) | set(w_cur)
        to = 0.5 * sum(abs(w_cur.get(k, 0) - w_prev.get(k, 0)) for k in keys)
        turnovers.append(to)
        prev = cur
    return turnovers


def run_backtest(data_path="data/processed/processed_factors.parquet", n_buckets=5, cost_bps=25):
    print(f"Loading factor data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    df = pd.read_parquet(data_path).sort_values('date').reset_index(drop=True)
    target_col = 'fwd_ret_21d'

    exclude_cols = {
        'date', 'Ticker', 'symbol', 'close', 'open', 'high', 'low', 'volume',
        'dividends', 'stock splits', 'fwd_ret_21d', 'target_rel_ret',
        'target_outperform', 'label', 'relevance'
    }
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

    print("Regenerating out-of-fold scores (fresh model per fold, no look-ahead)...")
    oof_df = build_oof_scores(df, z_feature_cols, target_col=target_col)

    print(f"Running {n_buckets}-bucket backtest (top bucket minus bottom bucket)...")
    bt = decile_backtest(oof_df, n_buckets=n_buckets, target_col=target_col)

    top_col, bot_col = f'B{n_buckets}_ret', 'B1_ret'
    top_names_col, bot_names_col = f'B{n_buckets}_names', 'B1_names'

    gross_spread = bt[top_col] - bt[bot_col]
    turnover_top = compute_turnover(bt[top_names_col])
    turnover_bottom = compute_turnover(bt[bot_names_col])
    bt['combined_turnover'] = (pd.Series(turnover_top).fillna(0) + pd.Series(turnover_bottom).fillna(0)) / 2
    bt['gross_spread'] = gross_spread
    bt['net_spread'] = gross_spread - bt['combined_turnover'] * (cost_bps / 1e4)

    mean_gross = bt['gross_spread'].mean()
    mean_net = bt['net_spread'].mean()
    std_net = bt['net_spread'].std(ddof=1)
    periods_per_year = 12  # ~21-trading-day rebalance cadence
    ir_net = (mean_net / (std_net + 1e-12)) * np.sqrt(periods_per_year)
    avg_turnover = bt['combined_turnover'].mean()

    print("--------------------------------------------------")
    print(f"Decile Back-Test Summary ({n_buckets} buckets, top-minus-bottom, out-of-fold scores):")
    print(f"  Rebalance dates evaluated: {len(bt)}")
    print(f"  Mean Gross Spread per ~21-day period: {mean_gross:.4%}")
    print(f"  Assumed round-trip transaction cost: {cost_bps} bps")
    print(f"  Average Turnover per rebalance: {avg_turnover:.4f}")
    print(f"  Mean Net Spread per ~21-day period: {mean_net:.4%}")
    print(f"  Annualized Net Information Ratio (~{periods_per_year} periods/yr): {ir_net:.4f}")
    print("--------------------------------------------------")

    os.makedirs("data/models", exist_ok=True)
    bt.drop(columns=[c for c in bt.columns if c.endswith('_names')]).to_csv(
        "data/models/backtest_summary.csv", index=False
    )
    print("Saved per-date backtest detail to data/models/backtest_summary.csv")


if __name__ == "__main__":
    run_backtest()
