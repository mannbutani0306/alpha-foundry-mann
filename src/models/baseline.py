import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def purged_time_splits(dates_sorted, n_splits=5, embargo=21):
    """
    Same split logic as train.py / ensemble.py, kept identical so the
    baseline is evaluated on the exact same out-of-sample windows.
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
    """Calculates Spearman Rank IC per date. Identical logic to train.py/ensemble.py."""
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

    return mean_ic, ic_ir, t_stat, ic_series


def run_baseline_pipeline(data_path="data/processed/processed_factors.parquet"):
    print(f"Loading factor data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Ensure factor_engineering.py was run.")

    df = pd.read_parquet(data_path).sort_values('date').reset_index(drop=True)
    target_col = 'fwd_ret_21d'

    # Same exclusion list as train.py / ensemble.py, so the baseline sees
    # exactly the same candidate feature set as the LightGBM engine.
    exclude_cols = {
        'date', 'Ticker', 'symbol', 'close', 'open', 'high', 'low', 'volume',
        'dividends', 'stock splits', 'fwd_ret_21d', 'target_rel_ret',
        'target_outperform', 'label', 'relevance'
    }
    candidate_features = [c for c in df.columns if c not in exclude_cols]

    # Cross-sectional z-scoring (same transform used elsewhere in the pipeline)
    z_feature_cols = []
    for c in candidate_features:
        z_col = f"{c}_z" if not c.endswith('_z') else c
        if z_col not in df.columns:
            df[z_col] = df.groupby('date')[c].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9)
            ).fillna(0.0)
        z_feature_cols.append(z_col)
    z_feature_cols = list(dict.fromkeys(z_feature_cols))

    print(f"Baseline uses {len(z_feature_cols)} equal-weighted z-scored features: {z_feature_cols}")

    # Equal-weighted composite score: simple mean of all z-scored features.
    # No fitting, no look-ahead, no hyperparameters — this is the "dumb"
    # linear benchmark the ML engine needs to beat.
    df['baseline_score'] = df[z_feature_cols].mean(axis=1)

    # Evaluate on the SAME out-of-sample validation windows as train.py,
    # so the comparison to the LightGBM engine is apples-to-apples rather
    # than evaluating the baseline on different dates.
    splits = purged_time_splits(df['date'].values, n_splits=5, embargo=21)

    fold_ics = []
    oof_frames = []
    for fold, (train_dates, valid_dates) in enumerate(splits, 1):
        va_df = df[df['date'].isin(valid_dates)].copy()
        mean_ic, ic_ir, t_stat, _ = compute_daily_ic(va_df, score_col='baseline_score', target_col=target_col)
        fold_ics.append(mean_ic)
        oof_frames.append(va_df[['date', 'baseline_score', target_col]])
        print(f"Fold {fold} | Baseline Mean Rank IC: {mean_ic:.4f}")

    # Overall out-of-sample IC/IC-IR/t-stat pooled across all fold validation windows
    oof_df = pd.concat(oof_frames, ignore_index=True)
    overall_ic, overall_ir, overall_t, _ = compute_daily_ic(oof_df, score_col='baseline_score', target_col=target_col)

    print("--------------------------------------------------")
    print("Baseline Equal-Weighted Linear Factor Score (Out-of-Sample):")
    print(f"  Overall Rank IC: {overall_ic:.4f}")
    print(f"  IC-IR: {overall_ir:.4f}")
    print(f"  t-statistic: {overall_t:.4f}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    run_baseline_pipeline()
