import os
import numpy as np
import pandas as pd
import joblib

def split_conformal_selection(p_cal, y_cal, p_test, alpha=0.20):
    """
    Split-conformal selection for outperformance set.
    Computes non-conformity scores (1 - P(true_class)) on calibration set,
    and returns a boolean mask indicating admission into the conformal set.
    """
    # Non-conformity score: distance from true outcome
    s_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    n = len(s_cal)
    
    # Finite-sample adjusted quantile threshold
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_hat = np.quantile(s_cal, q_level, method='higher')
    
    # Admit stock if P(outperform) >= 1 - q_hat
    admit_mask = p_test >= (1.0 - q_hat)
    return admit_mask, float(q_hat)

def mondrian_conformal_selection(p_cal, y_cal, grp_cal, p_test, grp_test, alpha=0.20):
    """
    Mondrian (Group/Sector-conditional) conformal selection.
    Computes a separate non-conformity threshold per group/sector to guarantee 
    coverage balance across sectors.
    """
    admit_mask = np.zeros(len(p_test), dtype=bool)
    sector_thresholds = {}
    
    unique_groups = np.unique(grp_test)
    for g in unique_groups:
        c_cal = (grp_cal == g)
        
        # Fallback to global split if sector sample size is too small (<20)
        if c_cal.sum() < 20:
            c_cal = np.ones_like(grp_cal, dtype=bool)
            
        s_g = np.where(y_cal[c_cal] == 1, 1.0 - p_cal[c_cal], p_cal[c_cal])
        n_g = len(s_g)
        q_level_g = np.ceil((n_g + 1) * (1.0 - alpha)) / n_g
        q_hat_g = np.quantile(s_g, q_level_g, method='higher')
        
        sector_thresholds[g] = float(q_hat_g)
        
        sel_test = (grp_test == g)
        admit_mask[sel_test] = p_test[sel_test] >= (1.0 - q_hat_g)
        
    return admit_mask, sector_thresholds

def run_conformal_pipeline(data_path="data/processed/processed_factors.parquet"):
    print(f"Loading data for Conformal Prediction from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    df = pd.read_parquet(data_path).sort_values('date').reset_index(drop=True)

    # Load trained LightGBM model, Isotonic calibrator, and scaler
    model_path = "data/models/lambdarank_model.pkl"
    calib_path = "data/models/isotonic_calibrator.pkl"
    scaler_path = "data/models/score_scaler.pkl"

    if not (os.path.exists(model_path) and os.path.exists(calib_path)):
        raise FileNotFoundError("Trained model or calibrator missing in data/models/. Run train.py first.")

    model = joblib.load(model_path)
    calibrator = joblib.load(calib_path)
    scaler = joblib.load(scaler_path)

    target_col = 'fwd_ret_21d'
    
    # Feature setup matching train.py
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

    # Binary label for conformal outperformance
    df['binary_outperform'] = (
        df[target_col] > df.groupby('date')[target_col].transform('median')
    ).astype(int)

    # Use simulated sector column if sector metadata is not present in parquet
    if 'sector' not in df.columns:
        # Assign pseudo-sector based on ticker hash for demonstration/structure
        df['sector'] = df['Ticker'].apply(lambda x: f"Sector_{hash(x) % 4 + 1}")

    # Generate model scores & calibrated propensities across the dataset
    X_all = df[z_feature_cols].to_numpy()
    raw_scores = model.predict(X_all)
    norm_scores = (raw_scores - scaler['raw_min']) / (scaler['raw_max'] - scaler['raw_min'] + 1e-9)
    df['calibrated_propensity'] = calibrator.predict(norm_scores)

    # Split dataset into calibration period (first 70% of dates) and test period (last 30%)
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * 0.70)
    
    cal_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    cal_df = df[df['date'].isin(cal_dates)]
    test_df = df[df['date'].isin(test_dates)].copy()

    # 1. Run Split Conformal (Global)
    alpha = 0.20  # Target significance level (80% confidence coverage)
    admit_split, q_hat_global = split_conformal_selection(
        cal_df['calibrated_propensity'].values,
        cal_df['binary_outperform'].values,
        test_df['calibrated_propensity'].values,
        alpha=alpha
    )
    test_df['conformal_admit_split'] = admit_split

    # 2. Run Mondrian Conformal (Sector-conditional)
    admit_mondrian, sector_q_hats = mondrian_conformal_selection(
        cal_df['calibrated_propensity'].values,
        cal_df['binary_outperform'].values,
        cal_df['sector'].values,
        test_df['calibrated_propensity'].values,
        test_df['sector'].values,
        alpha=alpha
    )
    test_df['conformal_admit_mondrian'] = admit_mondrian

    # --- PRECISION: of the stocks ADMITTED into the shortlist, what fraction actually outperformed? ---
    # (This is what the original code computed and mislabeled with "coverage" variable names.)
    split_precision = (test_df[test_df['conformal_admit_split']]['binary_outperform']).mean()
    mondrian_precision = (test_df[test_df['conformal_admit_mondrian']]['binary_outperform']).mean()

    # --- COVERAGE: of the stocks that TRULY outperformed, what fraction did we admit? ---
    # This is the quantity conformal prediction's (1 - alpha) guarantee actually refers to.
    true_outperformers = test_df[test_df['binary_outperform'] == 1]
    split_coverage_rate = true_outperformers['conformal_admit_split'].mean()
    mondrian_coverage_rate = true_outperformers['conformal_admit_mondrian'].mean()

    print("--------------------------------------------------")
    print("Conformal Prediction Pipeline Completed.")
    print(f"Target Coverage Level (1 - alpha): {(1 - alpha)*100:.1f}%")
    print(f"Global Split-Conformal q_hat Threshold: {q_hat_global:.4f}")
    print(f"Empirical Outperformance Precision (Split Set): {split_precision*100:.2f}%")
    print(f"Empirical Outperformance Precision (Mondrian Set): {mondrian_precision*100:.2f}%")
    print(f"Empirical Coverage (Split Set): {split_coverage_rate*100:.2f}%")
    print(f"Empirical Coverage (Mondrian Set): {mondrian_coverage_rate*100:.2f}%")
    print(f"Mondrian Sector Thresholds: {sector_q_hats}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_conformal_pipeline()