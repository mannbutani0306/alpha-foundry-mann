import pandas as pd
import numpy as np

def cross_sectional_zscore(df, factor_cols):
    """
    Standardizes factor values on each date across the stock universe.
    Z-score = (Factor - Daily_Mean) / Daily_Std
    Includes Winsorization to cap extreme outliers at +/- 3 standard deviations.
    """
    df_norm = df.copy()
    
    for col in factor_cols:
        # Group by daily date and standardize cross-sectionally
        mean = df_norm.groupby('date')[col].transform('mean')
        std = df_norm.groupby('date')[col].transform('std')
        
        z_col = f"{col}_zscore"
        df_norm[z_col] = (df_norm[col] - mean) / (std + 1e-8)
        
        # Winsorize extreme outliers at [-3.0, 3.0]
        df_norm[z_col] = df_norm[z_col].clip(lower=-3.0, upper=3.0)
        
    return df_norm

def compute_information_coefficient(df, factor_cols, target_col='target_rel_ret'):
    """
    Calculates Spearman Rank Information Coefficient (IC) for each factor per date.
    Returns the mean IC and Information Ratio (IR = Mean IC / Std IC).
    """
    ic_results = {}
    
    for col in factor_cols:
        z_col = f"{col}_zscore" if f"{col}_zscore" in df.columns else col
        
        # Compute daily Spearman rank correlation between factor and target return
        daily_ic = df.groupby('date').apply(
            lambda g: g[z_col].corr(g[target_col], method='spearman')
        )
        
        mean_ic = daily_ic.mean()
        std_ic = daily_ic.std()
        ir = mean_ic / (std_ic + 1e-8)
        
        ic_results[col] = {
            'Mean IC': round(mean_ic, 4),
            'Std IC': round(std_ic, 4),
            'Information Ratio (IR)': round(ir, 4)
        }
        
    return pd.DataFrame(ic_results).T

if __name__ == "__main__":
    # Load processed factor dataset
    input_path = "data/processed/processed_factors.parquet"
    if not pd.io.common.file_exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}. Run data_loader.py first!")
        
    df = pd.read_parquet(input_path)
    
    # Identify alpha factors
    factors = ['mom_12m_1m', 'ret_3m', 'vol_20d', 'amihud_illiquidity']
    
    # Run cross-sectional z-score normalization
    normalized_df = cross_sectional_zscore(df, factors)
    
    # Calculate Information Coefficient
    ic_summary = compute_information_coefficient(normalized_df, factors)
    
    print("\n=== Factor Predictive Power (Information Coefficient Analysis) ===")
    print(ic_summary)
    
    # Save normalized dataset
    output_path = "data/processed/normalized_factors.parquet"
    normalized_df.to_parquet(output_path, index=False)
    print(f"\nNormalized dataset saved to: {output_path}")