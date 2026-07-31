import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_backtest_performance(res_df, output_path="reports/figures/backtest_performance.png"):
    """
    Plots Cumulative Strategy Returns and Underwater Drawdown Curve.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate Drawdown series
    cum_wealth = (1 + res_df['portfolio_ret']).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Equity Curve
    ax1.plot(res_df['date'], (cum_wealth - 1) * 100, label='Top-Quantile Alpha Strategy', color='#1f77b4', linewidth=2)
    ax1.set_title('Alpha Foundry: Top-Quantile Factor Strategy Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=10)
    
    # 2. Drawdown Plot
    ax2.fill_between(res_df['date'], drawdown * 100, 0, color='#d62728', alpha=0.4, label='Drawdown')
    ax2.plot(res_df['date'], drawdown * 100, color='#d62728', linewidth=1)
    ax2.set_title('Portfolio Drawdown (Underwater Chart)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Performance chart successfully saved to: {output_path}")

if __name__ == "__main__":
    from src.models.train import train_alpha_model
    from src.backtest.backtester import run_quantile_backtest
    
    input_path = "data/processed/normalized_factors.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run factor_engineering.py first!")
        
    df = pd.read_parquet(input_path)
    feature_cols = [c for c in df.columns if c.endswith('_zscore')]
    
    model, _ = train_alpha_model(df, feature_cols)
    df['pred_prob'] = model.predict_proba(df[feature_cols])[:, 1]
    
    res_df, metrics = run_quantile_backtest(df)
    plot_backtest_performance(res_df)