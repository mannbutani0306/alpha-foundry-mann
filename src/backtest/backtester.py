import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_quantile_backtest(df, pred_col='pred_prob', return_col='fwd_ret_21d', top_q=0.2):
    """
    Simulates a top-quantile long-only factor portfolio rebalanced every 21 trading days.
    Computes Sharpe Ratio, Max Drawdown, and Cumulative Returns.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # Select rebalancing dates (every 21 trading days to avoid overlapping holding periods)
    unique_dates = sorted(df['date'].unique())
    rebalance_dates = unique_dates[::21]
    
    portfolio_returns = []
    
    for dt in rebalance_dates:
        day_data = df[df['date'] == dt].copy()
        if len(day_data) == 0:
            continue
            
        # Select top-quantile stocks based on model predictions
        cutoff = day_data[pred_col].quantile(1 - top_q)
        long_basket = day_data[day_data[pred_col] >= cutoff]
        
        # Equal-weighted portfolio return over next 21 trading days
        period_ret = long_basket[return_col].mean()
        portfolio_returns.append({'date': dt, 'portfolio_ret': period_ret})
        
    res_df = pd.DataFrame(portfolio_returns)
    res_df['cum_return'] = (1 + res_df['portfolio_ret']).cumprod() - 1.0
    
    # Calculate Risk & Return Metrics
    ann_ret = res_df['portfolio_ret'].mean() * (252 / 21)
    ann_vol = res_df['portfolio_ret'].std() * np.sqrt(252 / 21)
    sharpe_ratio = (ann_ret - 0.06) / (ann_vol + 1e-8)  # Assuming 6% risk-free rate
    
    # Maximum Drawdown calculation
    cum_wealth = (1 + res_df['portfolio_ret']).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    max_drawdown = drawdown.min()
    
    metrics = {
        'Annualized Return': f"{ann_ret * 100:.2f}%",
        'Annualized Volatility': f"{ann_vol * 100:.2f}%",
        'Sharpe Ratio (Rf=6%)': round(sharpe_ratio, 2),
        'Max Drawdown': f"{max_drawdown * 100:.2f}%"
    }
    
    return res_df, metrics

if __name__ == "__main__":
    # Import training script logic to get out-of-sample predictions
    from src.models.train import train_alpha_model
    
    input_path = "data/processed/normalized_factors.parquet"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run factor_engineering.py first!")
        
    df = pd.read_parquet(input_path)
    feature_cols = [c for c in df.columns if c.endswith('_zscore')]
    
    # Train model and generate stock scores
    model, _ = train_alpha_model(df, feature_cols)
    df['pred_prob'] = model.predict_proba(df[feature_cols])[:, 1]
    
    # Execute Backtest
    backtest_res, metrics = run_quantile_backtest(df)
    
    print("\n=== Quantile Backtest Performance Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")