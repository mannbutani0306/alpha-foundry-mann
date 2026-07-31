import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import time

def fetch_stock_data(tickers, start_date="2020-01-01", end_date=None):
    """
    Fetch daily OHLCV data for given stock tickers using Yahoo Finance.
    Appends '.NS' for National Stock Exchange (NSE) India tickers if needed.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    formatted_tickers = [t if t.endswith('.NS') or t.startswith('^') else f"{t}.NS" for t in tickers]
    
    print(f"Fetching data for {len(formatted_tickers)} tickers from {start_date} to {end_date}...")
    
    records = []
    for ticker in formatted_tickers:
        clean_ticker = ticker.replace('.NS', '')
        success = False
        
        # Retry logic up to 3 attempts
        for attempt in range(3):
            try:
                sub_df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
                if not sub_df.empty:
                    sub_df = sub_df.reset_index()
                    sub_df.columns = [str(c).lower() for c in sub_df.columns]
                    sub_df['Ticker'] = clean_ticker
                    records.append(sub_df)
                    print(f"Successfully downloaded {clean_ticker}")
                    success = True
                    break
            except Exception:
                time.sleep(1)
                
        if not success:
            print(f"Warning: Skipping {clean_ticker} (No data returned)")
            
    if not records:
        raise ValueError("No data fetched for any ticker.")
        
    combined = pd.concat(records, axis=0)
    combined = combined.rename(columns={'date': 'date', 'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'})
    combined['date'] = pd.to_datetime(combined['date']).dt.tz_localize(None)
    combined = combined.sort_values(['date', 'Ticker']).reset_index(drop=True)
    return combined

def compute_quantitative_factors(df):
    """
    Computes cross-sectional and time-series quantitative alpha factors.
    Prevent look-ahead bias by shifting forward targets appropriately.
    """
    df = df.sort_values(['Ticker', 'date']).reset_index(drop=True)
    
    # 1. Momentum Factors
    df['ret_1m'] = df.groupby('Ticker')['close'].pct_change(21)
    df['ret_3m'] = df.groupby('Ticker')['close'].pct_change(63)
    df['ret_12m'] = df.groupby('Ticker')['close'].pct_change(252)
    df['mom_12m_1m'] = df['ret_12m'] - df['ret_1m']  # Standard 12M minus 1M reversal-adjusted momentum
    
    # 2. Volatility & Risk
    df['daily_ret'] = df.groupby('Ticker')['close'].pct_change()
    df['vol_20d'] = df.groupby('Ticker')['daily_ret'].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    
    # 3. Microstructure / Illiquidity (Amihud Ratio)
    df['dollar_volume'] = df['close'] * df['volume']
    df['amihud_illiquidity'] = (df['daily_ret'].abs() / (df['dollar_volume'] + 1e-8)).transform(lambda x: x.rolling(21).mean())
    
    # 4. Target Generation: 21-day Forward Relative Return
    df['fwd_ret_21d'] = df.groupby('Ticker')['close'].shift(-21) / df['close'] - 1.0
    
    # Cross-sectional median subtraction (relative outperformance)
    daily_median_fwd = df.groupby('date')['fwd_ret_21d'].transform('median')
    df['target_rel_ret'] = df['fwd_ret_21d'] - daily_median_fwd
    
    # Binary Class Target (Outperformed median = 1, else 0)
    df['target_outperform'] = (df['target_rel_ret'] > 0).astype(int)
    
    # Clean up NaNs created by rolling windows/shifts
    df_clean = df.dropna(subset=['mom_12m_1m', 'vol_20d', 'target_rel_ret']).copy()
    return df_clean

if __name__ == "__main__":
    sample_universe = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
        'BHARTIARTL', 'ITC', 'SBIN', 'LT', 'AXISBANK',
        'KOTAKBANK', 'HINDUNILVR', 'MARUTI', 'SUNPHARMA', 'WIPRO'
    ]
    
    raw_df = fetch_stock_data(sample_universe, start_date="2020-01-01")
    processed_df = compute_quantitative_factors(raw_df)
    
    output_path = "data/processed/processed_factors.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_parquet(output_path, index=False)
    
    print(f"\nPipeline successfully generated {len(processed_df)} rows of factor data!")
    print(f"Saved dataset to: {output_path}")