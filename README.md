Alpha Foundry: Quantitative Equity Factor Framework
An end-to-end, cross-sectional quantitative equity alpha framework designed for Indian equities (NSE). Alpha Foundry automates point-in-time OHLCV data ingestion, factor engineering (Momentum, Volatility, Illiquidity), daily Z-score normalization, leak-free LightGBM ranking model training, and quantile portfolio backtesting.

Key Performance Results
Annualized Return: 34.01%

Annualized Volatility: 16.70%

Sharpe Ratio (Rf = 6.0%): 1.68

Max Drawdown: -14.19%

Model Validation: 5-Fold Purged Group Time-Series Cross-Validation

Project Architecture
data/processed/ — Parquet feature store

reports/figures/ — Strategy performance charts (backtest_performance.png)

src/features/data_loader.py — OHLCV Ingestion & Target Generation

src/features/factor_engineering.py — Daily Cross-Sectional Z-Score & IC Metrics

src/models/train.py — Purged Group Time-Series CV & LightGBM Model

src/backtest/backtester.py — Top-Quantile Rebalancing Backtester

src/visualization/plots.py — Performance & Drawdown Plotter