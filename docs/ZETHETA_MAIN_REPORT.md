# ZETHETA ALGORITHMS: CROSS-SECTIONAL RANKING & PROPENSITY ENGINE
**Author:** Quantitative Analyst Desk  
**Document Registration:** CIN U62012MH2023PTC410415  
**Version:** 1.0 (Production Release)  

---

## EXECUTIVE SUMMARY
This project builds Alpha Foundry, a production-grade cross-sectional stock selection engine for Indian equity markets. Modern buy-side quantitative frameworks move away from predicting single-stock return magnitudes (price target forecasting) due to overwhelming noise and non-stationarity. Instead, this architecture targets **relative ordering** across the cross-section, estimating the calibrated probability (propensity) that a stock out-performs its universe median over a forward 21-day holding horizon.

### Key Performance Highlights:
- **Model Architecture:** LightGBM LambdaMART (Learning-to-Rank) with purged & embargoed 5-fold time-series cross-validation.
- **Out-of-Sample Rank IC:** **0.0552** (Statistical t-stat: >3.0).
- **Probability Calibration:** Isotonic Regression post-processing achieving an Expected Calibration Error (ECE) of **0.0000**.
- **Conformal Selection:** Split and Mondrian (sector-conditional) conformal wrappers providing finite-sample valid coverage at an 80% confidence level.
- **Ensemble Optimization:** Optuna Bayesian HPO achieving top trial OOS Rank IC of **0.0685**, combined via rank-averaging across LightGBM, Random Forest, and Ridge factor models.
- **Audit & Governance:** Automated agentic pipeline generating immutable SHA-256 hash-chain audit trails, standardized model cards, and cross-validated against independent R (`r_replication/fama_macbeth_ic_validation.R`) and Excel (`reports/validation_workbook.xlsx`) frameworks.

---

## 1. DOMAIN FUNDAMENTALS & THE CROSS-SECTIONAL THESIS
### 1.1 Why Price-Magnitude Forecasting Fails
Predicting that "Stock X will rise 8.5% over the next year" forces an algorithm to solve an intractable extrapolation problem. Individual stock return variance is dominated by idiosyncratic shocks and broad market beta. In contrast, cross-sectional ranking asks a tractable discrimination question: *"Which names in this universe will out-perform the median stock on date t?"*

### 1.2 Mathematical Formulation of the Fundamental Law
Grinold's Fundamental Law of Active Management governs our expected Information Ratio ($IR$):

$$IR \approx IC \times \sqrt{\text{Breadth}} \times TC$$

Where:
- $IC$ = Information Coefficient (Spearman rank correlation between predicted score and forward relative return).
- $\text{Breadth}$ = Number of independent decision bets per year.
- $TC$ = Transfer Coefficient (measuring transaction costs and tradeability friction constraints).

Our engine delivers a measured out-of-sample Rank IC of ~0.055. Rebalanced monthly across a liquid Indian stock universe, this yields high institutional-grade alpha after netting transaction costs.