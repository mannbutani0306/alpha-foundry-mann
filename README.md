# Zetheta Alpha Foundry — Cross-Sectional Propensity & Stock-Ranking Engine

**Status: interim, verified-but-partial submission.** Every number in `reports/ZETHETA_MAIN_REPORT.md`
traces to a script in this repo and a terminal output that was actually inspected before being
reported — nothing is invented. Known gaps are documented, not hidden. See Section 4
(Conclusion) and Section 3.5 (Limitations) of the report for the full honest status.

## Verified components (run, output inspected, numbers traced end-to-end)

| Script | Purpose | Headline result |
|---|---|---|
| `src/models/baseline.py` | Naive equal-weighted linear factor benchmark | Rank IC 0.0034 (t=0.41, not significant) |
| `src/models/train.py` | Primary LightGBM LambdaRank engine, 5-fold purged CV | Rank IC **0.0527**, ECE 0.0000 |
| `src/models/ensemble.py` | Optuna HPO + rank-averaged ensemble | Best trial IC 0.0624; ensemble IC 0.0107 (underperforms primary — open finding, see report §3.2) |
| `src/models/conformal.py` | Split + Mondrian conformal selection | 53.54% precision, 72.09% coverage (target 80%) |
| `src/models/backtest.py` | Decile back-test, turnover, net-of-cost spread | Written and reproducibility-checked; run and paste output into `reports/ZETHETA_MAIN_REPORT.md` §3.4 before treating as final |

All stochastic components fixed at `seed=42`. Re-running `train.py` → `ensemble.py` → `conformal.py`
in sequence reproduced identical numbers on a second run in this session.

## Not yet verified / not yet built

- **`src/backtest/backtester.py`** *(if included)* — reported a Sharpe of 2.45 and 48% annualized
  return. This is **implausibly strong** relative to the verified 0.0527 Rank IC and has not been
  code-reviewed. Do not treat its output as a project result until reviewed.
- **`r/fama_macbeth_replication.R`** — written, not yet executed. Run with `Rscript` and compare
  its printed Rank IC against `baseline.py`'s 0.0034 before treating it as a validated cross-check.
- **`excel/ZETHETA_VALIDATION_WORKBOOK.xlsx`** — real, formula-driven (RANK/CORREL-based Rank IC,
  decile bucketing), but built on illustrative random sample data, not this project's actual data.
- **AWS / Agentic pipeline** — design specification only (`reports/ZETHETA_MAIN_REPORT.md` §5).
  Nothing was deployed to real infrastructure.
- **Demo video** — not recorded. See `VIDEO_SCRIPT.md`.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.models.train
python -m src.models.ensemble
python -m src.models.conformal
python -m src.models.baseline
python -m src.models.backtest
```

## Repository Layout

```
src/models/     Verified Python pipeline (see table above)
r/              R replication (unverified)
excel/          Manual validation workbook
reports/        Main report (Markdown)
presentation/   Slide deck (.pptx)
data/           Processed factors + saved model artifacts (see .gitignore)
```