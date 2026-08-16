# ZETHETA ALGORITHMS: CROSS-SECTIONAL RANKING & PROPENSITY ENGINE
**Document Control:** Production Release & Audit Master  
**Corporate Registration:** CIN U62012MH2023PTC410415  
**Version:** 1.0.0  
**Target Universe:** Indian Equities (NSE Liquid Universe)  

---

## 1. EXECUTIVE SUMMARY

### 1.1 Core Quantitative Rationale
Traditional quantitative equity strategies rely heavily on absolute return forecasting—attempting to predict whether a specific stock will yield an exact price target (e.g., $+8.5\%$) over a given holding period. In modern high-dimensional equity markets, this approach suffers from significant structural limitations:
1. **Pervasive Market Beta Noise**: Single-stock price trajectories are overwhelmingly dominated by macroeconomic drift, broader market sentiment, and sector-wide risk factors rather than stock-specific alpha.
2. **Non-Stationarity**: Absolute price return distributions exhibit time-varying conditional variance, fat tails (leptokurtosis), and non-stationary drift parameters that destabilize standard regression loss functions (such as Mean Squared Error).
3. **Execution Inefficiency**: Target price forecasts do not naturally translate into portfolio allocation weights without complex, noise-sensitive risk-model inversions.

The **Zetheta Alpha Foundry** shifts the operational paradigm from magnitude estimation to **relative cross-sectional ranking and propensity scoring**. Instead of estimating $E[R_{i, t+\tau}]$, the engine answers a robust, scale-invariant question:

$$\text{What is the probability } P(R_{i, t+\tau} > \text{Median}(R_{\text{universe}, t+\tau}) \mid \mathbf{X}_{i, t}) \text{ that Stock } i \text{ out-performs the universe median over a 21-day holding period?}$$

By ranking stocks relative to their peers on each trade date $t$, cross-sectional algorithms completely strip out market-wide beta trends, producing factor scores that remain stable across bull, bear, and sideways regimes.

### 1.2 System Architecture Overview
The Alpha Foundry platform integrates a multi-stage machine learning and statistical pipeline:

+-----------------------------------------------------------------------------------+
| 1. Feature Pipeline & Data Integrity                                             |
|    - Point-In-Time Ingestion (Zero Look-Ahead / Survivorship-Safe)                |
|    - 8 Microstructure, Volatility & Return Factors                                 |
|    - Daily Cross-Sectional Z-Score Transformation & Winsorization                 |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| 2. Machine Learning Core (Learning-to-Rank)                                       |
|    - LightGBM LambdaMART (NDCG Loss Optimization)                                 |
|    - Purged & Embargoed 5-Fold Time-Series Cross-Validation (21-Day Gap)          |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| 3. Calibration & Distributional Guarantees                                        |
|    - Isotonic Regression (Post-Processing Raw LambdaScores -> P(Outperform))     |
|    - Conformal Selection (Split & Mondrian Conditional Shortlists)                |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| 4. Governance, Auditability & Replication                                         |
|    - Autonomous Agent Pipeline & SHA-256 Hash-Chain Audit Trails                  |
|    - Independent R Replication (Fama-MacBeth Regressions & Rank IC Validation)    |
|    - Interactive Excel Validation Workbook (VBA & Native Audits)                 |
+-----------------------------------------------------------------------------------+

### 1.3 Key Empirical Metrics Summary
*(All metrics verified against a single, seeded, end-to-end pipeline run — `train.py` → `ensemble.py` → `conformal.py`, `seed=42` throughout — on the processed NSE liquid universe dataset)*:

- **Primary Learning-to-Rank Engine (LightGBM LambdaRank, 5-Fold Purged CV)**:
  - Overall Out-of-Sample Rank IC: **`0.0527`**
  - Per-fold Rank IC: Fold 1 `-0.0396`, Fold 2 `0.1112`, Fold 3 `0.0407`, Fold 4 `0.0943`, Fold 5 `0.0571`
  - Expected Calibration Error (ECE): **`0.0000`** (via Isotonic Regression)
- **Conformal Selection (80% Target Coverage, split-conformal)**:
  - Global threshold $\hat{q} = 0.5404$
  - Empirical Precision (of admitted names, fraction that outperformed): **`53.54%`**
  - Empirical Coverage (of true outperformers, fraction admitted): **`72.09%`** — below the 80% target. This is a genuine, expected limitation of split-conformal prediction under a strict chronological calibration/test split (first 70% of dates vs. last 30%): the theoretical coverage guarantee assumes exchangeability between calibration and test data, which a time-ordered split does not guarantee if factor-return relationships shift across the period. Reported as measured rather than adjusted to match the target.
  - Mondrian (sector-conditional) precision and coverage are **identical to split-conformal** (`53.54%` / `72.09%`), with an identical threshold (`0.5404`) assigned to all four sectors — confirming that sector conditioning currently adds no information, because sector labels are synthetic (`hash(ticker) % 4`) placeholders rather than real GICS classifications. See Limitations.
- **Ensemble & Hyperparameter Optimisation (`ensemble.py`, seed=42)**:
  - Optuna best trial (10-trial search) OOS Rank IC: **`0.0624`**; best params: `learning_rate=0.01525, num_leaves=29, min_data_in_leaf=114, feature_fraction=0.7728, bagging_fraction=0.7165`
  - Base learners (out-of-fold) — LightGBM LambdaRank (tuned): Rank IC `0.0057`, IC-IR `0.0208`; Random Forest: Rank IC `0.0104`, IC-IR `0.0377`; Ridge Linear Factor: Rank IC `0.0414`, IC-IR `0.1443`
  - Rank-Averaged Ensemble (50% LightGBM / 30% Random Forest / 20% Ridge): Rank IC `0.0107`, IC-IR `0.0391`
- **Seed Sensitivity (documented limitation)**: on the Optuna-tuned base LightGBM learner, a 3-seed test gave IC range `[0.0057, 0.0718]` (mean `0.0438`, std `0.0342`). The primary 5-fold engine above (`0.0527`) is comparatively stable under reseeding — attributable to more boosting rounds (500 vs. 200–300) and averaging across 5 purged folds rather than a single fold comparison.
- **Independent Benchmark Comparison** (`baseline.py`, equal-weighted composite of the same 8 z-scored features, evaluated on the identical 5 purged out-of-sample windows as the primary engine):
  - Baseline Equal-Weighted Linear Factor Score: Overall Rank IC **`0.0034`**, IC-IR **`0.0124`**, t-statistic **`0.4146`** — well below the $t > 2.0$ threshold for statistical significance, i.e. the naive linear composite carries no reliably detectable skill on this universe.
  - LightGBM LambdaRank engine advantage over the linear baseline: **`+0.0493`** net Rank IC (`0.0527 − 0.0034`).

---

## 2. DOMAIN FUNDAMENTALS & MATHEMATICAL FORMULATION

### 2.1 The Fundamental Law of Active Management
The efficiency of active portfolio management is governed by **Grinold's Fundamental Law of Active Management**. The Information Ratio ($IR$), which measures active return per unit of active risk, is modeled as:

$$IR \approx IC \times \sqrt{\text{Breadth}} \times TC$$

Where:
- **$IC$ (Information Coefficient)**: The correlation between predicted factor ranks and realized forward return ranks across the stock cross-section.
- **$\text{Breadth}$**: The number of statistically independent investment decisions executed per year.
- **$TC$ (Transfer Coefficient)**: The efficiency with which unconstrained quantitative forecasts are mapped into actual portfolio weights after netting transaction costs, liquidity bounds, and short-sale constraints ($0 \le TC \le 1$).

#### Derivation of the Expected Information Ratio
Assuming an investment universe of $N$ assets rebalanced $T$ times per year with independent errors across time:

$$\text{Breadth} = N \times T$$

For a universe of $N = 15$ liquid securities rebalanced monthly ($T = 12$ periods per year):

$$\text{Breadth} = 15 \times 12 = 180 \text{ opportunity bets/year}$$

Given the verified seeded Out-of-Sample $IC = 0.0527$ (primary 5-fold LambdaRank engine) and an **assumed** institutional execution Transfer Coefficient $TC = 0.65$ *(this value is not measured from the pipeline — it is a standard illustrative assumption for institutional execution efficiency and should be labeled as such wherever it is cited; it should be replaced with a measured value once turnover and transaction-cost data are run through Section A12's back-test)*:

$$IR \approx 0.0527 \times \sqrt{180} \times 0.65 \approx 0.0527 \times 13.416 \times 0.65 \approx \mathbf{0.460}$$

This confirms that even modest Rank IC values ($0.03 - 0.06$) yield institutional-grade active Information Ratios when scaled over continuous cross-sectional rebalancing. Note that this $IR$ is a **theoretical projection** from the Fundamental Law identity given an assumed $TC$ — it is not yet a measured, back-tested Information Ratio. A measured $IR$ should be reported separately once the decile back-test / portfolio overlay (Section A12 of the project methodology) has been run.

---

### 2.2 Mathematical Definition of Cross-Sectional Metrics

To evaluate model performance without assumption of normality or linear linearity, we rely on rank-based non-parametric statistics.

#### 1. Spearman's Rank Information Coefficient (Rank IC)
For a specific trade date $t$ with $N_t$ stocks, let $\mathbf{\hat{r}}_t$ denote the vector of predicted ranks and $\mathbf{r}_{t+\tau}$ denote the vector of realized forward return ranks over period $\tau$:

$$\text{Rank IC}_t = 1 - \frac{6 \sum_{i=1}^{N_t} d_{i, t}^2}{N_t(N_t^2 - 1)}$$

Where $d_{i, t} = \text{Rank}(\hat{y}_{i, t}) - \text{Rank}(y_{i, t+\tau})$.

The mean out-of-sample Rank IC across $M$ backtest dates is:

$$\overline{\text{Rank IC}} = \frac{1}{M} \sum_{t=1}^{M} \text{Rank IC}_t$$

#### 2. Information Coefficient Information Ratio (IC-IR)
The consistency of factor alpha over time is measured by the ratio of mean Rank IC to its volatility across time periods:

$$\text{IC-IR} = \frac{\overline{\text{Rank IC}}}{\sigma(\text{Rank IC}_t)}$$

Where $\sigma(\text{Rank IC}_t) = \sqrt{\frac{1}{M-1} \sum_{t=1}^{M} (\text{Rank IC}_t - \overline{\text{Rank IC}})^2}$.

#### 3. Statistical Significance (t-statistic)
The t-statistic tests the null hypothesis $H_0: \overline{\text{Rank IC}} = 0$:

$$t_{\text{stat}} = \text{IC-IR} \times \sqrt{M}$$

A $t_{\text{stat}} > 2.0$ indicates statistical significance at the 95% confidence level ($p < 0.05$).

---

### 2.3 Learning-to-Rank vs. Classification vs. Regression

| Dimension | Standard Regression (MSE) | Binary Classification (LogLoss) | Learning-to-Rank (LambdaMART) |
| :--- | :--- | :--- | :--- |
| **Objective Function** | $\sum (y_i - \hat{y}_i)^2$ | $-\sum [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$ | Optimizes Normalized Discounted Cumulative Gain (NDCG) |
| **Loss Sensitivity** | Sensitive to extreme price outliers | Treats near-median stocks same as top performers | Focuses gradient updates on top and bottom ranked items |
| **Cross-Sectional Focus**| Absolute per-stock drift | Binary class boundary | Direct pairwise/listwise ordinal sorting |
| **Market Regime Robustness** | Fails during high volatility regimes | Distorted by market-wide directional shifts | Scale-invariant across all macro environments |

#### Pairwise LambdaLoss Derivation
LambdaMART optimizes pairwise ranking ordering by dynamically weighting pairs based on their impact on NDCG. For a pair of stocks $i$ and $j$ on date $t$ where stock $i$ has a higher realized forward return than stock $j$ ($y_i > y_j$):

$$C_{ij} = \log\left(1 + e^{-\sigma (\hat{s}_i - \hat{s}_j)}\right)$$

Where $\hat{s}_i$ and $\hat{s}_j$ are raw predicted scores from the decision tree ensemble. The gradient $\lambda_{ij}$ with respect to score difference $\Delta \hat{s}_{ij} = \hat{s}_i - \hat{s}_j$ is scaled by the change in NDCG ($\Delta \text{NDCG}$) that results from swapping the ranks of item $i$ and item $j$:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma (\hat{s}_i - \hat{s}_j)}} \cdot |\Delta \text{NDCG}|$$

This ensures that the model spends the majority of its capacity ordering assets at the top of the selection queue (the long portfolio candidates) rather than wasting parameter space on uninteresting middle-tier stocks.

---

### 2.4 Probability Calibration (Isotonic Regression)
Raw LambdaMART outputs ($\hat{s}_i \in \mathbb{R}$) represent arbitrary ordinal scores, not calibrated probabilities. To convert these scores into $P(\text{Outperform})$, we fit a monotonic post-processing transformation $m(\cdot)$ using Isotonic Regression:

$$\min_{m} \sum_{i=1}^N w_i \left( y_i - m(\hat{s}_i) \right)^2 \quad \text{subject to } m(\hat{s}_a) \le m(\hat{s}_b) \text{ whenever } \hat{s}_a \le \hat{s}_b$$

Where $y_i \in \{0, 1\}$ is the binary outperformance indicator. Isotonic regression solves this under monotonicity constraints via the **Pool Adjacent Violators Algorithm (PAVA)**.

#### Expected Calibration Error (ECE)
To evaluate calibration accuracy, we bin predicted probabilities $p_i = m(\hat{s}_i)$ into $K = 10$ equally spaced intervals $B_k$:

$$\text{ECE} = \sum_{k=1}^K \frac{|B_k|}{N} \left| \text{acc}(B_k) - \text{conf}(B_k) \right|$$

Where:
- $\text{acc}(B_k) = \frac{1}{|B_k|} \sum_{i \in B_k} y_i$ (empirical fraction of outperforming stocks).
- $\text{conf}(B_k) = \frac{1}{|B_k|} \sum_{i \in B_k} p_i$ (average predicted probability).

A calibrated engine achieves $\text{ECE} \approx 0.0000$, ensuring that a score of $0.70$ corresponds to an exact 70% historical empirical probability of outperforming the universe median.

---

## 3. RESULTS

### 3.1 Model Comparison Table

*(All figures verified against a single seeded pipeline run — `baseline.py` → `train.py` → `ensemble.py` → `conformal.py`, `seed=42` throughout.)*

| Model | Overall Rank IC | IC-IR | t-statistic | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Baseline — equal-weighted linear composite (8 z-scored features) | `0.0034` | `0.0124` | `0.4146` | Not statistically significant ($t < 2.0$) |
| **Primary Engine — LightGBM LambdaRank, 5-fold purged CV** | **`0.0527`** | see §3.2 (per-fold) | see §3.2 | Headline verified result |
| Ensemble base learner — LightGBM (Optuna-tuned, within `ensemble.py`) | `0.0057` | `0.0208` | — | See §3.2 discrepancy note |
| Ensemble base learner — Random Forest | `0.0104` | `0.0377` | — | |
| Ensemble base learner — Ridge Linear Factor | `0.0414` | `0.1443` | — | Best individual base learner |
| Rank-Averaged Ensemble (50% LightGBM / 30% RF / 20% Ridge) | `0.0107` | `0.0391` | — | Underperforms the primary engine — see §3.2 |
| Optuna best single trial (10-trial HPO search, within `ensemble.py`) | `0.0624` | — | — | Best-of-10-trials search value, not a cross-validated headline figure |

**Engine advantage over baseline:** `+0.0493` Rank IC (`0.0527 − 0.0034`).

### 3.2 Out-of-Sample Stability Discussion

**Fold-level variability.** The primary engine's five purged folds range from `-0.0396` (Fold 1) to `0.1112` (Fold 2) — a wide spread that means the headline `0.0527` is an average over folds with genuinely different, and in one case negative, skill. IC-IR values per fold (`-0.1511` to `0.3915`) confirm this: the signal is not uniformly present across the sample period, and a reader should not treat `0.0527` as a stable, regime-independent number without qualification.

**Seed sensitivity.** A 3-seed stability test on the Optuna-tuned base LightGBM learner produced IC values of `0.0057`, `0.0718`, and `0.0539` (mean `0.0438`, std `0.0342`) — a range wide enough that a single unseeded run could plausibly report anywhere from "no skill" to "IC over 0.07" purely from random initialization. The primary 5-fold engine is comparatively more stable under reseeding (`0.0527` seeded vs. `0.0552` unseeded, ~5% relative difference), most plausibly because it averages across 5 purged folds with more boosting rounds (500) rather than reporting a single fold's result with fewer rounds (200–300, as used inside the HPO search and ensemble base learners).

**Open discrepancy — ensemble's own LightGBM learner underperforms the primary engine.** The `ensemble.py` base learner labeled "LightGBM LambdaRank" scores `IC 0.0057` — an order of magnitude below the primary engine's `0.0527`, despite both being LightGBM LambdaRank models on the same data. This is not explained away here; it is flagged as an open question. Plausible contributing factors, none yet confirmed: (a) fewer boosting rounds in the ensemble context (300 vs. 500, with early stopping at 30 rounds vs. 50); (b) the Optuna search itself only ran 10 trials, which may not have found parameters that generalize as well as `train.py`'s fixed, hand-set parameters; (c) the specific purged-fold IC being averaged in the HPO objective may weight folds differently than the primary engine's evaluation. Until isolated, this discrepancy should be treated as a known limitation of the ensembling pipeline, not resolved in the narrative.

**Consequence for the ensemble.** Because its constituent LightGBM learner is unexpectedly weak, the **rank-averaged ensemble (`IC 0.0107`) underperforms the standalone primary engine (`0.0527`)** by a wide margin, even though the best individual base learner (Ridge, `0.0414`) approaches it. This is the opposite of the intended effect of ensembling (Section A11.1 of the project brief: ensembling should reduce variance without sacrificing skill). As it stands, **the primary engine — not the ensemble — is the defensible champion model**, and this should be stated plainly in any champion-challenger decision (Section A11.8) rather than assumed in the ensemble's favor by default.

### 3.3 Calibration & Conformal Evidence

**Calibration.** The primary engine's isotonic-calibrated propensities achieve $\text{ECE} = 0.0000$ on out-of-fold predictions — no measurable gap between stated confidence and empirical outcome frequency across the 10 calibration bins.

**Conformal selection.** At a target 80% coverage level ($\alpha = 0.20$), split-conformal selection produced a global threshold $\hat{q} = 0.5404$, yielding:
- **Empirical precision `53.54%`** — of names admitted into the "likely out-performer" shortlist, just over half actually outperformed.
- **Empirical coverage `72.09%`** — below the 80% target. This is reported as measured, not adjusted. The shortfall is consistent with a known limitation of split-conformal prediction under a strict chronological calibration/test split (first 70% of dates vs. last 30%): the method's finite-sample coverage guarantee relies on exchangeability between calibration and test data, which a time-ordered split does not guarantee if the feature-to-return relationship shifts across the period (see §3.5).

**Mondrian conformal.** Sector-conditional thresholds were computed for four synthetic sectors, all of which converged to the **identical** threshold (`0.5404`) and identical precision/coverage as the global split-conformal result. This is expected, not a bug: sector labels are currently assigned by `hash(ticker) % 4`, carrying no genuine sector information, so conditioning on them cannot differ from the unconditional case. Mondrian conformal should be considered **not yet functional** until real GICS or NSE sector classifications are substituted.

### 3.5 Limitations

The following limitations are carried forward from the verification work in this section and should accompany any presentation of the headline results:

1. **Small universe.** The dataset spans roughly 15 liquid tickers, which is sufficient to demonstrate the methodology but small enough that IC estimates carry meaningful sampling variance — demonstrated directly by the 3-seed stability test (§3.2).
2. **Ensemble underperforms the primary engine.** The advanced model-development stack (ensembling, stacking, Optuna HPO) currently produces a *worse* signal (`IC 0.0107`) than the primary 5-fold engine (`0.0527`), for reasons not yet fully isolated (§3.2). The primary engine, not the ensemble, is the current champion model.
3. **Mondrian conformal is non-functional.** Sector-conditional coverage guarantees cannot be assessed until real sector classifications replace the current hash-based placeholders.
4. **Conformal coverage undershoots target.** Empirical coverage (`72.09%`) falls short of the 80% target under a chronological calibration/test split, likely reflecting non-exchangeability between the calibration and test periods rather than a implementation error.
5. **Baseline is a diagnostic floor, not a tuned competitor.** The `0.0034` linear baseline uses simple equal weighting with no fitting; it establishes that the engine's `0.0527` is meaningfully above an untrained floor, but does not rule out that a *fitted* linear model (e.g., Fama-MacBeth-weighted, per Section A3.3 of the project brief) might close some of that gap.
6. **Transfer Coefficient is unmeasured.** $TC = 0.65$ in Section 2.1 is an illustrative assumption, not a value derived from this project's data. The Information Ratio computed from it ($IR \approx 0.460$) is a theoretical projection, not a back-tested result.
7. **No back-tested portfolio Information Ratio exists yet.** Section 3.4 (decile back-test, turnover, net-of-cost spread) has not yet been built; until it is, no realized IR, transaction-cost-adjusted spread, or transfer coefficient can be reported as measured.
8. **No pooled significance test exists for the primary engine.** `train.py` reports the simple mean of five fold-level ICs (`0.0527`) but never pools daily Rank IC values across all out-of-sample dates into one series to compute an overall IC-IR and t-statistic (unlike `baseline.py`, which does this). The primary engine's result should therefore be read as directionally strong and well above baseline, but not yet accompanied by a formal significance test of its own.

---

## 4. CONCLUSION

### 4.1 What Was Verified

Across a single, seed-fixed, reproducible pipeline run (`baseline.py` → `train.py` → `ensemble.py` → `conformal.py`, `seed=42` throughout, confirmed identical on re-run), the project establishes the following, with no invented or unverified figures anywhere in this report:

- The primary cross-sectional LightGBM LambdaRank engine achieves an out-of-sample Rank IC of **`0.0527`**, meaningfully above a naive equal-weighted linear baseline (`0.0034`, not statistically significant) evaluated on identical out-of-sample windows.
- Isotonic calibration produces a well-calibrated propensity reading (**ECE = 0.0000**).
- Conformal selection is implemented and functioning for the global (split-conformal) case, achieving **53.54% precision** and **72.09% coverage** against an 80% target — the coverage shortfall is explained, not concealed, as a known consequence of evaluating conformal guarantees across a non-exchangeable, chronologically-split calibration/test period.
- Two genuine, unresolved findings were surfaced rather than hidden: the ensembling/stacking stack currently **underperforms** the standalone primary engine, and Mondrian (sector-conditional) conformal is **not yet functional** because sector labels are synthetic placeholders rather than real classifications.

### 4.2 What Remains

This report and codebase currently satisfy a meaningful subset of Zetheta's required deliverables (Part D, Section D3), but not all of them. In the interest of an audit-defensible submission, the honest status is:

- **Substantially complete:** primary engine, calibration, conformal selection (global), baseline comparison, documented limitations.
- **Not yet built:** the decile back-test / portfolio overlay (Section 3.4) and a measured Information Ratio; stacked (meta-learner) ensembling and feature-importance stability selection; a formal champion-challenger promotion mechanism; a Monte Carlo shortlist-outcome engine; real sector classifications; the agentic AWS pipeline and its compliance artefacts (lineage, model cards, audit log); the independent R replication; the Excel validation workbook; and the final presentation and demonstration video.

### 4.3 Recommendation

Given the gap between what fifteen days of the full methodology calls for and what has been completed, the most defensible path is to submit this report explicitly framed as an **interim, verified-but-partial submission** — every number in it is real and reproducible, and every known gap is named rather than glossed over. This is consistent with, not contrary to, the project's own stated emphasis on audit-defensibility (Section A10 of the project brief): a smaller set of honestly-verified results is more defensible than a complete-looking report containing unverified or fabricated figures.

---

## 5. AGENTIC PIPELINE & AWS INFRASTRUCTURE — DESIGN SPECIFICATION

**Status: designed, not deployed.** Under the submission deadline, no AWS account was provisioned and no component below was run against live infrastructure. This section is included so the architectural thinking required by the project brief (Part A, Section A9) is demonstrated and auditable, but it must not be read as evidence of a working deployment. Everything in this section is a specification for future implementation.

### 5.1 Reference Architecture for This Project

Mapped specifically to this codebase's actual scripts, not a generic template:

| Stage | AWS Service | What Runs Here (this project's real scripts) |
| :--- | :--- | :--- |
| Raw ingestion | S3 (raw zone) | NSE price/volume history landing as-is, immutable |
| Feature build | Glue (PySpark) | Point-in-time z-scoring logic currently in `train.py`/`ensemble.py`/`baseline.py`'s shared feature block, refactored into a standalone Glue job |
| Model training | SageMaker Training | `train.py`'s 5-fold purged-CV LightGBM LambdaRank training |
| Ensembling & HPO | SageMaker Processing | `ensemble.py`'s Optuna search and rank-averaging |
| Conformal scoring | SageMaker Processing / Lambda | `conformal.py`'s split & Mondrian selection |
| Back-test | SageMaker Processing | `backtest.py`'s OOF scoring and decile-spread calculation |
| Orchestration | Step Functions + EventBridge | Daily/rebalance-cadence DAG chaining the above in order |
| Audit trail | CloudTrail + append-only hash-chain log | Every stage's inputs/outputs hashed and chained, per Section A10.3 of the brief |

### 5.2 Step Functions State Machine (Specification)

```json
{
  "Comment": "Zetheta cross-sectional ranking pipeline (design spec, not deployed)",
  "StartAt": "IngestRawPrices",
  "States": {
    "IngestRawPrices": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:zetheta-ingest",
      "Next": "BuildFeatures"
    },
    "BuildFeatures": {
      "Type": "Task",
      "Resource": "arn:aws:states:::glue:startJobRun.sync",
      "Parameters": { "JobName": "zetheta-build-pit-features" },
      "Next": "TrainPrimaryEngine"
    },
    "TrainPrimaryEngine": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": { "TrainingJobName.$": "$.run_id", "AlgorithmSpecification": { "TrainingImage": "zetheta-lambdarank:latest" } },
      "Next": "RunEnsembleHPO"
    },
    "RunEnsembleHPO": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Next": "RunConformalSelection"
    },
    "RunConformalSelection": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Next": "RunBacktest"
    },
    "RunBacktest": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Next": "EmitAuditArtifacts"
    },
    "EmitAuditArtifacts": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:zetheta-audit-log",
      "End": true
    }
  }
}
```

### 5.3 Model Card Schema (populated with THIS project's real values)

```python
def build_model_card():
    return {
        "model_id": "zetheta-lambdarank-v1",
        "universe": "NSE liquid universe, ~15 tickers",
        "train_period": "see processed_factors.parquet date range",
        "label_definition": "5-quintile relevance grade on fwd_ret_21d, cross-sectional",
        "features": "8 features: ret_1m, ret_3m, ret_12m, mom_12m_1m, daily_ret, vol_20d, dollar_volume, amihud_illiquidity (all cross-sectionally z-scored)",
        "cv_scheme": "5-fold purged, 21-day embargo",
        "oos_mean_ic": 0.0527,
        "oos_ece": 0.0000,
        "conformal_coverage_split": 0.7209,
        "conformal_coverage_target": 0.80,
        "known_limitations": [
            "small universe -> seed-sensitive IC (documented, 3-seed test run)",
            "ensemble underperforms primary engine (unresolved, documented)",
            "Mondrian conformal non-functional (synthetic sector labels)",
            "conformal coverage below target (72.09% vs 80%, time-split non-exchangeability)"
        ],
        "approved_by": "PENDING — requires human sign-off before any real deployment",
        "deployment_status": "NOT DEPLOYED — design specification only"
    }
```

### 5.4 What Would Be Needed for Real Deployment

Honestly scoped, not glossed over: an AWS account with IAM roles for each stage, the four scripts (`train.py`, `ensemble.py`, `conformal.py`, `backtest.py`) refactored to read/write S3 paths instead of local `data/` paths, a Glue job wrapping the shared feature-engineering block, and the hash-chain audit logger from Section A10.3 of the brief wired into each stage's Lambda/Processing job. None of this was built or tested in this session.
