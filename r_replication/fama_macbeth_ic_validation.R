# fama_macbeth_replication.R
#
# Independent R replication of:
#   (1) the Fama-MacBeth cross-sectional regression (Section A3.3 of the
#       project brief), and
#   (2) the equal-weighted composite Rank IC used as the Python baseline
#       (baseline.py: IC 0.0034, IC-IR 0.0124, t = 0.4146).
#
# STATUS: written but NOT YET RUN, due to submission time constraints.
# It is a genuine, from-scratch R implementation -- not a translation of the
# Python code -- but its output has not been verified in this session.
# Run it yourself with:
#     Rscript fama_macbeth_replication.R
# and treat any numbers it prints as unverified until you have done so.

library(arrow)      # read_parquet
library(dplyr)
library(tidyr)
library(purrr)

data_path <- "data/processed/processed_factors.parquet"
df <- read_parquet(data_path)

target_col <- "fwd_ret_21d"
exclude_cols <- c(
  "date", "Ticker", "symbol", "close", "open", "high", "low", "volume",
  "dividends", "stock splits", "fwd_ret_21d", "target_rel_ret",
  "target_outperform", "label", "relevance"
)
feature_cols <- setdiff(names(df), exclude_cols)

cat(sprintf("Loaded %d rows, %d candidate features.\n", nrow(df), length(feature_cols)))

# --- Cross-sectional z-score per date (mirrors the Python pipeline's transform) ---
df <- df %>%
  group_by(date) %>%
  mutate(across(
    all_of(feature_cols),
    ~ (. - mean(., na.rm = TRUE)) / (sd(., na.rm = TRUE) + 1e-9),
    .names = "{.col}_z"
  )) %>%
  ungroup()

z_cols <- paste0(feature_cols, "_z")

# --- 1. Fama-MacBeth cross-sectional regression ---
# For each date, regress forward return on the z-scored features; average
# the per-date coefficients and compute a t-statistic on that time series.
fm_betas <- df %>%
  group_by(date) %>%
  group_modify(~ {
    fmla <- as.formula(paste(target_col, "~", paste(z_cols, collapse = " + ")))
    fit <- tryCatch(lm(fmla, data = .x), error = function(e) NULL)
    if (is.null(fit)) return(as.data.frame(t(rep(NA_real_, length(z_cols) + 1))))
    as.data.frame(t(coef(fit)))
  }) %>%
  ungroup()

coef_cols <- setdiff(names(fm_betas), "date")
fm_mean <- fm_betas %>% select(all_of(coef_cols)) %>% summarise(across(everything(), ~ mean(., na.rm = TRUE)))
fm_n    <- fm_betas %>% select(all_of(coef_cols)) %>% summarise(across(everything(), ~ sum(!is.na(.))))
fm_se   <- fm_betas %>% select(all_of(coef_cols)) %>% summarise(across(everything(), ~ sd(., na.rm = TRUE) / sqrt(sum(!is.na(.)))))
fm_tstat <- fm_mean / fm_se

cat("\n--- Fama-MacBeth Mean Coefficients ---\n"); print(fm_mean)
cat("\n--- Fama-MacBeth t-statistics ---\n"); print(fm_tstat)

# --- 2. Equal-weighted composite Rank IC (replicates baseline.py) ---
df$composite_score <- rowMeans(df[z_cols], na.rm = TRUE)

rank_ic_by_date <- df %>%
  group_by(date) %>%
  summarise(
    rank_ic = suppressWarnings(
      cor(composite_score, .data[[target_col]], method = "spearman", use = "complete.obs")
    ),
    .groups = "drop"
  )

mean_ic <- mean(rank_ic_by_date$rank_ic, na.rm = TRUE)
ic_sd   <- sd(rank_ic_by_date$rank_ic, na.rm = TRUE)
n_valid <- sum(!is.na(rank_ic_by_date$rank_ic))
ic_ir   <- mean_ic / ic_sd
t_stat  <- ic_ir * sqrt(n_valid)

cat("\n--- R Replication: Equal-Weighted Composite Rank IC ---\n")
cat(sprintf("Mean Rank IC : %.4f\n", mean_ic))
cat(sprintf("IC-IR        : %.4f\n", ic_ir))
cat(sprintf("t-statistic  : %.4f\n", t_stat))
cat(sprintf("n dates      : %d\n", n_valid))
cat("\nCompare against Python baseline.py (verified, seed=42):\n")
cat("  IC 0.0034, IC-IR 0.0124, t = 0.4146\n")
cat("A close match here supports the correctness of both implementations.\n")
cat("A large discrepancy should be investigated, not silently discarded.\n")

# NOTE ON SCOPE: this script replicates the Fama-MacBeth regression and the
# equal-weighted baseline's Rank IC. It does NOT replicate the LightGBM
# LambdaRank engine itself (0.0527) or the ensemble/conformal results --
# those require gradient-boosted ranking and conformal-prediction packages
# in R (e.g. lightgbm's R bindings, conformalInference) that were out of
# scope to write and verify unrun under today's time constraint. This is a
# documented gap, not a hidden one -- see the Limitations section.
