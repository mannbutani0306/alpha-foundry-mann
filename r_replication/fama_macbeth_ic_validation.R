# Independent R Replication & Validation Script for Zetheta Ranking Engine
# Fama-MacBeth Cross-Sectional Regression & Out-of-Sample Rank IC Validation

options(repos = c(CRAN = "https://cloud.r-project.org"))

# Set up a user-writable package library directory to bypass Windows Program Files permission limits
user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library", "4.6")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c("arrow", "dplyr", "data.table")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing missing R package: %s into user library...\n", pkg))
    install.packages(pkg, lib = user_lib, dependencies = TRUE)
  }
}

suppressPackageStartupMessages({
  library(arrow, lib.loc = user_lib)
  library(dplyr, lib.loc = user_lib)
  library(data.table, lib.loc = user_lib)
})

cat("--------------------------------------------------\n")
cat("Starting R Replication & Validation Pipeline...\n")

data_path <- "data/processed/processed_factors.parquet"

if (!file.exists(data_path)) {
  stop(paste("Parquet data file not found at", data_path))
}

# 1. Load Parquet Data
df <- read_parquet(data_path)
cat(sprintf("Successfully loaded %d records across %d columns.\n", nrow(df), ncol(df)))

# 2. Identify and Z-Score Numeric Factors per Date
numeric_factors <- c("ret_1m", "ret_3m", "ret_12m", "mom_12m_1m", "daily_ret", "vol_20d", "dollar_volume", "amihud_illiquidity")
factors_present <- intersect(numeric_factors, colnames(df))

for (factor in factors_present) {
  z_col <- paste0(factor, "_z")
  df <- df %>%
    group_by(date) %>%
    mutate(!!z_col := (get(factor) - mean(get(factor), na.rm=TRUE)) / (sd(get(factor), na.rm=TRUE) + 1e-9)) %>%
    ungroup()
}

df[is.na(df)] <- 0.0

# 3. Fama-MacBeth Cross-Sectional Regression
cat("Running Fama-MacBeth Cross-Sectional Regressions...\n")
unique_dates <- unique(df$date)
betas_list <- list()

for (d in unique_dates) {
  sub_df <- df[df$date == d, ]
  if (nrow(sub_df) >= 10) {
    z_cols <- paste0(factors_present, "_z")
    formula_str <- paste("fwd_ret_21d ~", paste(z_cols, collapse = " + "))
    fit <- lm(as.formula(formula_str), data = sub_df)
    betas_list[[as.character(d)]] <- coef(fit)
  }
}

if (length(betas_list) > 0) {
  betas_df <- do.call(rbind, betas_list)
  mean_betas <- colMeans(betas_df, na.rm = TRUE)
  sd_betas <- apply(betas_df, 2, sd, na.rm = TRUE)
  t_stats <- mean_betas / (sd_betas / sqrt(nrow(betas_df)))

  fama_macbeth_summary <- data.frame(
    Factor = names(mean_betas),
    Mean_Beta = mean_betas,
    t_stat = t_stats
  )

  print(fama_macbeth_summary)
}

# 4. Compute Daily Rank IC for Equal-Weighted Composite Factor
z_cols <- paste0(factors_present, "_z")
df$composite_score <- rowMeans(df[, z_cols, drop=FALSE])

compute_daily_ic <- function(data_dt, score_col, target_col) {
  dates <- unique(data_dt$date)
  ics <- numeric(length(dates))
  
  for (i in seq_along(dates)) {
    sub_dt <- data_dt[data_dt$date == dates[i], ]
    if (nrow(sub_dt) >= 5) {
      ics[i] <- cor(sub_dt[[score_col]], sub_dt[[target_col]], method = "spearman")
    } else {
      ics[i] <- NA
    }
  }
  return(na.omit(ics))
}

rank_ics <- compute_daily_ic(df, score_col = "composite_score", target_col = "fwd_ret_21d")

mean_ic <- mean(rank_ics)
ic_std <- sd(rank_ics)
ic_ir <- mean_ic / (ic_std + 1e-12)
t_stat_ic <- ic_ir * sqrt(length(rank_ics))

cat("--------------------------------------------------\n")
cat("Independent R Validation Results:\n")
cat(sprintf("  Mean Rank IC: %.4f\n", mean_ic))
cat(sprintf("  IC-IR:        %.4f\n", ic_ir))
cat(sprintf("  t-statistic:  %.4f\n", t_stat_ic))
cat("--------------------------------------------------\n")