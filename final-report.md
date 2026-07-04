# Auto-Train Final Report — House Prices

**Objective:** minimize log-space RMSE of `log1p(SalePrice)` (matches Kaggle leaderboard metric).

**Status:** CONVERGED  |  **Experiments:** 9  |  **Architecture classes:** linear, neighbors, svm, tree_based (4)

**Baseline (exp_000 Ridge):** 0.12332

**Winner:** `exp_ensemble`  CV log-RMSE **0.10711**


## Leaderboard (5-fold CV, log-space RMSE)

| rank | experiment | class | cv_rmse | fold_std | beats baseline |
|---|---|---|---|---|---|
| 1 | exp_001_lasso | linear | 0.10998 | 0.00633 | True |
| 2 | exp_005_gbr | tree_based | 0.11033 | 0.00719 | True |
| 3 | exp_002_elasticnet | linear | 0.11035 | 0.00630 | True |
| 4 | exp_003_xgboost | tree_based | 0.11332 | 0.00834 | True |
| 5 | exp_007_kernelridge | svm | 0.11544 | 0.00741 | True |
| 6 | exp_004_lightgbm | tree_based | 0.11699 | 0.00999 | True |
| 7 | exp_000_baseline | linear | 0.12332 | 0.00377 | True |
| 8 | exp_008_knn | neighbors | 0.16103 | 0.01906 | False |
| 9 | exp_006_svr | svm | 0.16989 | 0.02209 | False |
| — | **exp_ensemble** | ensemble | **0.10711** | — | True |

## Caruana ensemble

Greedy forward selection with replacement, 44 rounds. Improvement over best single (`exp_001_lasso` 0.10998): **0.00288**.

| component | weight |
|---|---|
| exp_001_lasso | 0.409 |
| exp_005_gbr | 0.341 |
| exp_007_kernelridge | 0.159 |
| exp_004_lightgbm | 0.091 |

## Convergence

Tier 2 coverage: 4 classes >= 3 required → True.

| class | variants | best cv_rmse | exhausted | reason |
|---|---|---|---|---|
| linear | 3 | 0.10998 | True | multiple_variants |
| neighbors | 1 | 0.16103 | True | non_competitive(+46%) |
| svm | 2 | 0.11544 | True | multiple_variants |
| tree_based | 3 | 0.11033 | True | multiple_variants |

## Submission

`submission.csv` written: 1459 rows, Id-aligned to sample_submission.csv, all predictions positive. SalePrice range $41,884–$1,200,896, median $156,014.
