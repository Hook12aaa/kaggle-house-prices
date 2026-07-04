# Kaggle House Prices -- Advanced Regression

This is an autonomous run of my [auto-model-trainer](https://github.com/Hook12aaa/auto-model-trainer) plugin on the House Prices competition. I handed it the objective file and let it explore on its own -- 9 experiments across 4 architecture classes (linear, neighbors, svm, tree_based). It converged on a Caruana ensemble blending Lasso, GBR, KernelRidge, and LightGBM as the winner.

## Competition
- **Task:** Regression -- predict `SalePrice` for residential homes in Ames, Iowa.
- **Metric:** RMSE on log-transformed sale prices (lower is better).
- **Data:** 1460 training rows, 1459 test rows, 79 explanatory features.
- **Link:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Results
Public leaderboard RMSE of **0.12556**. The winning ensemble scored a CV RMSE of **0.10711**.

Experiment leaderboard from the final report:

| Experiment | Class | CV RMSE |
|---|---|---|
| exp_ensemble | ensemble | 0.10711 |
| exp_001_lasso | linear | 0.10998 |
| exp_005_gbr | tree_based | 0.11033 |
| exp_002_elasticnet | linear | 0.11035 |
| exp_003_xgboost | tree_based | 0.11332 |
| exp_007_kernelridge | svm | 0.11544 |
| exp_004_lightgbm | tree_based | 0.11699 |
| exp_000_baseline | linear | 0.12332 |
| exp_008_knn | neighbors | 0.16103 |
| exp_006_svr | svm | 0.16989 |

The ensemble weights: Lasso 40.9%, GBR 34.1%, KernelRidge 15.9%, LightGBM 9.1%.

## Project Structure
```
objective.yaml                    # the goal I handed the trainer
final-report.md                   # what it tried, what won, and why
submission.csv                    # winner's predictions on the test set
.auto-trainer/
  experiment-tree.json            # lineage of every variant explored
  data_report.json                # data validation results
  experiments/                    # one folder per variant (run.py + EVALUATION.json)
  lib/                            # shared feature engineering
  scripts/                        # convergence, ensemble, and report scripts
```

## Usage
Reproduce with the auto-model-trainer plugin:
```
/auto-train objective.yaml
```
The plugin is here: https://github.com/Hook12aaa/auto-model-trainer
