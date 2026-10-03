# Next-Day S&P 500 Direction Prediction

This project tests how well common machine-learning classifiers predict whether the S&P 500 (SPY) closes up the next day, using technical indicators as features. Every model is benchmarked out of sample against a naive base-rate forecast.

## Setup

- **Data:** Daily SPY prices from 2015 to 2024, downloaded with `tidyquant`.
- **Target:** `up_tomorrow` equals 1 if the next day's log-return is positive.
- **Features:** Lagged returns (1, 2, 10 and 30 days), the distance from 5-, 10- and 30-day simple moving averages, and further momentum and volatility indicators (`TTR`, `zoo`).
- **Split:** Chronological, with the first 85% as training data and the last 15% as test data, so there is no look-ahead.

## Models

| Model | Package |
|---|---|
| Logistic regression | `stats::glm` |
| LASSO variable selection + logit | `lars` (cross-validated) |
| L1-penalised logistic regression | `glmnet` (10-fold CV) |
| Decision tree (default and tuned depth) | `rpart` |
| Random forest | `randomForest` |
| GAM-logit | `mgcv` |
| Naive base-rate benchmark | none |

Each model is scored on the test set by AUC, accuracy, the confusion matrix and normalised deviance (`ROCR`).

## Takeaway

At a one-day horizon, returns are close to a random walk. The real test is whether a model beats the naive base-rate forecast out of sample. The project compares model families on that question and shows how flexible models such as deep trees overfit noisy financial data.

## Run

Run `R/sp500_direction_ml.R`. It installs missing packages and downloads the data itself.
