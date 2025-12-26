# 🏦 Stock Price Risk Prediction Model

Predicting stock prices is only part of financial analytics — **quantifying risk associated with future price movements is equally critical**.  
This project builds an **end-to-end risk prediction pipeline** using historical stock prices, machine learning models, and interactive dashboards to support risk-aware investment decisions.

---

## 🚀 Project Overview

This repository provides a complete workflow for:

- Cleaning and preprocessing historical stock price data  
- Engineering features for risk modeling  
- Training machine learning models to estimate future risk  
- Evaluating model performance and risk metrics  
- Visualizing results through an interactive dashboard  

The project focuses on **risk analytics rather than pure price prediction**, making it suitable for applications in **Risk Management, Quantitative Finance, and Portfolio Analysis**.

---

## 🔑 Key Features

- **Multiple risk models**
  - LightGBM
  - XGBoost
  - Monte Carlo simulation

- **Risk-oriented outputs**
  - Future return distributions
  - Quantile-based risk levels (1%, 10%, 30%, 50%)
  - Group-level risk summaries

- **Interactive dashboard**
  - Visualizes risk trends across tickers
  - Compares model outputs
  - Supports exploratory analysis

- **Modular project structure**
  - Easy to extend with new models or datasets

---

## 📊 Data Description

The project uses historical price data for multiple stock tickers.

| File | Description |
|-----|------------|
| `forecast_all_tickers.csv` | Aggregated historical price data |
| `risk_*_tickers_*.csv` | Risk estimation outputs from different models |
| `ticker_ratings.csv` | Additional rating or grouping information |

> **Note:**  
> Large raw price datasets are excluded from the repository due to GitHub size limits.  
> They can be regenerated using scripts in the `src/` directory or obtained from public data sources.

---

## 🧠 Modeling Approach

### Machine Learning Models
- **LightGBM & XGBoost** are trained on engineered features derived from historical returns and volatility patterns.
- Models estimate risk levels through quantile-based predictions.

### Monte Carlo Simulation
- Generates multiple future price paths
- Estimates the distribution of possible returns
- Used to assess downside risk and uncertainty

### Evaluation
- Backtesting on historical periods
- Validation metrics stored as CSV for analysis and reporting

---

## 🛠 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt

---

### Data preprocessing
```bash
python src/data_preprocessing.py

