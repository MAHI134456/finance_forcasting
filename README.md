# GMF Investments – Time Series Forecasting & Portfolio Optimization

## Project Overview
Guide Me in Finance (GMF) Investments is a forward-thinking financial advisory firm specializing in **personalized portfolio management**.  
This project applies **time series forecasting** and **Modern Portfolio Theory (MPT)** to historical financial data in order to **enhance portfolio management strategies**.  

Using data for:
- **TSLA** – High-growth, high-risk stock (Automobile Manufacturing)
- **BND** – Bond ETF for stability and income
- **SPY** – Broad U.S. equity market exposure

We leverage both **statistical models** (ARIMA/SARIMA, GARCH) and **deep learning models** (LSTM) to forecast returns and volatility. Forecasts are used to **optimize portfolio allocation**, aiming to **maximize returns while managing risk**.

---

## Objectives
- Fetch and preprocess historical financial data from **Yahoo Finance** (`yfinance`).
- Perform exploratory data analysis (EDA) to identify patterns, correlations, and volatility trends.
- Build forecasting models:
  - ARIMA/SARIMA (classical time series)
  - GARCH (volatility forecasting)
  - LSTM (deep learning for sequence prediction)
- Implement **portfolio optimization** using **PyPortfolioOpt** to generate the **Efficient Frontier**.
- Backtest strategies using historical data to evaluate performance (CAGR, Sharpe ratio, drawdowns).
- Communicate results through **notebooks, reports, and visualizations**.

---

## Project Structure
```bash
finance_forcasting/
│
├── data/ # Raw and cleaned datasets (TSLA, BND, SPY)
├── notebooks/ # Jupyter notebooks for EDA, modeling, and backtesting
│ ├── 01_eda.ipynb
│ ├── 02_arima_model.ipynb
│ ├── 03_lstm_model.ipynb
│ └── 04_portfolio_opt.ipynb
├── scripts/ # Modular Python scripts
│ ├── fetch_data.py
│ ├── preprocess.py
│ ├── train_arima.py
│ ├── train_lstm.py
│ └── backtest.py
├── reports/ # Summary reports, assumptions, findings
├── outputs/plots/ # Saved visualizations (forecasts, efficient frontier)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## Data
Data is sourced via the [`yfinance`](https://pypi.org/project/yfinance/) API for the period **2015-07-01 to 2025-07-31**.

**Fields:**
- `Date` – Trading day timestamp  
- `Open`, `High`, `Low`, `Close`, `Adj Close` – Daily prices (adjusted for dividends/splits)  
- `Volume` – Number of shares/units traded  

---

##  Tech Stack
**Programming & Data:**
- Python 3.x
- `yfinance`, `pandas`, `numpy`
- `matplotlib`, `plotly`, `seaborn`

**Modeling & Forecasting:**
- `statsmodels`, `pmdarima` (ARIMA/SARIMA)
- `arch` (GARCH models)
- `tensorflow` / `keras` (LSTM)
- `scikit-learn` (scalers, metrics)

**Portfolio Optimization & Backtesting:**
- `PyPortfolioOpt`
- Custom walk-forward backtesting scripts

---

## Methodology

1. **Data Acquisition**
   - Download historical OHLCV data for TSLA, BND, SPY using `yfinance`.

2. **Data Preprocessing**
   - Handle missing values & align trading days across assets.
   - Compute log returns, rolling volatility, momentum indicators.

3. **Exploratory Data Analysis**
   - Visualize prices, returns, volatility.
   - Calculate correlations & covariance between assets.

4. **Model Development**
   - **ARIMA/SARIMA** for returns forecasting.
   - **GARCH** for volatility forecasting.
   - **LSTM** for capturing nonlinear patterns.

5. **Portfolio Optimization**
   - Use forecasted returns and covariance matrix to compute the **Efficient Frontier**.
   - Identify portfolios that maximize the Sharpe ratio or meet custom risk constraints.

6. **Backtesting**
   - Simulate portfolio rebalancing using a walk-forward approach.
   - Evaluate with CAGR, volatility, Sharpe/Sortino ratios, drawdowns.

---

## Expected Outcomes
- **Improved portfolio risk-adjusted returns** through data-driven allocation.
- Insights into how predictive modeling can be **an input** (not a standalone decision-maker) in investment strategies.
- A reproducible pipeline for **financial time series analysis** and **portfolio optimization**.

---

## Key Concepts
- **Efficient Market Hypothesis (EMH)**: Pure price prediction is difficult; models here serve as *inputs* to broader decision-making.
- **Stationarity**: Essential for ARIMA-type models; tested using ADF/KPSS.
- **Efficient Frontier**: The set of optimal portfolios balancing risk and return.
- **Backtesting**: Validating strategies on historical data to assess robustness.

---

## How to Run

###  Install Dependencies
```bash
pip install -r requirements.txt
