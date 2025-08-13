import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller

cleaned_dir = "data/cleaned"
reports_dir = "reports"

os.makedirs(reports_dir, exist_ok=True)

for file in os.listdir(cleaned_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(cleaned_dir, file)

        # Read CSV
        df = pd.read_csv(file_path)

        # Parse Date column safely
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df.set_index("Date", inplace=True)

        # Ensure Close is numeric
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)

        # Calculate daily returns
        df["Daily_Return"] = df["Close"].pct_change()
        df.dropna(subset=["Daily_Return"], inplace=True)

        # Augmented Dickey-Fuller test for Close
        adf_close = adfuller(df["Close"])
        adf_close_result = {
            "ADF Statistic": adf_close[0],
            "p-value": adf_close[1],
            "Critical Values": adf_close[4],
            "Stationary": adf_close[1] <= 0.05
        }

        # Augmented Dickey-Fuller test for Daily Returns
        adf_returns = adfuller(df["Daily_Return"])
        adf_returns_result = {
            "ADF Statistic": adf_returns[0],
            "p-value": adf_returns[1],
            "Critical Values": adf_returns[4],
            "Stationary": adf_returns[1] <= 0.05
        }

        # Save report
        report_text = f"""
File: {file}

Augmented Dickey-Fuller Test on Closing Prices:
ADF Statistic: {adf_close_result['ADF Statistic']}
p-value: {adf_close_result['p-value']}
Critical Values: {adf_close_result['Critical Values']}
Stationary: {adf_close_result['Stationary']}

Augmented Dickey-Fuller Test on Daily Returns:
ADF Statistic: {adf_returns_result['ADF Statistic']}
p-value: {adf_returns_result['p-value']}
Critical Values: {adf_returns_result['Critical Values']}
Stationary: {adf_returns_result['Stationary']}
"""

        report_path = os.path.join(reports_dir, f"{file.replace('.csv', '')}_stationarity_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"Processed: {file} → Report saved to {report_path}")
