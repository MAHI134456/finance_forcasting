import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yfinance as yf  # Ensure yfinance is installed: pip install yfinance

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "cleaned"
OUTPUT_DIR = ROOT_DIR / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Function to process a single CSV file
def visualize_file(file_path):
    file_name = file_path.stem  # Get file name without extension
    print(f"\nProcessing file: {file_path}")

    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # Check for required columns
    if "Date" not in df.columns or "Close" not in df.columns:
        print(f"Error: {file_path} must contain 'Date' and 'Close' columns.")
        return

    # Ensure correct datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Convert Close to numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # 1. Closing price over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Close")
    plt.title(f"Closing Price Over Time ({file_name})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"closing_price_over_time_{file_name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")

    # 2. Daily percentage change
    df["Pct_Change"] = df["Close"].pct_change() * 100
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Pct_Change")
    plt.title(f"Daily Percentage Change ({file_name})")
    plt.xlabel("Date")
    plt.ylabel("Percentage Change (%)")
    plt.grid(True)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"daily_pct_change_{file_name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")

    # 3. Rolling mean & std for volatility
    df["Rolling_Mean"] = df["Close"].rolling(window=30).mean()
    df["Rolling_STD"] = df["Close"].rolling(window=30).std()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Close", label="Close")
    sns.lineplot(data=df, x="Date", y="Rolling_Mean", label="30-Day Rolling Mean")
    plt.fill_between(df["Date"], 
                     df["Rolling_Mean"] - df["Rolling_STD"], 
                     df["Rolling_Mean"] + df["Rolling_STD"], 
                     color="gray", alpha=0.3, label="Rolling Std Dev")
    plt.title(f"Volatility Analysis ({file_name})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"volatility_analysis_{file_name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")

    # 4. Outlier detection in returns
    threshold = 3
    mean = df["Pct_Change"].mean()
    std = df["Pct_Change"].std()
    df["Outlier"] = abs(df["Pct_Change"] - mean) > threshold * std
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="Date", y="Pct_Change", hue="Outlier", palette={True: "red", False: "blue"}, legend=False)
    plt.title(f"Outlier Detection in Daily Returns ({file_name})")
    plt.xlabel("Date")
    plt.ylabel("Percentage Change (%)")
    plt.grid(True)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"outlier_detection_{file_name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")

    # 5. Days with unusually high or low returns
    unusual_days = df[df["Outlier"]]
    output_path = OUTPUT_DIR / f"unusual_days_{file_name}.csv"
    unusual_days.to_csv(output_path, index=False)
    print(f"✅ Saved unusual days to: {output_path}")

# Find all CSV files in data/cleaned
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in {DATA_DIR}. Fetching Brent oil prices as an example...")
    brent = yf.download("BZ=F", start="2000-01-01")
    brent = brent[["Close"]].reset_index()
    DATA_PATH = DATA_DIR / "brent_oil_prices.csv"
    brent.to_csv(DATA_PATH, index=False)
    print(f"✅ Saved fetched data to: {DATA_PATH}")
    csv_files = [DATA_PATH]

# Process each CSV file
print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
for csv_file in csv_files:
    visualize_file(csv_file)