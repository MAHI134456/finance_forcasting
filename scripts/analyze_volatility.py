# scripts/analyze_volatility.py
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Config
# -------------------------
ROLL_WINDOW = 20                   # rolling window (trading days ~1 month)
TRADING_DAYS = 252
RF_ANNUAL = 0.02                   # annual risk-free rate (2%), change if needed
VAR_LEVELS = [0.95, 0.99]          # historical VaR levels

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse dates robustly
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Choose price column: Adj Close preferred if present
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"{csv_path.name}: neither 'Close' nor 'Adj Close' found")

    # Ensure numeric price
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    # Daily returns (decimal form, not %)
    df["Return"] = df[price_col].pct_change()
    df = df.dropna(subset=["Return"]).copy()

    df["Price"] = df[price_col]
    return df

def cagr(df: pd.DataFrame) -> float:
    # CAGR from first to last price
    start, end = df["Price"].iloc[0], df["Price"].iloc[-1]
    years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start)**(1/years) - 1

def annualized_return_sigma(df: pd.DataFrame):
    mu_d = df["Return"].mean()
    sigma_d = df["Return"].std(ddof=1)
    mu_a = mu_d * TRADING_DAYS
    sigma_a = sigma_d * math.sqrt(TRADING_DAYS)
    return mu_d, sigma_d, mu_a, sigma_a

def sharpe_ratio(mu_d, sigma_d, rf_annual=RF_ANNUAL):
    if sigma_d == 0 or np.isnan(sigma_d):
        return np.nan
    rf_d = rf_annual / TRADING_DAYS
    sr = (mu_d - rf_d) / sigma_d * math.sqrt(TRADING_DAYS)
    return sr

def max_drawdown(df: pd.DataFrame) -> float:
    # Compute drawdown from price
    running_max = df["Price"].cummax()
    drawdown = df["Price"] / running_max - 1.0
    return drawdown.min()  # most negative (e.g., -0.65 = -65%)

def hist_var(returns: pd.Series, level: float) -> float:
    # Historical VaR (right-tail losses are negative)
    # VaR at 95% ~ 5th percentile of returns
    tail = (1 - level) * 100
    return np.nanpercentile(returns, tail)

def write_report(name: str, df: pd.DataFrame, out_path: Path,
                 cagr_val, mu_d, sigma_d, mu_a, sigma_a, sr,
                 var95, var99, mdd, top_gains, top_losses):
    direction = "uptrend" if cagr_val > 0 else "downtrend" if cagr_val < 0 else "flat"

    focus_note = ""
    if "TSLA" in name.upper():
        focus_note = (
            "\n**Tesla focus:** Given TSLA’s high-growth profile, elevated volatility is expected. "
            "Rolling standard deviation spikes often align with major news or earnings; use volatility-aware position sizing "
            "and consider scenario analysis around catalysts.\n"
        )

    md = f"""# Volatility & Risk Report — {name}

**Period:** {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}  
**Observations:** {len(df):,} trading days

## Price Direction
- **CAGR:** {cagr_val:.2%}  ({direction})
- **Total Price Change:** {(df['Price'].iloc[-1] / df['Price'].iloc[0] - 1):.2%}

## Return & Volatility (Daily / Annualized)
- **Mean Daily Return:** {mu_d:.4%}
- **Daily Volatility (σ):** {sigma_d:.4%}
- **Annualized Return:** {mu_a:.2%}
- **Annualized Volatility:** {sigma_a:.2%}
- **Sharpe Ratio (annualized, rf={RF_ANNUAL:.2%}):** {sr:.2f}

## Risk — Historical VaR (based on daily returns)
- **VaR 95% (≈ 5th percentile):** {var95:.2%} (daily)
- **VaR 99% (≈ 1st percentile):** {var99:.2%} (daily)

Interpretation: A **VaR 95% of {var95:.2%}** means that on 95% of days, losses are not expected to exceed this magnitude. On ~5% of days, losses may be worse.

## Max Drawdown
- **Max Drawdown:** {mdd:.2%}

## Notable Daily Moves
**Top 5 gains**  
{top_gains.to_string(index=False)}

**Top 5 losses**  
{top_losses.to_string(index=False)}
{focus_note}

## Rolling Behavior (Short-Term Trends)
- We computed a **{ROLL_WINDOW}-day rolling mean** of price and **{ROLL_WINDOW}-day rolling standard deviation** of returns.
- Rising rolling mean with falling rolling std usually indicates **steady uptrends**; spikes in rolling std indicate **volatility shocks**.

---

**Notes:**  
- VaR here is *historical* (non-parametric) and does not assume normality.  
- Sharpe uses a constant annual risk-free rate of {RF_ANNUAL:.2%}; adjust if needed.  
- Past performance does not guarantee future results.
"""
    out_path.write_text(md, encoding="utf-8")

def make_plots(name: str, df: pd.DataFrame):
    # Rolling metrics
    df["RollingMeanPrice"] = df["Price"].rolling(ROLL_WINDOW).mean()
    df["RollingVol"] = df["Return"].rolling(ROLL_WINDOW).std()

    # 1) Price + Rolling Mean
    plt.figure(figsize=(11, 6))
    plt.plot(df["Date"], df["Price"], label="Price")
    plt.plot(df["Date"], df["RollingMeanPrice"], label=f"{ROLL_WINDOW}D Rolling Mean")
    plt.title(f"{name} — Price with {ROLL_WINDOW}D Rolling Mean")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True); plt.legend()
    p1 = PLOTS_DIR / f"{name}_price_rolling_mean.png"
    plt.tight_layout(); plt.savefig(p1); plt.close()

    # 2) Rolling Volatility (std of returns)
    plt.figure(figsize=(11, 5))
    plt.plot(df["Date"], df["RollingVol"])
    plt.title(f"{name} — {ROLL_WINDOW}D Rolling Volatility (Std of Returns)")
    plt.xlabel("Date"); plt.ylabel("Rolling Std (daily)"); plt.grid(True)
    p2 = PLOTS_DIR / f"{name}_rolling_volatility.png"
    plt.tight_layout(); plt.savefig(p2); plt.close()

    # 3) Returns histogram with VaR lines
    plt.figure(figsize=(10, 5))
    returns = df["Return"].dropna()
    plt.hist(returns, bins=60, alpha=0.7)
    v95 = hist_var(returns, 0.95)
    v99 = hist_var(returns, 0.99)
    plt.axvline(v95, linestyle="--", label=f"VaR 95%: {v95:.2%}")
    plt.axvline(v99, linestyle="--", label=f"VaR 99%: {v99:.2%}")
    plt.title(f"{name} — Daily Returns Histogram with VaR")
    plt.xlabel("Daily Return"); plt.ylabel("Frequency"); plt.legend(); plt.grid(True)
    p3 = PLOTS_DIR / f"{name}_returns_hist_var.png"
    plt.tight_layout(); plt.savefig(p3); plt.close()

    return [p1, p2, p3]

def process_file(csv_path: Path):
    name = csv_path.stem  # e.g., TSLA, SPY, BND if files are named that way
    df = load_and_prepare(csv_path)

    # Core stats
    cagr_val = cagr(df)
    mu_d, sigma_d, mu_a, sigma_a = annualized_return_sigma(df)
    sr = sharpe_ratio(mu_d, sigma_d)

    # VaR (historical)
    var95 = hist_var(df["Return"], 0.95)
    var99 = hist_var(df["Return"], 0.99)

    # Drawdown
    mdd = max_drawdown(df)

    # Notable days
    top_gains = df.nlargest(5, "Return")[["Date", "Return"]].copy()
    top_losses = df.nsmallest(5, "Return")[["Date", "Return"]].copy()
    top_gains["Date"] = top_gains["Date"].dt.date
    top_losses["Date"] = top_losses["Date"].dt.date
    top_gains["Return"] = top_gains["Return"].map(lambda x: f"{x:.2%}")
    top_losses["Return"] = top_losses["Return"].map(lambda x: f"{x:.2%}")

    # Plots
    saved = make_plots(name, df)

    # Report
    report_path = REPORTS_DIR / f"{name}_volatility_report.md"
    write_report(
        name=name, df=df, out_path=report_path,
        cagr_val=cagr_val, mu_d=mu_d, sigma_d=sigma_d, mu_a=mu_a, sigma_a=sigma_a,
        sr=sr, var95=var95, var99=var99, mdd=mdd,
        top_gains=top_gains, top_losses=top_losses
    )

    print(f"✅ {name}: saved plots -> {[str(p) for p in saved]}")
    print(f"📝 {name}: report  -> {report_path}")

if __name__ == "__main__":
    print(f"Reading cleaned CSVs from: {CLEANED_DIR}")
    files = [p for p in CLEANED_DIR.glob("*.csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {CLEANED_DIR}")

    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"❌ Skipping {f.name}: {e}")
