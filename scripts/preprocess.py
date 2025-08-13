import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Get root directory (parent of scripts/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
CLEANED_DATA_DIR = os.path.join(ROOT_DIR, "data", "cleaned")

os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

def preprocess_file(file_path, save_clean=True):
    print(f"📂 Reading file: {file_path}")
    df = pd.read_csv(file_path)

    # Ensure appropriate data types
    df = df.convert_dtypes()

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("Unknown")

    # Example scaling (remove if not needed)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save cleaned file
    if save_clean:
        output_path = os.path.join(CLEANED_DATA_DIR, os.path.basename(file_path))
        df.to_csv(output_path, index=False)
        print(f"✅ Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    print(f"🔍 RAW data dir: {RAW_DATA_DIR}")
    print(f"💾 CLEANED data dir: {CLEANED_DATA_DIR}")

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".csv"):
            preprocess_file(os.path.join(RAW_DATA_DIR, filename))

    print("\n📄 Files now in CLEANED directory:")
    print(os.listdir(CLEANED_DATA_DIR))
