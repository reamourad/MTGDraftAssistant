import pandas as pd
import gzip
import os

# Path to your raw 17Lands data file (.csv.gz)
RAW_DATA_PATH = "app/data/draft_data_public.MH3.PremierDraft.csv.gz"

# Where to save the cleaned file
CLEAN_DATA_PATH = "app/data/MH3_clean.csv"

# How many rows to keep (set None to keep all)
MAX_ROWS = 100_000

def main():
    # if not os.path.exists(RAW_DATA_PATH):
    #     raise FileNotFoundError(f"Cannot find file: {RAW_DATA_PATH}")

    # print(f"Loading dataset from {RAW_DATA_PATH} ...")
    # with gzip.open(RAW_DATA_PATH, "rt", encoding="utf-8") as f:
    #     df = pd.read_csv(f, nrows=MAX_ROWS, low_memory=False)

    # print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns")

    # if MAX_ROWS is not None and len(df) > MAX_ROWS:
    #     df = df.head(MAX_ROWS)
    #     print(f"Trimmed to first {MAX_ROWS:,} rows for preview/training speed")

    # os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    # df.to_csv(CLEAN_DATA_PATH, index=False)
    # print(f"Saved cleaned dataset → {CLEAN_DATA_PATH}")
    # print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    df = pd.read_csv(CLEAN_DATA_PATH, nrows=50_000)
    df.to_csv(CLEAN_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
