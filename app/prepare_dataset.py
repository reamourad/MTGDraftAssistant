import pandas as pd
import gzip
import os

# ============================================================
# CONFIGURATION
# ============================================================
# Path to your raw 17Lands data file (.csv.gz)
RAW_DATA_PATH = "app/data/draft_data_public.MH3.PremierDraft.csv.gz"

# Where to save the cleaned file (useful for training your model)
CLEAN_DATA_PATH = "app/data/MH3_clean.csv"

# How many rows to keep (set None to keep all)
MAX_ROWS = 100_000

# ============================================================
# LOAD AND FILTER
# ============================================================
def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Cannot find file: {RAW_DATA_PATH}")

    print(f"🔍 Loading dataset from {RAW_DATA_PATH} ...")
    with gzip.open(RAW_DATA_PATH, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, low_memory=False)

    print(f"✅ Loaded {len(df):,} rows and {len(df.columns):,} columns")

    # ========================================================
    # KEEP ONLY USEFUL COLUMNS
    # ========================================================
    base_columns = [
        "pick",
        "pack_number",
        "pick_number",
        "pick_win_rate",
        "game_in_hand_win_rate",
        "user_game_win_rate_bucket",
    ]

    # Add all pack_ and pool_ columns (one per card)
    card_columns = [c for c in df.columns if c.startswith(("pack_card_", "pool_"))]
    keep_columns = [c for c in base_columns + card_columns if c in df.columns]

    # Filter down
    df = df[keep_columns].dropna(subset=["pick"]).reset_index(drop=True)

    print(f"⚙️ Filtered down to {len(df.columns):,} columns")

    # ========================================================
    # OPTIONAL: LIMIT ROWS FOR SPEED
    # ========================================================
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
        print(f"✂️ Trimmed to first {MAX_ROWS:,} rows for preview/training speed")

    # ========================================================
    # SAVE CLEANED VERSION
    # ========================================================
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"💾 Saved cleaned dataset → {CLEAN_DATA_PATH}")
    print(f"✅ Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")


if __name__ == "__main__":
    main()
