"""
CLI script for training draft models for specific Magic sets.

Usage:
    python -m app.train_model --set MH3 --epochs 10
"""
import argparse
import os
import glob
import sys

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.DraftData import DraftData
from app.ModelBuilder import ModelBuilder
from app.booster.mtgjson_fetcher import save_booster_data


def find_training_data(set_code: str) -> str:
    """Find the training CSV file for a set in data/{set_code}/"""
    set_dir = f"data/{set_code.upper()}"

    if not os.path.exists(set_dir):
        raise FileNotFoundError(f"Data directory not found: {set_dir}")

    # Look for .csv.gz files
    csv_files = glob.glob(f"{set_dir}/*.csv.gz")

    if not csv_files:
        # Try uncompressed
        csv_files = glob.glob(f"{set_dir}/*.csv")

    if not csv_files:
        raise FileNotFoundError(
            f"No training data found in {set_dir}\n"
            f"Download from https://www.17lands.com/public_datasets"
        )

    return csv_files[0]


def main():
    parser = argparse.ArgumentParser(description="Train a draft model for a Magic set")
    parser.add_argument("--set", required=True, help="Set code (e.g., 'MH3', 'BLB')")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    args = parser.parse_args()
    set_code = args.set.upper()

    print(f"\n=== Training Model for {set_code} ===\n")

    # 1. Find training data
    print("Step 1: Finding training data...")
    csv_path = find_training_data(set_code)
    print(f"✓ Found: {csv_path}")

    # 2. Fetch and cache MTGJson data
    print("\nStep 2: Fetching MTGJson data...")
    output_dir = f"app/models/{set_code}"
    save_booster_data(set_code, output_dir)

    # 3. Load training data
    print("\nStep 3: Loading training data...")
    draft_data = DraftData(csv_path)
    print(f"✓ Loaded {len(draft_data.cards)} cards")

    # 3.5. Save training card list for booster generation
    print("\nStep 3.5: Saving training card list...")
    training_cards = [col[len("pack_card_"):] for col in draft_data.draft_data.columns if col.startswith("pack_card_")]
    training_cards_path = f"{output_dir}/training_cards.json"

    import json
    with open(training_cards_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(training_cards), f, indent=2)
    print(f"✓ Saved {len(training_cards)} card names to {training_cards_path}")

    # 4. Train model
    print(f"\nStep 4: Training model ({args.epochs} epochs)...")
    model_builder = ModelBuilder(draft_data)
    model_builder.train_model(args.epochs)

    # 5. Save model
    model_path = f"{output_dir}/{set_code.lower()}_model.keras"
    model_builder._model.save(model_path)
    print(f"✓ Model saved to {model_path}")

    print(f"\n=== Training Complete! ===")
    print(f"\nModel ready at: {model_path}")
    print(f"Booster data cached in: {output_dir}/")


if __name__ == "__main__":
    main()
