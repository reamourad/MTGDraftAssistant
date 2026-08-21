"""
Pre-encode all cards for each set using CardEncoder.

This script generates data compatible with BOTH TensorFlow (legacy) and PyTorch (future) systems:
- TensorFlow system: Uses training_cards.json and DraftData for integer-based encoding
- PyTorch system: Uses card_encodings.pkl with 407-dimensional feature vectors

Reads CSVs from:      data/{SET}/draft_data_public.{SET}.PremierDraft.csv.gz
Saves outputs to:     app/models/{SET}/
  - booster_config.json  (booster generation rules) - SHARED by both systems
  - sheets.json          (filtered sheets) - SHARED by both systems
  - card_encodings.pkl   (407-dim encodings) - PYTORCH two-tower model
  - training_cards.json  (card list) - TENSORFLOW legacy system

The 407-dimensional encoding format:
  - Rarity: 4 dims (one-hot)
  - Mana cost: 7 dims (6 colors + CMC)
  - Types: 9 dims (one-hot)
  - Power/Toughness: 3 dims (can_attack, power, toughness)
  - Oracle text: 384 dims (sentence embedding)
  Total: 407 dims

This only needs to be run once (or when adding new sets).

Usage:
    python preprocess_cards.py              # Process all sets in data/
    python preprocess_cards.py MH3          # Process only MH3
    python preprocess_cards.py MH3 BLB EOE  # Process multiple sets
"""

import os
import pickle
import pandas as pd
import numpy as np
import argparse
import json
from app.ml.experimental.card_encoder import CardEncoder
from app.booster import mtgjson_fetcher


def extract_unique_cards_from_csv(csv_path):
    """Extract unique card names from CSV columns (pack_card_* and pool_*)."""
    print(f"Loading CSV: {csv_path}")

    # Load just the column names (no data needed)
    df = pd.read_csv(csv_path, nrows=0)

    card_names = set()
    for col in df.columns:
        if col.startswith('pack_card_') or col.startswith('pool_'):
            card_name = col.replace('pack_card_', '').replace('pool_', '')
            card_names.add(card_name)

    print(f"Found {len(card_names)} unique cards in CSV")
    return card_names


def fetch_and_prepare_card_data(set_code, card_names_filter):
    """Fetch MTGJson data and filter to cards in training data."""
    print(f"\nFetching MTGJson data for {set_code}...")

    # Use mtgjson_fetcher helper functions
    set_data = mtgjson_fetcher.fetch_set_data(set_code)
    all_cards = mtgjson_fetcher.extract_cards(set_data)

    # Filter to only cards in training data
    card_name_to_data = {card['name']: card for card in all_cards}

    filtered_cards = []
    missing_cards = []

    for card_name in card_names_filter:
        if card_name in card_name_to_data:
            filtered_cards.append(card_name_to_data[card_name])
        else:
            missing_cards.append(card_name)

    if missing_cards:
        print(f"WARNING: {len(missing_cards)} cards from CSV not found in MTGJson:")
        for card in missing_cards:  # Show first 10
            print(f"  - {card}")

    print(f"Prepared {len(filtered_cards)} cards for encoding")
    return filtered_cards


def encode_cards(cards):
    """Encode all cards using CardEncoder (PyTorch-compatible 407-dim vectors)."""
    print("\nInitializing CardEncoder...")
    encoder = CardEncoder(card_list=cards)

    print(f"Encoding {len(cards)} cards...")
    encodings = {}

    for i, card in enumerate(cards):
        if i % 50 == 0:
            print(f"  Encoded {i}/{len(cards)} cards...")

        # Use UUID as key instead of name for uniqueness
        card_uuid = card['uuid']
        card_name = card['name']
        encoding = encoder.encode(card)  # Returns 407-dim np.array
        
        # Store with both UUID and name for compatibility
        encodings[card_uuid] = {
            'encoding': encoding,
            'name': card_name,
            'uuid': card_uuid
        }

    print(f"Finished encoding {len(encodings)} cards")
    return encodings


def save_training_cards_list(card_names, output_dir):
    """
    Save training_cards.json for backward compatibility with TensorFlow system.
    
    The TensorFlow system (app/DraftData.py, app/ModelBuilder.py) expects a simple
    list of card names in training_cards.json. This maintains compatibility during
    the migration to PyTorch.
    """
    import json
    
    training_cards_path = os.path.join(output_dir, 'training_cards.json')
    card_list = sorted(list(card_names))
    
    print(f"\nSaving training_cards.json for TensorFlow compatibility...")
    with open(training_cards_path, 'w', encoding='utf-8') as f:
        json.dump(card_list, f, indent=2)
    
    print(f"Saved {len(card_list)} card names to {training_cards_path}")
    return training_cards_path


def preprocess_set(set_code, data_dir='data', models_dir='app/models'):
    """Pre-encode all cards for a single set and save booster data."""
    print(f"\n{'='*60}")
    print(f"PREPROCESSING SET: {set_code}")
    print(f"{'='*60}")

    # CSV is in data/{SET}/
    data_set_dir = os.path.join(data_dir, set_code)

    # Output goes to models/{SET}/
    models_set_dir = os.path.join(models_dir, set_code)
    os.makedirs(models_set_dir, exist_ok=True)

    # Find CSV file
    csv_files = [f for f in os.listdir(data_set_dir) if f.endswith('.csv.gz') or f.endswith('.csv')]
    if not csv_files:
        print(f"ERROR: No CSV file found in {data_set_dir}")
        return False

    csv_path = os.path.join(data_set_dir, csv_files[0])

    # Step 1: Extract unique cards from CSV
    print("\n[Step 1/4] Extracting card names from CSV...")
    card_names = extract_unique_cards_from_csv(csv_path)

    # Step 2: Fetch all card data using mtgjson_fetcher helper
    # This handles: main set, companion sets (Commander), and SPG cards
    print("\n[Step 2/4] Fetching card data from MTGJson...")
    print("  - Fetching main set data")
    print("  - Checking for companion sets (Commander decks)")
    print("  - Checking for SPG cards in boosters")
    
    all_cards, booster_config = mtgjson_fetcher.fetch_all_card_data(
        set_code=set_code,
        card_names_filter=card_names
    )
    
    print(f"[OK] Total cards loaded: {len(all_cards)}")

    # Step 3: Save booster data (config and sheets)
    print("\n[Step 3/4] Saving booster data...")
    mtgjson_fetcher.save_booster_files(
        set_code=set_code,
        output_dir=models_set_dir,
        all_cards=all_cards,
        card_names_filter=card_names,
        booster_config=booster_config
    )

    # Step 4: Encode cards and save
    print("\n[Step 4/4] Encoding cards...")
    
    # Build lookup dictionaries
    card_name_to_data = {card['name']: card for card in all_cards}
    
    filtered_cards = []
    missing_cards = []
    card_name_to_uuid = {}  # Track name -> UUID mapping

    # Match cards from CSV names to MTGJson data
    for name in card_names:
        card = None
        
        # Try exact name match first
        if name in card_name_to_data:
            card = card_name_to_data[name]
        # Try case-insensitive match as fallback
        else:
            for card_name, card_data in card_name_to_data.items():
                if card_name.lower() == name.lower():
                    card = card_data
                    print(f"[INFO] Found '{name}' via case-insensitive match: '{card_name}'")
                    break
        
        if card:
            filtered_cards.append(card)
            card_name_to_uuid[name] = card['uuid']  # Store mapping
        else:
            missing_cards.append(name)

    if missing_cards:
        print(f"\nWARNING: {len(missing_cards)} cards could not be found:")
        for card in missing_cards:  # Show first 10
            print(f"  - {card}")
        print(f"\nThese cards will be skipped during training.")
        print(f"This may happen if:")
        print(f"  - Card names differ between 17Lands and MTGJson")
        print(f"  - Cards are from a different set or special edition")
        print(f"  - Cards are variants not in the main set data")

    # Encode cards (now uses UUIDs as keys)
    encodings = encode_cards(filtered_cards)

    # Save encodings with UUID keys
    output_path = os.path.join(models_set_dir, 'card_encodings.pkl')
    print(f"\nSaving PyTorch-compatible encodings to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(encodings, f)

    # Save name->UUID mapping for lookup during training
    mapping_path = os.path.join(models_set_dir, 'card_name_to_uuid.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(card_name_to_uuid, f, indent=2)
    print(f"Saved card name->UUID mapping to {mapping_path}")

    # Verify file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved successfully! File size: {file_size_mb:.2f} MB")
    print(f"Encoding format: UUID -> {{encoding: 407-dim array, name: str, uuid: str}}")

    # Save training_cards.json for TensorFlow backward compatibility
    print("\nSaving training_cards.json for TensorFlow compatibility...")
    save_training_cards_list(card_names, models_set_dir)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {set_code}")
    print(f"  - booster_config.json: saved (SHARED)")
    print(f"  - sheets.json: saved (SHARED)")
    print(f"  - card_encodings.pkl: {len(encodings)} cards @ 407-dim (PYTORCH, UUID-keyed)")
    print(f"  - card_name_to_uuid.json: {len(card_name_to_uuid)} mappings")
    print(f"  - training_cards.json: {len(card_names)} card names (TENSORFLOW)")
    print(f"{'='*60}")

    return True


def preprocess_all_sets(sets=None, data_dir='data', models_dir='app/models'):
    """Pre-encode cards for all sets."""
    if sets is None:
        # Auto-detect sets from data directory
        sets = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

    print(f"Will preprocess {len(sets)} sets: {', '.join(sets)}")

    results = {}
    for set_code in sets:
        try:
            success = preprocess_set(set_code, data_dir, models_dir)
            results[set_code] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"\nERROR processing {set_code}: {e}")
            import traceback
            traceback.print_exc()
            results[set_code] = "ERROR"

    # Print summary
    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    for set_code, status in results.items():
        status_icon = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_icon} {set_code}: {status}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-encode cards for MTG draft sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_cards.py              # Process all sets in data/
  python preprocess_cards.py MH3          # Process only MH3
  python preprocess_cards.py MH3 BLB EOE  # Process multiple specific sets
        """
    )

    parser.add_argument(
        'sets',
        nargs='*',  # 0 or more arguments
        help='Set codes to process (e.g., MH3 BLB). If not specified, processes all sets.'
    )

    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing set folders (default: data)'
    )

    parser.add_argument(
        '--models-dir',
        default='app/models',
        help='Directory to save outputs (default: app/models)'
    )

    args = parser.parse_args()

    # If no sets specified, process all
    sets_to_process = args.sets if args.sets else None

    preprocess_all_sets(sets=sets_to_process, data_dir=args.data_dir, models_dir=args.models_dir)