"""
Pre-encode all cards for each set using CardEncoder.

Reads CSVs from:      data/{SET}/draft_data_public.{SET}.PremierDraft.csv.gz
Saves outputs to:     app/models/{SET}/
  - cards.json           (MTGJson card data)
  - booster_config.json  (booster generation rules)
  - sheets.json          (filtered sheets)
  - card_encodings.pkl   (407-dim encodings for training)

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
from app.CardEncoder import CardEncoder
from app.booster.mtgjson_fetcher import (
    fetch_set_data,
    extract_cards,
    extract_booster_config,
    fetch_spg_set,
    extract_uuids_from_booster_config
)


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

    set_data = fetch_set_data(set_code) #fetches all the data from mtjjson
    all_cards = extract_cards(set_data) #fetches all the cards from mtgjson

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
    """Encode all cards using CardEncoder."""
    print("\nInitializing CardEncoder...")
    encoder = CardEncoder(card_list=cards)

    print(f"Encoding {len(cards)} cards...")
    encodings = {}

    for i, card in enumerate(cards):
        if i % 50 == 0:
            print(f"  Encoded {i}/{len(cards)} cards...")

        card_name = card['name']
        encoding = encoder.encode(card)  # Returns 407-dim np.array
        encodings[card_name] = encoding

    print(f"Finished encoding {len(encodings)} cards")
    return encodings


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
    print("\n[Step 1/6] Extracting card names from CSV...")
    card_names = extract_unique_cards_from_csv(csv_path)

    # Step 2: Fetch MTGJson data
    print("\n[Step 2/6] Fetching MTGJson data...")
    set_data = fetch_set_data(set_code)
    all_cards = extract_cards(set_data)
    booster_config = extract_booster_config(set_data)

    # Step 2a: Fetch and add SPG cards if present in booster config
    print("\n[Step 2a/6] Checking for SPG cards...")
    booster_uuids = extract_uuids_from_booster_config(booster_config)
    print(f"[INFO] Found {len(booster_uuids)} UUIDs in booster configuration")

    spg_data = fetch_spg_set()
    spg_all_cards = spg_data.get('data', {}).get('cards', [])
    spg_uuids = {card['uuid'] for card in spg_all_cards}
    print(f"[INFO] SPG set contains {len(spg_uuids)} cards")

    spg_uuids_in_boosters = booster_uuids.intersection(spg_uuids)

    if spg_uuids_in_boosters:
        print(f"[INFO] Found {len(spg_uuids_in_boosters)} SPG cards in booster configuration")
        spg_cards = extract_cards(spg_data, spg_uuids_in_boosters)
        all_cards.extend(spg_cards)
        print(f"[OK] Added {len(spg_cards)} SPG cards:")
        for spg_card in spg_cards:
            print(f"  - {spg_card['name']}")
    else:
        print(f"[INFO] No SPG cards found in booster configuration")

    # Step 3: Save cards.json
    print("\n[Step 3/6] Saving cards.json...")
    cards_path = os.path.join(models_set_dir, 'cards.json')
    with open(cards_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(all_cards, f, indent=2)
    print(f"Saved {len(all_cards)} cards to {cards_path}")

    # Step 4: Save booster_config.json
    print("\n[Step 4/6] Saving booster_config.json...")
    config_path = os.path.join(models_set_dir, 'booster_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(booster_config, f, indent=2)
    print(f"Saved booster config to {config_path}")

    # Step 5: Build and save sheets.json (filtered by training cards)
    print("\n[Step 5/6] Building filtered sheets...")
    try:
        from app.booster.mtgjson_fetcher import build_filtered_sheets
        sheets = build_filtered_sheets(all_cards, card_names, booster_config)
        sheets_path = os.path.join(models_set_dir, 'sheets.json')
        with open(sheets_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(sheets, f, indent=2)
        print(f"Saved {len(sheets)} sheets to {sheets_path}")
    except Exception as e:
        print(f"WARNING: Could not build sheets: {e}")

    # Step 6: Encode cards and save encodings
    print("\n[Step 6/6] Encoding cards...")
    # Filter to only cards in training data
    card_name_to_data = {card['name']: card for card in all_cards}
    filtered_cards = []
    missing_cards = []

    # Add cards from the set (including SPG cards if present)
    for name in card_names:
        if name in card_name_to_data:
            filtered_cards.append(card_name_to_data[name])
        else:
            missing_cards.append(name)

    if missing_cards:
        print(f"\nWARNING: {len(missing_cards)} cards could not be found:")
        for card in missing_cards[:10]:  # Show first 10
            print(f"  - {card}")
        if len(missing_cards) > 10:
            print(f"  ... and {len(missing_cards) - 10} more")

    encodings = encode_cards(filtered_cards)

    output_path = os.path.join(models_set_dir, 'card_encodings.pkl')
    print(f"\nSaving encodings to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(encodings, f)

    # Verify file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved successfully! File size: {file_size_mb:.2f} MB")
    print(f"Average encoding shape: {list(encodings.values())[0].shape}")

    print(f"\n{'='*60}")
    print(f"COMPLETED: {set_code}")
    print(f"  - cards.json: {len(all_cards)} cards")
    print(f"  - booster_config.json: saved")
    print(f"  - sheets.json: saved")
    print(f"  - card_encodings.pkl: {len(encodings)} encoded cards")
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