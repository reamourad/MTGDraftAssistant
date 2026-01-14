"""
Fetch and cache MTGJson data during training.
"""
import requests
import json
import os
from typing import Dict, List, Any, Set, Union


MTGJSON_BASE_URL = "https://mtgjson.com/api/v5"


def fetch_set_data(set_code: str) -> Dict[str, Any]:
    """
    Fetch complete set data from MTGJson.

    Args:
        set_code: Three or four letter set code (e.g., 'MH3', 'BLB')

    Returns:
        Dictionary containing full set data

    Raises:
        requests.HTTPError: If the API request fails
    """
    url = f"{MTGJSON_BASE_URL}/{set_code.upper()}.json"
    print(f"Fetching set data from {url}...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    print(f"[OK] Fetched data for {set_code}")
    return data

def fetch_spg_set():
    url = f"{MTGJSON_BASE_URL}/SPG.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    print(f"[OK] Fetched data for SPG")
    return data


def extract_cards(set_data: Dict[str, Any], uuids=None) -> List[Dict[str, Any]]:
    """
    Extract card list from set data.

    For double-faced cards, only keeps the front face.
    Optionally combines oracle text from both faces.
    """
    cards = set_data.get('data', {}).get('cards', [])

    # Group cards by name to detect DFCs
    cards_by_name = {}

    for card in cards:
        name = card.get('name')
        uuid = card.get('uuid')

        if uuids is not None:
            if uuid not in uuids:
                continue

        # Skip if we've already processed this card name
        if name in cards_by_name:
            # This is a back face - we'll handle it below
            continue

        cards_by_name[name] = card

    # Now process cards, handling DFCs
    simplified_cards = []
    processed_names = set()

    for card in cards:
        name = card.get('name')

        if "//" in name:
            name = name.split(" // ")[0].strip()

        # Skip if already processed
        if name in processed_names:
            continue

        # Check if this is a DFC
        side = card.get('side')

        if side == 'b':
            # This is a back face, skip it
            # (We'll get the front face separately)
            continue

        # Extract oracle text
        oracle_text = card.get('text', '')

        # If this is a DFC (has a side), find the back face
        if side == 'a':
            # Find the back face
            back_face = None
            for other_card in cards:
                if (other_card.get('name') == name and
                    other_card.get('side') == 'b'):
                    back_face = other_card
                    break

            if back_face:
                # Combine oracle text from both faces
                back_text = back_face.get('text', '')
                oracle_text = f"{oracle_text} // {back_text}"

                print(f"[DFC] {name} (combined front + back text)")

        # Create simplified card
        simplified_cards.append({
            'name': name,
            'uuid': card.get('uuid'),
            'rarity': card.get('rarity', '').lower(),
            'colors': card.get('colors', []),
            'types': card.get('types', []),
            'manaCost': card.get('manaCost', ''),
            'text': oracle_text,  # Combined text for DFCs
            'power': card.get('power'),
            'toughness': card.get('toughness'),
            'keywords': card.get('keywords', []),
        })

        processed_names.add(name)

    print(f"[OK] Extracted {len(simplified_cards)} cards (DFCs merged)")
    return simplified_cards


def extract_booster_config(set_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract booster configuration from set data.

    Args:
        set_data: Full set data from MTGJson

    Returns:
        Booster configuration dictionary
    """
    booster_config = set_data.get('data', {}).get('booster', {})

    if not booster_config:
        print("[WARN] No booster configuration found, will use fallback rules")
        return {}

    print(f"[OK] Extracted booster configuration")
    return booster_config


def extract_uuids_from_booster_config(booster_config: Dict[str, Any]) -> Set[str]:
    """
    Extract all card UUIDs referenced in the booster configuration.

    Args:
        booster_config: Booster configuration dictionary

    Returns:
        Set of all UUIDs found in booster sheets
    """
    uuids = set()

    if not booster_config or 'play' not in booster_config:
        return uuids

    play_config = booster_config['play']
    sheets = play_config.get('sheets', {})

    for sheet_name, sheet_data in sheets.items():
        sheet_cards = sheet_data.get('cards', {})
        uuids.update(sheet_cards.keys())

    return uuids


def save_booster_data(set_code: str, output_dir: str):
    """
    Fetch MTGJson data and save to model directory.

    Called during training to cache booster data locally.

    Args:
        set_code: Set code (e.g., 'MH3')
        output_dir: Directory to save files (e.g., 'app/models/MH3/')

    Creates:
        - output_dir/booster_config.json
        - output_dir/cards.json
    """
    print(f"\n=== Fetching MTGJson data for {set_code} ===")

    # Fetch set data
    set_data = fetch_set_data(set_code)

    # Extract cards and booster config
    cards = extract_cards(set_data)
    booster_config = extract_booster_config(set_data)

    # Extract all UUIDs from booster config
    booster_uuids = extract_uuids_from_booster_config(booster_config)
    print(f"[INFO] Found {len(booster_uuids)} UUIDs in booster configuration")

    # Fetch SPG set data
    spg_data = fetch_spg_set()
    spg_all_cards = spg_data.get('data', {}).get('cards', [])
    spg_uuids = {card['uuid'] for card in spg_all_cards}
    print(f"[INFO] SPG set contains {len(spg_uuids)} cards")

    # Find SPG UUIDs that appear in this set's booster config
    spg_uuids_in_boosters = booster_uuids.intersection(spg_uuids)

    if spg_uuids_in_boosters:
        print(f"[INFO] Found {len(spg_uuids_in_boosters)} SPG cards in booster configuration")
        # Extract only the SPG cards that appear in boosters
        spg_cards = extract_cards(spg_data, spg_uuids_in_boosters)
        # Combine main set cards with SPG cards
        cards.extend(spg_cards)
        print(f"[OK] Added {len(spg_cards)} SPG cards to card list:")
        for spg_card in spg_cards:
            print(f"  - {spg_card['name']}")
    else:
        print(f"[INFO] No SPG cards found in booster configuration")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save cards.json
    cards_path = os.path.join(output_dir, 'cards.json')
    with open(cards_path, 'w', encoding='utf-8') as f:
        json.dump(cards, f, indent=2)
    print(f"[OK] Saved {len(cards)} cards to {cards_path}")

    # Save booster_config.json
    config_path = os.path.join(output_dir, 'booster_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(booster_config, f, indent=2)
    print(f"[OK] Saved booster config to {config_path}")

    print(f"=== MTGJson data cached successfully ===\n")

def build_filtered_sheets(cards: List[Dict], training_cards: Set[str], booster_config: Dict = None) -> Dict[str, Dict[str, float]]:
    """
    Transform MTGJson sheets from UUIDs to card names, filtered by 17lands training data.

    Args:
        cards: All cards from MTGJson
        training_cards: Card names from 17lands training data
        booster_config: Booster config with weighted sheets

    Returns:
        Dictionary of sheets with card names mapped to weights
    """
    if not booster_config or 'play' not in booster_config:
        raise ValueError("No booster config found - cannot build sheets")

    # Build UUID to name mapping
    uuid_to_name = {card['uuid']: card['name'] for card in cards}

    sheets = {}
    play_config = booster_config['play']

    for sheet_name, sheet_data in play_config.get('sheets', {}).items():
        sheet_cards = sheet_data.get('cards', {})
        weighted_cards = {}

        for uuid, weight in sheet_cards.items():
            if uuid in uuid_to_name:
                card_name = uuid_to_name[uuid]

                # Check if card should be included
                should_include = not training_cards or card_name in training_cards

                if should_include:
                    weighted_cards[card_name] = float(weight)

        if weighted_cards:
            sheets[sheet_name] = {
                "cards": weighted_cards,
                "totalWeight": sum(weighted_cards.values())
            }

    return sheets

def build_and_save_sheets(set_code: str, output_dir: str):
    """Build sheets and save to disk. Called during training/add_set."""

    with open(f"{output_dir}/cards.json", 'r', encoding='utf-8') as f:
        cards = json.load(f)

    with open(f"{output_dir}/booster_config.json", 'r', encoding='utf-8') as f:
        booster_config = json.load(f)

    training_cards_path = f"{output_dir}/seventeenlands_cards.json"
    if os.path.exists(training_cards_path):
        with open(training_cards_path, 'r', encoding='utf-8') as f:
            training_cards = set(json.load(f))
    else:
        training_cards = set()

    sheets = build_filtered_sheets(cards, training_cards, booster_config)

    with open(f"{output_dir}/sheets.json", 'w', encoding='utf-8') as f:
        json.dump(sheets, f, indent=2)

    print(f"=== Saved sheets to {output_dir}/sheets.json ===")

