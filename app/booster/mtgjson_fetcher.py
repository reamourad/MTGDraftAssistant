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


def extract_cards(set_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract card list from set data.

    Args:
        set_data: Full set data from MTGJson

    Returns:
        List of card dictionaries with name, uuid, rarity, etc.
    """
    cards = set_data.get('data', {}).get('cards', [])

    # Simplify card data - keep only what we need
    simplified_cards = []
    for card in cards:
        power = card.get('power')
        toughness = card.get('toughness')

        power_val = float(power) if power and power.lstrip('-').isdigit() else None
        toughness_val = float(toughness) if toughness and toughness.lstrip('-').isdigit() else None

        simplified_cards.append({
            'name': card.get('name'),
            'uuid': card.get('uuid'),
            'rarity': card.get('rarity', '').lower(),
            'colors': card.get('colors', []),
            'mana_cost': card.get('manaCost', ''),
            'converted_mana_cost': card.get('manaValue', 0),
            'types': card.get('types', []),
            'subtypes': card.get('subtypes', []),
            'power': power_val,
            'toughness': toughness_val,
            'can_attack': power_val is not None and toughness_val is not None,
            'keywords': card.get('keywords', []),
            'oracle_text': card.get('text', ''),
        })

    print(f"[OK] Extracted {len(simplified_cards)} cards")
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

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save cards.json
    cards_path = os.path.join(output_dir, 'cards.json')
    with open(cards_path, 'w', encoding='utf-8') as f:
        json.dump(cards, f, indent=2)
    print(f"[OK] Saved cards to {cards_path}")

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
                # Filter by training cards
                if not training_cards or card_name in training_cards:
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

