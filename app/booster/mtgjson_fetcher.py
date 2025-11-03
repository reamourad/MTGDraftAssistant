"""
Fetch and cache MTGJson data during training.
"""
import requests
import json
import os
from typing import Dict, List, Any

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
        simplified_cards.append({
            'name': card.get('name'),
            'uuid': card.get('uuid'),
            'rarity': card.get('rarity', '').lower(),
            'colors': card.get('colors', []),
            'types': card.get('types', []),
            'manaCost': card.get('manaCost', ''),
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
