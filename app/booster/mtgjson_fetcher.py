"""
Fetch and cache MTGJson data during training.
"""
import requests
import json
import os
from typing import Dict, List, Any, Set, Union


MTGJSON_BASE_URL = "https://mtgjson.com/api/v5"

# Cache for set list to avoid repeated API calls
_SET_LIST_CACHE = None


def get_set_list() -> List[Dict[str, Any]]:
    """
    Fetch and cache the list of all MTGJson sets.
    
    Returns:
        List of set metadata dictionaries
    """
    global _SET_LIST_CACHE
    
    if _SET_LIST_CACHE is None:
        print("[INFO] Fetching MTGJson set list...")
        response = requests.get(f"{MTGJSON_BASE_URL}/SetList.json", timeout=30)
        response.raise_for_status()
        _SET_LIST_CACHE = response.json()['data']
        print(f"[OK] Loaded {len(_SET_LIST_CACHE)} sets from MTGJson")
    
    return _SET_LIST_CACHE


def find_companion_sets(set_code: str) -> List[str]:
    """
    Dynamically find companion sets (Commander, Jumpstart, etc.) for a given set.
    
    Uses MTGJson's parentCode metadata to find related sets.
    
    Args:
        set_code: Main set code (e.g., 'MH3', 'TLA')
    
    Returns:
        List of companion set codes (e.g., ['M3C'] for MH3, ['TLE'] for TLA)
    """
    try:
        all_sets = get_set_list()
        
        # Find sets where parentCode matches our set_code
        # Include commander, jumpstart, eternal, and other related types
        companion_sets = [
            s['code'] for s in all_sets 
            if s.get('parentCode') == set_code
        ]
        
        if companion_sets:
            print(f"[INFO] Found companion sets for {set_code}: {companion_sets}")
        else:
            print(f"[INFO] No companion sets found for {set_code}")
        
        return companion_sets
    
    except Exception as e:
        print(f"[WARN] Could not fetch companion sets for {set_code}: {e}")
        return []


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

def fetch_bonus_sheet_cards(set_code: str, booster_config: Dict[str, Any], card_names_filter: set) -> List[Dict[str, Any]]:
    """
    Fetch bonus sheet cards (SPG, The List, Source Material, etc.) from booster config.
    
    These are cards from other sets that appear in boosters. We check:
    1. SPG (Special Guests)
    2. PLST (The List - contains reprints from many sets)
    
    Args:
        set_code: Main set code
        booster_config: Booster configuration with UUIDs
        card_names_filter: Card names from 17Lands CSV
    
    Returns:
        List of bonus sheet cards
    """
    bonus_cards = []
    
    # Get all UUIDs from booster config
    booster_uuids = extract_uuids_from_booster_config(booster_config)
    if not booster_uuids:
        return bonus_cards
    
    print(f"[INFO] Found {len(booster_uuids)} UUIDs in booster configuration")
    
    # Check bonus sheet sets: SPG and PLST
    bonus_sets = [
        ('SPG', fetch_spg_set),
        ('PLST', lambda: fetch_set_data('PLST'))
    ]
    
    for bonus_set_code, fetch_func in bonus_sets:
        try:
            bonus_set_data = fetch_func()
            bonus_set_cards = bonus_set_data.get('data', {}).get('cards', [])
            bonus_set_uuids = {card['uuid'] for card in bonus_set_cards}
            
            # Find UUIDs that are in both booster config and this bonus set
            matching_uuids = booster_uuids.intersection(bonus_set_uuids)
            
            if matching_uuids:
                print(f"[INFO] Found {len(matching_uuids)} {bonus_set_code} cards in booster configuration")
                # Extract only the matching cards
                matching_cards = extract_cards(bonus_set_data, matching_uuids)
                
                # Filter to only cards in 17Lands CSV
                filtered_cards = [
                    card for card in matching_cards
                    if card['name'] in card_names_filter
                ]
                
                if filtered_cards:
                    bonus_cards.extend(filtered_cards)
                    print(f"[OK] Added {len(filtered_cards)} {bonus_set_code} cards from 17Lands data")
                    for card in filtered_cards[:5]:  # Show first 5
                        print(f"  - {card['name']}")
                    if len(filtered_cards) > 5:
                        print(f"  ... and {len(filtered_cards) - 5} more")
        
        except Exception as e:
            print(f"[WARN] Could not fetch bonus set {bonus_set_code}: {e}")
            continue
    
    return bonus_cards


def fetch_companion_sets(set_code: str, card_names_filter: set = None) -> List[Dict[str, Any]]:
    """
    Fetch cards from companion sets (e.g., Commander decks).
    
    Automatically discovers companion sets using MTGJson's parentCode metadata.
    Only fetches cards that are in the card_names_filter (from 17Lands CSV).
    
    Args:
        set_code: Main set code (e.g., 'MH3')
        card_names_filter: Set of card names to filter by (from 17Lands CSV)
    
    Returns:
        List of cards from companion sets that match the filter
    """
    if not card_names_filter:
        return []
    
    companion_cards = []
    
    # Dynamically find companion sets
    companion_set_codes = find_companion_sets(set_code)
    
    if not companion_set_codes:
        return companion_cards
    
    for companion_code in companion_set_codes:
        try:
            print(f"[INFO] Fetching companion set: {companion_code}")
            companion_data = fetch_set_data(companion_code)
            all_companion_cards = extract_cards(companion_data)
            
            # Filter to only cards in 17Lands CSV
            filtered_cards = [
                card for card in all_companion_cards 
                if card['name'] in card_names_filter
            ]
            
            if filtered_cards:
                companion_cards.extend(filtered_cards)
                print(f"[OK] Found {len(filtered_cards)}/{len(all_companion_cards)} cards from {companion_code} in 17Lands data")
                for card in filtered_cards:
                    print(f"  - {card['name']}")
            else:
                print(f"[INFO] No cards from {companion_code} found in 17Lands data")
                
        except Exception as e:
            print(f"[WARN] Could not fetch companion set {companion_code}: {e}")
            continue
    
    return companion_cards


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
    
    Args:
        set_data: Full set data from MTGJson
        uuids: Optional set of UUIDs to filter cards (None = all cards)
    
    Returns:
        List of simplified card dictionaries
    """
    cards = set_data.get('data', {}).get('cards', [])

    # Filter by UUIDs first if provided
    if uuids is not None:
        cards = [card for card in cards if card.get('uuid') in uuids]
        print(f"[DEBUG] Filtered to {len(cards)} cards matching {len(uuids)} UUIDs")

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

        # Calculate converted mana cost (CMC)
        cmc = card.get('manaValue', 0)  # MTGJson uses 'manaValue' for CMC
        
        # Determine if card can attack (creatures with power/toughness)
        can_attack = 'Creature' in card.get('types', []) and card.get('power') is not None
        
        # Get subtypes
        subtypes = card.get('subtypes', [])
        
        # Create simplified card
        simplified_cards.append({
            'name': name,
            'uuid': card.get('uuid'),
            'rarity': card.get('rarity', '').lower(),
            'colors': card.get('colors', []),
            'types': card.get('types', []),
            'subtypes': subtypes,
            'manaCost': card.get('manaCost', ''),
            'mana_cost': card.get('manaCost', ''),  # Alias for CardEncoder
            'converted_mana_cost': cmc,  # For CardEncoder
            'text': oracle_text,  # Combined text for DFCs
            'oracle_text': oracle_text,  # Alias for CardEncoder
            'power': card.get('power'),
            'toughness': card.get('toughness'),
            'can_attack': can_attack,  # For CardEncoder
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
        - output_dir/booster_config.json (structure only, no card data)
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

    # Extract and save minimal booster structure (no card data)
    booster_structure = extract_booster_structure(booster_config)
    config_path = os.path.join(output_dir, 'booster_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(booster_structure, f, indent=2)
    print(f"[OK] Saved booster structure to {config_path}")

    print(f"=== MTGJson data cached successfully ===\n")

def extract_booster_structure(booster_config: Dict) -> Dict:
    """
    Extract just the booster structure (no card data).
    
    This includes the boosters array with weights and contents,
    but removes the actual card sheets.
    
    Args:
        booster_config: Full booster config from MTGJson
        
    Returns:
        Minimal booster structure without card data
    """
    if not booster_config or 'play' not in booster_config:
        raise ValueError("No play booster configuration found")
    
    play_config = booster_config['play']
    
    # Extract structure without card data
    structure = {
        'play': {
            'boosters': play_config.get('boosters', []),
            'boostersTotalWeight': play_config.get('boostersTotalWeight', 0),
            'name': play_config.get('name', ''),
        }
    }
    
    return structure


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
                "totalWeight": sum(weighted_cards.values()),
                "foil": sheet_data.get('foil', False)
            }

    return sheets

def build_and_save_sheets(set_code: str, output_dir: str):
    """
    Build sheets from cards and booster config, save to disk.
    
    Called during training/add_set to create the sheets.json file
    with card names and weights (separate from booster structure).
    """

    with open(f"{output_dir}/cards.json", 'r', encoding='utf-8') as f:
        cards = json.load(f)

    # Load the FULL booster config from MTGJson (with UUIDs)
    # We need this to build the sheets, but we won't save it
    set_data = fetch_set_data(set_code)
    full_booster_config = extract_booster_config(set_data)

    training_cards_path = f"{output_dir}/seventeenlands_cards.json"
    if os.path.exists(training_cards_path):
        with open(training_cards_path, 'r', encoding='utf-8') as f:
            training_cards = set(json.load(f))
    else:
        training_cards = set()

    sheets = build_filtered_sheets(cards, training_cards, full_booster_config)

    with open(f"{output_dir}/sheets.json", 'w', encoding='utf-8') as f:
        json.dump(sheets, f, indent=2)

    print(f"[OK] Saved {len(sheets)} sheets to {output_dir}/sheets.json")


def fetch_all_card_data(set_code: str, card_names_filter: set) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch all card data for a set including main set, companion sets, and bonus sheets.
    
    This function fetches:
    1. Main set cards
    2. Companion sets (Commander decks)
    3. SPG cards (if in booster config)
    4. Bonus sheet cards from other sets (e.g., "The List", "Source Material")
    
    Args:
        set_code: Main set code (e.g., 'MH3')
        card_names_filter: Set of card names from 17Lands CSV
    
    Returns:
        Tuple of (all_cards, booster_config)
    """
    # Fetch main set data
    set_data = fetch_set_data(set_code)
    all_cards = extract_cards(set_data)
    booster_config = extract_booster_config(set_data)
    
    # Fetch companion sets (Commander decks, etc.)
    companion_cards = fetch_companion_sets(set_code, card_names_filter)
    if companion_cards:
        all_cards.extend(companion_cards)
    
    # Fetch bonus sheet cards (SPG and others from booster config)
    bonus_cards = fetch_bonus_sheet_cards(set_code, booster_config, card_names_filter)
    if bonus_cards:
        all_cards.extend(bonus_cards)
        print(f"[OK] Added {len(bonus_cards)} bonus sheet cards")
    
    return all_cards, booster_config


def save_booster_files(
    set_code: str,
    output_dir: str,
    all_cards: List[Dict[str, Any]],
    card_names_filter: set,
    booster_config: Dict[str, Any]
):
    """
    Save booster configuration, cards, and sheets to disk.
    
    Args:
        set_code: Set code
        output_dir: Directory to save files
        all_cards: All cards (main set + companion + SPG)
        card_names_filter: Card names from 17Lands CSV
        booster_config: Booster configuration
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cards.json (needed by training script)
    cards_path = os.path.join(output_dir, 'cards.json')
    with open(cards_path, 'w', encoding='utf-8') as f:
        json.dump(all_cards, f, indent=2)
    print(f"[OK] Saved {len(all_cards)} cards to {cards_path}")
    
    # Save booster_config.json
    booster_structure = extract_booster_structure(booster_config)
    config_path = os.path.join(output_dir, 'booster_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(booster_structure, f, indent=2)
    print(f"[OK] Saved booster config to {config_path}")
    
    # Build and save sheets.json
    try:
        sheets = build_filtered_sheets(all_cards, card_names_filter, booster_config)
        sheets_path = os.path.join(output_dir, 'sheets.json')
        with open(sheets_path, 'w', encoding='utf-8') as f:
            json.dump(sheets, f, indent=2)
        print(f"[OK] Saved {len(sheets)} sheets to {sheets_path}")
    except Exception as e:
        print(f"[WARN] Could not build sheets: {e}")


