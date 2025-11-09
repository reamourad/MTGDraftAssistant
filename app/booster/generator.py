"""
Generate draft booster packs using cached MTGJson data.

Uses MTGJson play booster structure but filters to cards available
in 17lands training data.
"""
import json
import os
import random
import glob
import pandas as pd
from typing import List, Dict, Set, Union, Tuple


def select_weighted_item(weighted_items: Dict[str, float], total_weight: float) -> str:
    """
    Select a random item based on weights (matching React approach).

    Args:
        weighted_items: Dictionary mapping items to their weights
        total_weight: Sum of all weights

    Returns:
        Selected item key
    """
    random_num = random.random() * total_weight

    for item_key, weight in weighted_items.items():
        if random_num < weight:
            return item_key
        random_num -= weight

    # Fallback to last item (handles floating point precision issues)
    return list(weighted_items.keys())[-1] if weighted_items else ""


def load_training_card_names(set_code: str) -> Set[str]:
    """
    Load card names from saved training_cards.json.

    This file is created during training and contains the list of cards
    from the 17lands dataset, so we don't need the CSV at runtime.

    Args:
        set_code: Set code (e.g., 'MH3')

    Returns:
        Set of card names that exist in training data
    """
    training_cards_path = f"app/models/{set_code.upper()}/training_cards.json"

    # Try to load from saved training cards (created during training)
    if os.path.exists(training_cards_path):
        with open(training_cards_path, 'r', encoding='utf-8') as f:
            card_names = json.load(f)
        return set(card_names)

    # Fallback: Try to load from CSV (if available locally)
    csv_files = glob.glob(f"data/{set_code.upper()}/*.csv.gz")
    if not csv_files:
        csv_files = glob.glob(f"data/{set_code.upper()}/*.csv")

    if not csv_files:
        # No training data - use all MTGJson cards
        print(f"[BOOSTER] Warning: No training_cards.json or CSV found for {set_code}")
        print(f"[BOOSTER] Using all MTGJson cards (train model first to enable filtering)")
        return set()

    # Read just the column names (first row) to get card list
    df = pd.read_csv(csv_files[0], nrows=0)

    # Extract card names from "pack_card_" columns
    card_names = set()
    for col in df.columns:
        if col.startswith("pack_card_"):
            card_names.add(col[len("pack_card_"):])

    return card_names


def build_filtered_sheets(cards: List[Dict], training_cards: Set[str], booster_config: Dict = None) -> Dict[str, Union[List[str], Dict[str, float]]]:
    """
    Build card sheets filtered to training data cards.

    Args:
        cards: All cards from MTGJson
        training_cards: Card names from 17lands training data
        booster_config: Optional booster config with weighted sheets

    Returns:
        Dictionary of sheets. Each sheet is either:
        - List[str]: Simple list of card names (for basic rarity sheets)
        - Dict[str, float]: Card names mapped to weights (for weighted MTGJSON sheets)
    """
    # Basic lands should only appear in the 'land' slot, not in rarity sheets
    BASIC_LANDS = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}

    sheets = {
        'common': [],
        'uncommon': [],
        'rare': [],
        'mythic': [],
        'land': [],
        'all': []  # Fallback for wildcards
    }

    # Track which card names we've already added to prevent duplicates
    seen_in_sheets = {
        'common': set(),
        'uncommon': set(),
        'rare': set(),
        'mythic': set(),
        'land': set(),
        'all': set()
    }

    # Build UUID to name mapping
    uuid_to_name = {card['uuid']: card['name'] for card in cards}

    for card in cards:
        name = card['name']

        # Filter: only include if in training data (or if no training data available)
        if training_cards and name not in training_cards:
            continue

        rarity = card.get('rarity', 'common').lower()
        types = card.get('types', [])
        is_basic_land = name in BASIC_LANDS

        # Add to rarity sheets (but exclude basic lands from rarity sheets)
        # Only add if we haven't seen this card name in this sheet before
        if rarity in sheets and not is_basic_land and name not in seen_in_sheets[rarity]:
            sheets[rarity].append(name)
            seen_in_sheets[rarity].add(name)

        # Add to land sheet (including basic lands)
        if 'Land' in types and name not in seen_in_sheets['land']:
            sheets['land'].append(name)
            seen_in_sheets['land'].add(name)

        # Add to 'all' sheet (but exclude basic lands to prevent them in wildcards)
        if not is_basic_land and name not in seen_in_sheets['all']:
            sheets['all'].append(name)
            seen_in_sheets['all'].add(name)

    # Add weighted sheets from booster config if available
    if booster_config and 'play' in booster_config:
        play_config = booster_config['play']
        config_sheets = play_config.get('sheets', {})

        for sheet_name, sheet_data in config_sheets.items():
            sheet_cards = sheet_data.get('cards', {})
            weighted_cards = {}

            for uuid, weight in sheet_cards.items():
                if uuid in uuid_to_name:
                    card_name = uuid_to_name[uuid]
                    # Filter by training cards
                    if not training_cards or card_name in training_cards:
                        # Preserve exact weights from MTGJSON
                        weighted_cards[card_name] = float(weight)

            if weighted_cards:
                sheets[sheet_name] = weighted_cards

    return sheets


def pick_from_sheet(sheet: Union[List[str], Dict[str, float]], count: int, already_picked: Set[str] = None) -> List[str]:
    """
    Pick cards from a sheet, with or without weights.

    Args:
        sheet: Either a list of card names or dict of {card_name: weight}
        count: Number of cards to pick
        already_picked: Set of cards already picked (to avoid duplicates)

    Returns:
        List of picked card names
    """
    if not sheet:
        return []

    if already_picked is None:
        already_picked = set()

    picked = []

    # Check if sheet is weighted (dict) or unweighted (list)
    if isinstance(sheet, dict):
        # Weighted selection
        for _ in range(count):
            # Filter out already picked cards
            available = {name: weight for name, weight in sheet.items() if name not in already_picked}
            if not available:
                break

            total_weight = sum(available.values())
            selected = select_weighted_item(available, total_weight)
            picked.append(selected)
            already_picked.add(selected)
    else:
        # Unweighted selection (simple random sample)
        available = [card for card in sheet if card not in already_picked]
        if available:
            picked = random.sample(available, min(count, len(available)))
            already_picked.update(picked)

    return picked


def generate_booster(set_code: str) -> List[str]:
    """
    Generate a draft booster pack using MTGJson play booster config.

    Follows authentic MTGA pack structure but only uses cards from
    17lands training data.

    Args:
        set_code: Set code (e.g., 'MH3', 'BLB')

    Returns:
        List of card names in the booster pack
    """
    set_dir = f"app/models/{set_code.upper()}"

    # Load cached MTGJson data
    with open(f"{set_dir}/cards.json", 'r', encoding='utf-8') as f:
        cards = json.load(f)

    with open(f"{set_dir}/booster_config.json", 'r', encoding='utf-8') as f:
        booster_config = json.load(f)

    # Load training data card names
    training_cards = load_training_card_names(set_code)
    print(f"\n[BOOSTER] Generating pack for {set_code}")
    print(f"[BOOSTER] Training cards found: {len(training_cards)}")
    print(f"[BOOSTER] Filter active: {bool(training_cards)}")

    # Build filtered sheets (including weighted sheets from booster config)
    sheets = build_filtered_sheets(cards, training_cards, booster_config)
    print(f"[BOOSTER] Sheet sizes: common={len(sheets['common'])}, uncommon={len(sheets['uncommon'])}, rare={len(sheets['rare'])}, mythic={len(sheets['mythic'])}, land={len(sheets['land'])}")

    # Log basic lands check
    basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
    basics_in_common = sum(1 for c in sheets['common'] if c in basic_lands)
    basics_in_land = sum(1 for c in sheets['land'] if c in basic_lands)
    print(f"[BOOSTER] Basic lands: {basics_in_land} in land sheet, {basics_in_common} in common sheet")

    # Use "play" booster if available
    if 'play' not in booster_config:
        # Fallback: simple 10/3/1 distribution
        print(f"[BOOSTER] No 'play' config found, using fallback 10/3/1 distribution")
        pack = []
        picked_cards = set()

        pack.extend(pick_from_sheet(sheets['common'], 10, picked_cards))
        pack.extend(pick_from_sheet(sheets['uncommon'], 3, picked_cards))

        # For rare slot: 1/8 chance for mythic
        rare_mythic = []
        if random.random() < 0.125 and sheets['mythic']:
            rare_mythic = pick_from_sheet(sheets['mythic'], 1, picked_cards)
        elif sheets['rare']:
            rare_mythic = pick_from_sheet(sheets['rare'], 1, picked_cards)
        pack.extend(rare_mythic)

        print(f"[BOOSTER] Generated {len(pack)} cards using fallback\n")
        return pack

    # Parse play booster config
    play_config = booster_config['play']
    boosters = play_config.get('boosters', [])
    print(f"[BOOSTER] Using 'play' booster config ({len(boosters)} variations)")

    # Select a booster configuration based on weight
    booster_weights = {f"config_{i}": booster.get('weight', 1) for i, booster in enumerate(boosters)}
    total_weight = play_config.get('boostersTotalWeight', sum(booster_weights.values()))

    selected_key = select_weighted_item(booster_weights, total_weight)
    selected_index = int(selected_key.split('_')[1])
    selected_booster = boosters[selected_index]

    booster_weight = selected_booster.get('weight', 1)
    print(f"[BOOSTER] Selected variation #{selected_index + 1} (weight: {booster_weight}/{total_weight})")

    # Generate pack following the selected configuration
    pack = []
    picked_cards = set()  # Track cards already picked to prevent duplicates
    contents = selected_booster.get('contents', {})
    print(f"[BOOSTER] Pack contents: {contents}")

    for slot_name, count in contents.items():
        slot_lower = slot_name.lower()
        before_count = len(pack)

        # First try to match exact sheet name from booster config
        if slot_name in sheets and sheets[slot_name]:
            picked = pick_from_sheet(sheets[slot_name], count, picked_cards)
            pack.extend(picked)
            sheet_type = "weighted" if isinstance(sheets[slot_name], dict) else "unweighted"
            print(f"[BOOSTER]   {slot_name} ({count}x, {sheet_type}): {', '.join(picked) if picked else 'EMPTY'}")
        # Map slot names to sheets with fallbacks
        # Check uncommon BEFORE common (since "uncommon" contains "common")
        elif 'uncommon' in slot_lower or slot_lower == 'newuncommon':
            picked = pick_from_sheet(sheets['uncommon'], count, picked_cards)
            pack.extend(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'common' in slot_lower:
            picked = pick_from_sheet(sheets['common'], count, picked_cards)
            pack.extend(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'rare' in slot_lower or 'mythic' in slot_lower or slot_lower == 'newraremythic':
            # Use the weighted newRareMythic sheet if available
            if 'newRareMythic' in sheets and sheets['newRareMythic']:
                picked = pick_from_sheet(sheets['newRareMythic'], count, picked_cards)
                pack.extend(picked)
                sheet_type = "weighted" if isinstance(sheets['newRareMythic'], dict) else "unweighted"
                print(f"[BOOSTER]   {slot_name} ({count}x, {sheet_type}): {', '.join(picked) if picked else 'EMPTY'}")
            else:
                # Fallback: 1/8 chance for mythic, otherwise rare
                picked = []
                for _ in range(count):
                    # Try mythic first (1/8 chance)
                    if random.random() < 0.125 and sheets['mythic']:
                        mythic_pick = pick_from_sheet(sheets['mythic'], 1, picked_cards)
                        if mythic_pick:
                            pack.extend(mythic_pick)
                            picked.append(mythic_pick[0] + " (M)")
                            continue

                    # Fall back to rare
                    if sheets['rare']:
                        rare_pick = pick_from_sheet(sheets['rare'], 1, picked_cards)
                        if rare_pick:
                            pack.extend(rare_pick)
                            picked.append(rare_pick[0] + " (R)")
                print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'land' in slot_lower:
            picked = pick_from_sheet(sheets['land'], count, picked_cards)
            pack.extend(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        else:
            # Unknown slot: try to use 'all' as fallback
            picked = pick_from_sheet(sheets.get('all', []), count, picked_cards)
            pack.extend(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'} [FALLBACK]")

    print(f"[BOOSTER] Total pack size: {len(pack)} cards\n")
    return pack
