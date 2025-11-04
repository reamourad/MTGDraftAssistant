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
from typing import List, Dict, Set


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


def build_filtered_sheets(cards: List[Dict], training_cards: Set[str]) -> Dict[str, List[str]]:
    """
    Build card sheets filtered to training data cards.

    Args:
        cards: All cards from MTGJson
        training_cards: Card names from 17lands training data

    Returns:
        Dictionary of sheets (rarity-based and special categories)
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

    return sheets


def pick_from_sheet(sheet: List[str], count: int) -> List[str]:
    """Pick cards from a sheet, without replacement."""
    if not sheet:
        return []
    return random.sample(sheet, min(count, len(sheet)))


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

    # Build filtered sheets
    sheets = build_filtered_sheets(cards, training_cards)
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
        pack.extend(pick_from_sheet(sheets['common'], 10))
        pack.extend(pick_from_sheet(sheets['uncommon'], 3))

        if random.random() < 0.125 and sheets['mythic']:
            pack.append(random.choice(sheets['mythic']))
        else:
            pack.append(random.choice(sheets['rare']) if sheets['rare'] else '')

        print(f"[BOOSTER] Generated {len(pack)} cards using fallback\n")
        return pack

    # Parse play booster config
    play_config = booster_config['play']
    boosters = play_config.get('boosters', [])
    print(f"[BOOSTER] Using 'play' booster config ({len(boosters)} variations)")

    # Select a booster configuration based on weight
    total_weight = play_config.get('boostersTotalWeight', len(boosters))
    rand_value = random.uniform(0, total_weight)

    selected_booster = boosters[0]
    selected_index = 0
    cumulative = 0
    for i, booster in enumerate(boosters):
        cumulative += booster.get('weight', 1)
        if rand_value <= cumulative:
            selected_booster = booster
            selected_index = i
            break

    booster_weight = selected_booster.get('weight', 1)
    print(f"[BOOSTER] Selected variation #{selected_index + 1} (weight: {booster_weight}/{total_weight})")

    # Generate pack following the selected configuration
    pack = []
    picked_cards = set()  # Track cards already picked to prevent duplicates
    contents = selected_booster.get('contents', {})
    print(f"[BOOSTER] Pack contents: {contents}")

    def pick_unique(sheet: List[str], count: int, already_picked: set) -> List[str]:
        """Pick cards from sheet, excluding already picked cards."""
        available = [card for card in sheet if card not in already_picked]
        if not available:
            return []
        return random.sample(available, min(count, len(available)))

    for slot_name, count in contents.items():
        slot_lower = slot_name.lower()
        before_count = len(pack)

        # Map slot names to sheets with fallbacks
        # Check uncommon BEFORE common (since "uncommon" contains "common")
        if 'uncommon' in slot_lower or slot_lower == 'newuncommon':
            picked = pick_unique(sheets['uncommon'], count, picked_cards)
            pack.extend(picked)
            picked_cards.update(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'common' in slot_lower:
            picked = pick_unique(sheets['common'], count, picked_cards)
            pack.extend(picked)
            picked_cards.update(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'rare' in slot_lower or 'mythic' in slot_lower or slot_lower == 'newraremythic':
            # 1/8 chance for mythic, otherwise rare
            picked = []
            for _ in range(count):
                # Try mythic first (1/8 chance)
                if random.random() < 0.125 and sheets['mythic']:
                    available_mythics = [c for c in sheets['mythic'] if c not in picked_cards]
                    if available_mythics:
                        card = random.choice(available_mythics)
                        pack.append(card)
                        picked_cards.add(card)
                        picked.append(card + " (M)")
                        continue

                # Fall back to rare
                if sheets['rare']:
                    available_rares = [c for c in sheets['rare'] if c not in picked_cards]
                    if available_rares:
                        card = random.choice(available_rares)
                        pack.append(card)
                        picked_cards.add(card)
                        picked.append(card + " (R)")
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'land' in slot_lower:
            picked = pick_unique(sheets['land'], count, picked_cards)
            pack.extend(picked)
            picked_cards.update(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'wildcard' in slot_lower or 'reprint' in slot_lower:
            # Wildcard/reprint: pick from any rarity
            picked = pick_unique(sheets['all'], count, picked_cards)
            pack.extend(picked)
            picked_cards.update(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        elif 'foil' in slot_lower or 'showcase' in slot_lower:
            # Foil/showcase: pick from all cards (simplified)
            picked = pick_unique(sheets['all'], count, picked_cards)
            pack.extend(picked)
            picked_cards.update(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'}")
        else:
            # Unknown slot: pick from all
            picked = pick_unique(sheets['all'], count, picked_cards)
            pack.extend(picked)
            picked_cards.update(picked)
            print(f"[BOOSTER]   {slot_name} ({count}x): {', '.join(picked) if picked else 'EMPTY'} [UNKNOWN SLOT]")

    print(f"[BOOSTER] Total pack size: {len(pack)} cards\n")
    return pack
