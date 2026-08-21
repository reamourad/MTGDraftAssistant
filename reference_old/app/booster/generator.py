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
import logging

logger = logging.getLogger("uvicorn.error")

_sheets_cache = {}
_booster_config_cache = {}


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

#Gets and returns the sheets model and the booster model that we will cache
def load_set_data(set_code: str) -> tuple[Dict, Dict]:
    set_code = set_code.upper();

    if set_code not in _sheets_cache:
        set_dir = f"app/models/{set_code}"

        with open(f"{set_dir}/sheets.json", "r") as set_data_json:
            _sheets_cache[set_code] = json.load(set_data_json)

        with open(f"{set_dir}/booster_config.json", "r") as booster_data_json:
            _booster_config_cache[set_code] = json.load(booster_data_json)

    return _sheets_cache[set_code], _booster_config_cache[set_code]


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

    logger.info("hello")
    #Check if the data is already in cache, if not load it
    sheets, booster_config = load_set_data(set_code)

    #Error if there's no play booster
    if "play" not in booster_config:
        raise ValueError("There is no play booster in the booster config")

    play_config = booster_config["play"]
    #If there is, randomize the booster sheet we need
    total_weight = play_config["boostersTotalWeight"]
    logger.info("total weight:" + str(total_weight))

    sheet_random_weight = random.random() * total_weight
    sheet_pick = {}
    #for each sheet in booster config, we get the weight, we check if the weight is bigger than our sheet_pick
    #if not we minus the weight and we go to next


    for sheet in play_config["boosters"]:
        sheet_random_weight -= sheet["weight"]
        if sheet_random_weight <= 0:
            sheet_pick = sheet
            break

    logger.info("sheet pick:" + str(sheet_pick))
    pack = []
    picked_cards = set()
    contents = sheet_pick.get('contents', {})

    logger.info(f"contents:{contents}")
    for sheet_name, count in contents.items():
        logger.info(f"sheet:{sheets[sheet_name]}")
        #find the sheet and pick the right number from it
        if sheet_name in sheets:
            picked = pick_from_sheet(sheets[sheet_name]["cards"], count, picked_cards)
            pack.extend(picked)

    return pack