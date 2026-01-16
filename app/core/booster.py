"""
Booster generation service for MTG Draft Assistant.

This module provides a clean interface for generating draft booster packs
using MTGJson play booster configuration and filtered card sheets.
"""
import json
import random
import logging
from typing import List, Dict, Set, Union, Optional

logger = logging.getLogger("uvicorn.error")


class BoosterService:
    """
    Service for generating draft booster packs.
    
    Uses MTGJson play booster structure but filters to cards available
    in 17lands training data.
    """
    
    def __init__(self):
        """Initialize the BoosterService with empty caches."""
        self._booster_structure_cache: Dict[str, Dict] = {}
        self._sheets_cache: Dict[str, Dict] = {}
    
    def generate_booster(self, set_code: str) -> List[str]:
        """
        Generate a draft booster pack using MTGJson play booster config.
        
        Follows authentic MTGA pack structure but only uses cards from
        17lands training data.
        
        Args:
            set_code: Set code (e.g., 'MH3', 'BLB')
            
        Returns:
            List of card names in the booster pack
            
        Raises:
            ValueError: If the set has no play booster configuration
            FileNotFoundError: If set data files are not found
        """
        set_code = set_code.upper()
        
        # Load set data (uses cache if available)
        booster_structure, sheets = self._load_set_data(set_code)
        
        # Validate play booster exists
        if "play" not in booster_structure:
            raise ValueError(f"No play booster configuration found for set {set_code}")
        
        play_config = booster_structure["play"]
        
        # Select a random booster sheet based on weights
        sheet_pick = self._select_booster_sheet(play_config)
        
        # Generate pack by picking cards from sheets
        pack = self._pick_cards_from_sheets(sheet_pick, sheets)
        
        return pack
    
    def _load_set_data(self, set_code: str) -> tuple[Dict, Dict]:
        """
        Load sheets and booster configuration for a set.
        
        Uses caching to avoid repeated file I/O.
        
        Args:
            set_code: Set code (uppercase)
            
        Returns:
            Tuple of (booster_config, sheets_data)
            
        Raises:
            FileNotFoundError: If set data files are not found
        """
        if set_code not in self._sheets_cache:
            set_dir = f"app/models/{set_code}"
            
            try:
                with open(f"{set_dir}/sheets.json", "r") as f:
                    self._sheets_cache[set_code] = json.load(f)
                
                with open(f"{set_dir}/booster_config.json", "r") as f:
                    self._booster_structure_cache[set_code] = json.load(f)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Set data not found for {set_code}. "
                    f"Ensure sheets.json and booster_config.json exist in {set_dir}"
                ) from e
        
        return self._booster_structure_cache[set_code], self._sheets_cache[set_code]

    
    def _select_booster_sheet(self, play_config: Dict) -> Dict:
        """
        Select a random booster sheet based on weights.
        
        Args:
            play_config: Play booster configuration with boosters and weights
            
        Returns:
            Selected booster sheet configuration
        """
        total_weight = play_config["boostersTotalWeight"]
        sheet_random_weight = random.random() * total_weight
        
        for sheet in play_config["boosters"]:
            sheet_random_weight -= sheet["weight"]
            if sheet_random_weight <= 0:
                return sheet
        
        # Fallback to last sheet (handles floating point precision)
        return play_config["boosters"][-1] if play_config["boosters"] else {}
    
    def _pick_cards_from_sheets(self, sheet_pick: Dict, sheets: Dict) -> List[str]:
        """
        Pick cards from the selected booster sheet configuration.
        
        Args:
            sheet_pick: Selected booster sheet with contents
            sheets: Available card sheets
            
        Returns:
            List of card names for the pack
        """
        pack = []
        picked_cards = set()
        contents = sheet_pick.get('contents', {})
        
        for sheet_name, count in contents.items():
            if sheet_name in sheets:
                sheet_data = sheets[sheet_name]["cards"]
                picked = self._pick_from_sheet(sheet_data, count, picked_cards)
                pack.extend(picked)
        
        return pack
    
    def _pick_from_sheet(
        self, 
        sheet: Union[List[str], Dict[str, float]], 
        count: int, 
        already_picked: Set[str]
    ) -> List[str]:
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
        
        picked = []
        
        # Check if sheet is weighted (dict) or unweighted (list)
        if isinstance(sheet, dict):
            # Weighted selection
            for _ in range(count):
                # Filter out already picked cards
                available = {
                    name: weight 
                    for name, weight in sheet.items() 
                    if name not in already_picked
                }
                if not available:
                    break
                
                total_weight = sum(available.values())
                selected = self._select_weighted_item(available, total_weight)
                picked.append(selected)
                already_picked.add(selected)
        else:
            # Unweighted selection (simple random sample)
            available = [card for card in sheet if card not in already_picked]
            if available:
                picked = random.sample(available, min(count, len(available)))
                already_picked.update(picked)
        
        return picked
    
    def _select_weighted_item(
        self, 
        weighted_items: Dict[str, float], 
        total_weight: float
    ) -> str:
        """
        Select a random item based on weights.
        
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
    
    def clear_cache(self, set_code: Optional[str] = None):
        """
        Clear cached data for a specific set or all sets.
        
        Args:
            set_code: Set code to clear (None to clear all)
        """
        if set_code:
            set_code = set_code.upper()
            self._sheets_cache.pop(set_code, None)
            self._booster_config_cache.pop(set_code, None)
        else:
            self._sheets_cache.clear()
            self._booster_config_cache.clear()


# Module-level service instance for convenience
_booster_service = BoosterService()


def generate_booster(set_code: str) -> List[str]:
    """
    Generate a draft booster pack using MTGJson play booster config.
    
    Convenience function that uses a module-level BoosterService instance.
    
    Args:
        set_code: Set code (e.g., 'MH3', 'BLB')
        
    Returns:
        List of card names in the booster pack
        
    Raises:
        ValueError: If the set has no play booster configuration
        FileNotFoundError: If set data files are not found
    """
    return _booster_service.generate_booster(set_code)
