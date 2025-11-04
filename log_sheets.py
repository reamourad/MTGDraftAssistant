"""Log sheet contents for debugging booster generation."""
import json
import sys
from app.booster.generator import load_training_card_names, build_filtered_sheets

def log_sheets(set_code="MH3", show_cards=False):
    """
    Log the filtered sheets for a set.

    Args:
        set_code: Set code (e.g., 'MH3')
        show_cards: If True, show all card names in each sheet
    """
    # Load data
    with open(f"app/models/{set_code}/cards.json", 'r', encoding='utf-8') as f:
        cards = json.load(f)

    training_cards = load_training_card_names(set_code)
    sheets = build_filtered_sheets(cards, training_cards)

    print(f"=== Sheet Sizes for {set_code} ===\n")
    for sheet_name, card_list in sheets.items():
        print(f"{sheet_name.upper()}: {len(card_list)} cards")

    if show_cards:
        print(f"\n=== Sheet Contents ===\n")
        for sheet_name, card_list in sheets.items():
            print(f"\n{sheet_name.upper()}:")
            for card in sorted(set(card_list)):  # Remove duplicates for display
                count = card_list.count(card)
                if count > 1:
                    print(f"  - {card} (x{count})")
                else:
                    print(f"  - {card}")

    # Check for basic lands
    basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
    print("\n=== Basic Lands Check ===")
    for sheet_name, card_list in sheets.items():
        basics = [c for c in card_list if c in basic_lands]
        if basics:
            print(f"{sheet_name}: {len(basics)} basics - {set(basics)}")
        else:
            print(f"{sheet_name}: No basic lands [OK]")

if __name__ == "__main__":
    show_cards = "--cards" in sys.argv or "-c" in sys.argv

    # Get set code (ignore flags)
    set_code = "MH3"
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            set_code = arg
            break

    log_sheets(set_code, show_cards)
