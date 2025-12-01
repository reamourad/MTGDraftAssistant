"""
Add a new MTG set to the draft assistant.

Fetches card data and booster configuration from MTGJson API.
Use this for sets that don't have 17lands data yet.

Usage:
    python add_set.py <SET_CODE>

Example:
    python add_set.py FDN
"""
import sys
import os
from app.booster.mtgjson_fetcher import save_booster_data


def main():
    if len(sys.argv) < 2:
        print("Error: Missing set code argument")
        print("\nUsage:")
        print("  python add_set.py <SET_CODE>")
        print("\nExample:")
        print("  python add_set.py FDN")
        sys.exit(1)

    set_code = sys.argv[1].upper()
    output_dir = f"app/models/{set_code}"

    print(f"\n{'='*60}")
    print(f"Adding new set: {set_code}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    try:
        # Fetch and save MTGJson data
        save_booster_data(set_code, output_dir)

        print(f"\n{'='*60}")
        print(f"✓ Successfully added {set_code}!")
        print(f"{'='*60}")
        print(f"\nCreated files:")
        print(f"  - {output_dir}/cards.json")
        print(f"  - {output_dir}/booster_config.json")
        print(f"\nThe booster generator can now create packs for {set_code}.")
        print(f"\nTo train a model (when 17lands data is available):")
        print(f"  python train_model.py {set_code}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error adding {set_code}")
        print(f"{'='*60}")
        print(f"\nError: {e}")
        print(f"\nMake sure the set code is valid and exists on MTGJson.")
        print(f"Check: https://mtgjson.com/api/v5/{set_code}.json")
        sys.exit(1)


if __name__ == '__main__':
    main()
