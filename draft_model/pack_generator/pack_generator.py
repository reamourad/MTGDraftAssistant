import json
import os
import random

from ..external_api.config import DATA_DIR, CARD_LIST_FILENAME, BOOSTER_CONFIG_FILENAME, SHEETS_FILENAME
from ..external_api.mtgjson_data import MTGJson

RARITY_ORDER = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}

# Cube "sets" aren't real MTGJSON sets, so they have no rarity-weighted
# booster sheets to draw from. They're drafted as a flat, unweighted random
# sample from the fixed cube list instead — see generate_cube_pack.
CUBE_SETS = {"POWERED_CUBE"}
CUBE_PACK_SIZE = 15


class PackGenerator:
    def __init__(self):
        self.mtgjson = MTGJson()

    def generate(self, set_code):
        # match CUBE_SETS case-insensitively, but keep set_code's original
        # casing for filesystem paths — Powered_Cube's on-disk directory
        # isn't all-uppercase, unlike real MTGJSON set codes
        if set_code.upper() in CUBE_SETS:
            return self.generate_cube_pack(set_code)

        #pick the card distribution
        pack_distribution = self.pick_card_distribution(set_code)
        print(pack_distribution)

        picked_cards = []
        for sheet_name, amount in pack_distribution.items():
            cards = self.pick_card_from_sheet(set_code, sheet_name, amount)
            picked_cards.extend(cards)

        return self.sort_pack_by_rarity(set_code, picked_cards)

    def generate_cube_pack(self, set_code, pack_size=CUBE_PACK_SIZE):
        set_dir = os.path.join(DATA_DIR, set_code)
        card_list_path = os.path.join(set_dir, CARD_LIST_FILENAME)

        if not os.path.exists(card_list_path):
            raise FileNotFoundError(
                f"{card_list_path} not found, run unpack_csv_to_card_list for {set_code} first"
            )

        with open(card_list_path, "r", encoding="utf-8") as f:
            card_list = json.load(f)

        if len(card_list) < pack_size:
            raise ValueError(
                f"Cube {set_code} only has {len(card_list)} cards, need at least {pack_size}"
            )

        # unweighted, no-replacement draw — a cube has no rarity sheets,
        # every card in the list is equally likely
        names = random.sample(card_list, pack_size)
        return [{"name": name, "uuid": None} for name in names]

    def pick_card_distribution(self, set_code):
        #read the json of card distribution
        set_dir = os.path.join(DATA_DIR, set_code)
        cache_file = os.path.join(set_dir, BOOSTER_CONFIG_FILENAME)

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                booster_config = json.load(f)
        else:
            raise FileNotFoundError(
                f"{cache_file} not found, run save_pack_data for {set_code} first"
            )


        #check the total weight, choose a random number from that weight and then go to that number
        total_weight = booster_config["boostersTotalWeight"]
        random_weight = random.randint(1, total_weight)

        current_weight = 0
        for booster in booster_config["boosters"]:
            current_weight += booster["weight"]
            if current_weight >= random_weight:
                return booster["contents"]


        print("Something went wrong with choosing a booster config, default to first booster content")
        return booster_config["boosters"][0]["contents"]

    def pick_card_from_sheet(self, set_code, sheet_name, amount):
        #with set code and sheet name, get the relevant card list
        set_dir = os.path.join(DATA_DIR, set_code)
        sheets_file = os.path.join(set_dir, SHEETS_FILENAME)

        if os.path.exists(sheets_file):
            with open(sheets_file, "r", encoding="utf-8") as f:
                sheets = json.load(f)
        else:
            raise FileNotFoundError(
                f"{sheets_file} not found, run save_pack_data for {set_code} first"
            )

        if sheet_name not in sheets:
            raise ValueError(f"'{sheet_name}' is not a sheet in {sheets_file}")

        #create a pool so we don't pick the same card twice
        remaining_cards = list(sheets[sheet_name]["cards"])
        picked_cards = []

        for i in range(amount):
            if not remaining_cards:
                raise ValueError(
                    f"Asked for {amount} cards from '{sheet_name}' but only "
                    f"{len(picked_cards)} were available after filtering"
                )

            #calculate total weight, choose a random number from that weight and then go to that number
            total_weight = sum(card["weight"] for card in remaining_cards)
            random_weight = random.randint(1, total_weight)

            #once the card is found add it to picked_cards to save it and then remove it from remaining cards so we don't pick the same card twice
            current_weight = 0
            for index, card in enumerate(remaining_cards):
                current_weight += card["weight"]
                if current_weight >= random_weight:
                    picked_cards.append({"name": card["name"], "uuid": card["uuid"]})
                    remaining_cards.pop(index)
                    break

        return picked_cards

    def sort_pack_by_rarity(self, set_code, pack):
        uuid_to_card = self.mtgjson.get_combined_uuid_lookup(set_code)

        def rarity_rank(picked_card):
            rarity = uuid_to_card[picked_card["uuid"]]["rarity"]
            return RARITY_ORDER.get(rarity, len(RARITY_ORDER))

        return sorted(pack, key=rarity_rank)




if __name__ == '__main__':
    pack_generator = PackGenerator()
    pack = pack_generator.generate(set_code="MH3")
    print(pack)
    for picked_card in pack:
        print(picked_card["name"])