import json
import os
import random

from ..data_processing.config import DATA_DIR, CARD_LIST_FILENAME, BOOSTER_CONFIG_FILENAME, SHEETS_FILENAME


class PackGenerator:
    def __init__(self):
        pass

    def generate(self, set_code):
        #pick the card distribution
        pack_distribution = self.pick_card_distribution(set_code)
        print(pack_distribution)

        picked_cards = []
        for sheet_name, amount in pack_distribution.items():
            cards = self.pick_card_from_sheet(set_code, sheet_name, amount)
            picked_cards.extend(cards)

        return picked_cards

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
        picked_names = []

        for i in range(amount):
            if not remaining_cards:
                raise ValueError(
                    f"Asked for {amount} cards from '{sheet_name}' but only "
                    f"{len(picked_names)} were available after filtering"
                )

            #calculate total weight, choose a random number from that weight and then go to that number
            total_weight = sum(card["weight"] for card in remaining_cards)
            random_weight = random.randint(1, total_weight)

            #once the card is found add it to picked_card to save it and then remove it from remaining cards so we don't pick the same card twice
            current_weight = 0
            for index, card in enumerate(remaining_cards):
                current_weight += card["weight"]
                if current_weight >= random_weight:
                    picked_names.append(card["name"])
                    remaining_cards.pop(index)
                    break

        return picked_names




if __name__ == '__main__':
    pack_generator = PackGenerator()
    pack = pack_generator.generate(set_code="MH3")
    print(pack)