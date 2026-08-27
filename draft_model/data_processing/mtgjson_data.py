import requests
import json
import os

from config import (
    DATA_DIR,
    MTGJSON_BASE_URL,
    MTGJSON_CACHE_FILENAME,
    CARD_LIST_FILENAME,
    SHEETS_FILENAME,
    BOOSTER_CONFIG_FILENAME,
)


class MTGJson:
    def __init__(self):
        self._set_data_cache = {}

    def fetch_set_data(self, set_code):
        set_code = set_code.upper()

        if set_code in self._set_data_cache:
            return self._set_data_cache[set_code]

        set_dir = os.path.join(DATA_DIR, set_code)
        cache_file = os.path.join(set_dir, MTGJSON_CACHE_FILENAME)

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                set_data = json.load(f)
        else:
            print(f"Could not find {MTGJSON_CACHE_FILENAME} in {set_dir}, downloading now")
            set_data = self._download_and_cache_set_data(set_code, set_dir, cache_file)

        self._set_data_cache[set_code] = set_data
        return set_data

    def _download_and_cache_set_data(self, set_code, set_dir, cache_file):
        response = requests.get(f"{MTGJSON_BASE_URL}/{set_code}.json")
        response.raise_for_status()
        set_data = response.json()["data"]

        os.makedirs(set_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(set_data, f)

        return set_data

    def get_uuid_to_card_features(self, set_data):
        features = {}
        for card in set_data["cards"]:
            power = _safe_float(card.get("power"))
            toughness = _safe_float(card.get("toughness"))
            features[card["uuid"]] = {
                "name": card.get("faceName") or card["name"],
                "rarity": card.get("rarity", "common"),
                "mana_cost": card.get("manaCost", ""),
                "converted_mana_cost": card.get("manaValue", 0),
                "types": card.get("types", []),
                "subtypes": card.get("subtypes", []),
                "can_attack": power is not None,
                "power": power,
                "toughness": toughness,
                "oracle_text": card.get("text", "") or "",
            }
        return features

    def get_combined_uuid_lookup(self, set_code):
        set_data = self.fetch_set_data(set_code)
        source_set_codes = set_data["booster"]["play-arena"]["sourceSetCodes"]

        combined_features = self.get_uuid_to_card_features(set_data)
        for source_code in source_set_codes:
            if source_code != set_code:
                bonus_set_data = self.fetch_set_data(source_code)
                combined_features.update(self.get_uuid_to_card_features(bonus_set_data))

        return combined_features

    def build_sheets(self, set_code):
        set_data = self.fetch_set_data(set_code)
        play_arena_sheets = set_data["booster"]["play-arena"]["sheets"]
        uuid_to_card = self.get_combined_uuid_lookup(set_code)
        card_list = _load_card_list(set_code)

        sheets = {}
        for sheet_name, sheet_data in play_arena_sheets.items():
            cards = []
            for uuid, weight in sheet_data["cards"].items():
                card = uuid_to_card.get(uuid)
                if card is None:
                    continue  # unresolvable even with bonus sets (e.g. tokens)
                if card["name"] not in card_list:
                    continue  # never appears in our actual 17lands training data
                cards.append({"uuid": uuid, "name": card["name"], "weight": weight})
            sheets[sheet_name] = {"cards": cards}

        return sheets

    def build_booster_config(self, set_code):
        set_data = self.fetch_set_data(set_code)
        play_arena = set_data["booster"]["play-arena"]
        return {
            "boosters": play_arena["boosters"],
            "boostersTotalWeight": play_arena["boostersTotalWeight"],
        }

    def save_pack_data(self, set_code):
        set_dir = os.path.join(DATA_DIR, set_code)
        os.makedirs(set_dir, exist_ok=True)

        sheets = self.build_sheets(set_code)
        sheets_path = os.path.join(set_dir, SHEETS_FILENAME)
        with open(sheets_path, "w", encoding="utf-8") as f:
            json.dump(sheets, f, indent=2)

        booster_config = self.build_booster_config(set_code)
        booster_config_path = os.path.join(set_dir, BOOSTER_CONFIG_FILENAME)
        with open(booster_config_path, "w", encoding="utf-8") as f:
            json.dump(booster_config, f, indent=2)

        return sheets_path, booster_config_path


def _load_card_list(set_code):
    path = os.path.join(DATA_DIR, set_code, CARD_LIST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found:  run unpack_csv_to_card_list for {set_code} first"
        )
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    mtgjson = MTGJson()
    result = mtgjson.save_pack_data("MH3")
    print(result)
