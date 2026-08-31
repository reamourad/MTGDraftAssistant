import requests
import json
import os
import time

from .config import (
    DATA_DIR,
    MTGJSON_BASE_URL,
    MTGJSON_CACHE_FILENAME,
    CARD_LIST_FILENAME,
    SHEETS_FILENAME,
    BOOSTER_CONFIG_FILENAME,
    ATOMIC_CARDS_CACHE_FILENAME,
)


class MTGJson:
    def __init__(self):
        self._set_data_cache = {}
        self._atomic_cards_cache = None

    def fetch_set_data(self, set_code):
        set_code = set_code.upper()

        if set_code in self._set_data_cache:
            return self._set_data_cache[set_code]

        set_dir = os.path.join(DATA_DIR, set_code)
        cache_file = os.path.join(set_dir, MTGJSON_CACHE_FILENAME)

        set_data = _read_json_cache(cache_file)
        if set_data is None:
            print(f"Could not find {MTGJSON_CACHE_FILENAME} in {set_dir}, downloading now")
            set_data = self._download_and_cache_set_data(set_code, set_dir, cache_file)

        self._set_data_cache[set_code] = set_data
        return set_data

    def _download_and_cache_set_data(self, set_code, set_dir, cache_file):
        start = time.perf_counter()
        response = requests.get(f"{MTGJSON_BASE_URL}/{set_code}.json")
        response.raise_for_status()
        set_data = response.json()["data"]

        os.makedirs(set_dir, exist_ok=True)
        _atomic_write_json(cache_file, set_data)

        print(f"[MTGJson] downloaded and cached {set_code} in {time.perf_counter() - start:.2f}s")
        return set_data

    def get_uuid_to_card_features(self, set_data):
        features = {}
        for card in set_data["cards"]:
            features[card["uuid"]] = _build_card_features(card, card.get("faceName") or card["name"])
        return features

    def fetch_atomic_cards(self):
        if self._atomic_cards_cache is not None:
            return self._atomic_cards_cache

        cache_file = os.path.join(DATA_DIR, ATOMIC_CARDS_CACHE_FILENAME)
        atomic_data = _read_json_cache(cache_file)
        if atomic_data is None:
            print(f"Could not find {ATOMIC_CARDS_CACHE_FILENAME}, downloading now (~160MB, one-time)")
            start = time.perf_counter()
            response = requests.get(f"{MTGJSON_BASE_URL}/AtomicCards.json")
            response.raise_for_status()
            atomic_data = response.json()["data"]

            os.makedirs(DATA_DIR, exist_ok=True)
            _atomic_write_json(cache_file, atomic_data)
            print(f"[MTGJson] downloaded and cached AtomicCards in {time.perf_counter() - start:.2f}s")

        self._atomic_cards_cache = atomic_data
        return atomic_data

    def get_name_to_features_from_atomic(self, names):
        """
        Resolves cards by NAME against MTGJSON's name-deduplicated card database,
        for products with no real MTGJSON set page (e.g. 17lands Cube drafts) —
        no uuid/booster/sourceSetCodes path exists for those. Atomic data has no
        'rarity' field (it's printing-specific); _build_card_features already
        defaults missing rarity to 'common', same as the normal set-based path.

        Top-level keys are the COMBINED name for modal double-faced/split/adventure
        cards (e.g. "Bonecrusher Giant // Stomp"). 17lands references some of
        these by the individual face name (each entry's own 'faceName' has
        that, same convention as the uuid-based path) and others — old-style
        split cards like "Life // Death" — by the combined name instead. So we
        index by both.
        """
        atomic_data = self.fetch_atomic_cards()
        wanted = set(names)
        name_to_features = {}
        for entries in atomic_data.values():
            for card in entries:
                for candidate_name in (card.get("faceName"), card.get("name")):
                    if candidate_name and candidate_name in wanted and candidate_name not in name_to_features:
                        name_to_features[candidate_name] = _build_card_features(card, candidate_name)
        return name_to_features

    def get_arena_booster(self, set_data, set_code):
        booster = set_data.get("booster", {})
        for key in ("play-arena", "arena"):
            if key in booster:
                return booster[key]
        raise KeyError(
            f"{set_code} has no 'play-arena' or 'arena' booster config "
            f"(available: {list(booster.keys())})"
        )

    def get_combined_uuid_lookup(self, set_code):
        set_data = self.fetch_set_data(set_code)
        source_set_codes = self.get_arena_booster(set_data, set_code)["sourceSetCodes"]

        combined_features = self.get_uuid_to_card_features(set_data)
        for source_code in source_set_codes:
            if source_code != set_code:
                bonus_set_data = self.fetch_set_data(source_code)
                combined_features.update(self.get_uuid_to_card_features(bonus_set_data))

        return combined_features

    def build_sheets(self, set_code):
        set_data = self.fetch_set_data(set_code)
        play_arena_sheets = self.get_arena_booster(set_data, set_code)["sheets"]
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
        play_arena = self.get_arena_booster(set_data, set_code)
        return {
            "boosters": play_arena["boosters"],
            "boostersTotalWeight": play_arena["boostersTotalWeight"],
        }

    def save_pack_data(self, set_code):
        set_dir = os.path.join(DATA_DIR, set_code)
        os.makedirs(set_dir, exist_ok=True)

        sheets = self.build_sheets(set_code)
        sheets_path = os.path.join(set_dir, SHEETS_FILENAME)
        _atomic_write_json(sheets_path, sheets, indent=2)

        booster_config = self.build_booster_config(set_code)
        booster_config_path = os.path.join(set_dir, BOOSTER_CONFIG_FILENAME)
        _atomic_write_json(booster_config_path, booster_config, indent=2)

        return sheets_path, booster_config_path


def _read_json_cache(path):
    """
    Returns None on a missing OR corrupted cache file (self-healing against
    the concurrent-write race below — multiple parallel training folds can
    each hit a cold cache for the same file at once), so callers always fall
    back to rebuilding it rather than crashing on a half-written file.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[MTGJson] {path} is corrupted (likely a concurrent-write race), rebuilding it")
        return None


def _atomic_write_json(path, data, **kwargs):
    """
    Writes to a pid-tagged temp file then renames over the real path — rename
    is atomic, so concurrent writers from parallel training folds never leave
    a reader looking at a half-written file (which is what corrupts a naive
    'open(path, "w")' when two processes write the same path at once).
    """
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)
    os.replace(tmp_path, path)


def _build_card_features(card, name):
    power = _safe_float(card.get("power"))
    toughness = _safe_float(card.get("toughness"))
    return {
        "name": name,
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
