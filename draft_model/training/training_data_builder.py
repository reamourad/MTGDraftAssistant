import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import torch

from ..card_encoder.card_encoder import CardEncoder
from ..external_api.config import DATA_DIR, CARD_LIST_FILENAME, FILTERED_DRAFT_DATA_FILENAME
from ..external_api.mtgjson_data import MTGJson
from ..model.sequence_builder import SequenceBuilder


def group_fold_by_set(fold: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    draft_ids_by_set: Dict[str, Set[str]] = {}
    for set_code, draft_id in fold:
        draft_ids_by_set.setdefault(set_code, set()).add(draft_id)
    return draft_ids_by_set


def _read_parquet_cache(path: Path, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Returns None on a missing OR corrupted parquet cache (self-healing against
    the concurrent-write race — multiple parallel training folds can each hit
    a cold cache for the same set at once), so callers always fall back to
    rebuilding it rather than crashing on a half-written file.
    """
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        print(f"[TrainingDataBuilder] {path} is corrupted (likely a concurrent-write race), rebuilding it")
        return None


def _placeholder_card_features(name: str) -> dict:
    return {
        "name": name,
        "rarity": "common",
        "mana_cost": "",
        "converted_mana_cost": 0,
        "types": [],
        "subtypes": [],
        "can_attack": False,
        "power": None,
        "toughness": None,
        "oracle_text": "",
    }


# manually transcribed from Wizards' official Gatherer — Through the Omenpaths (OM1)
# is MTG Arena-exclusive, released after this project's MTGJSON/Scryfall snapshots,
# so neither has these cards under any name variant. Confirmed via direct Gatherer
# lookup, not guessed.
MANUALLY_CURATED_CARDS = {
    "Ademi of the Silkchutes": {
        "rarity": "rare",
        "mana_cost": "{1}{W}",
        "converted_mana_cost": 2,
        "types": ["Creature"],
        "subtypes": ["Spider", "Human", "Hero"],
        "can_attack": True,
        "power": 3,
        "toughness": 2,
        "oracle_text": "Flash\n{1}: Ademi gains flying until end of turn.\n{1}, Sacrifice Ademi: "
                        "Creatures you control gain hexproof and indestructible until end of turn.",
    },
    "Goben, Gene-Splice Savant": {
        "rarity": "mythic",
        "mana_cost": "{1}{U}",
        "converted_mana_cost": 2,
        "types": ["Creature"],
        "subtypes": ["Human", "Scientist", "Villain"],
        "can_attack": True,
        "power": 1,
        "toughness": 1,
        "oracle_text": "Goben can't be blocked. Whenever Goben deals combat damage to a player, he connives.\n"
                        "{1}{U}{B}{R}: Transform Goben. Activate only as a sorcery.",
    },
    "Luis, Pompous Pillager": {
        "rarity": "rare",
        "mana_cost": "{X}{B}{B}",
        "converted_mana_cost": 2,
        "types": ["Creature"],
        "subtypes": ["Vampire", "Villain"],
        "can_attack": True,
        "power": 2,
        "toughness": 1,
        "oracle_text": "Lifelink\nLuis enters with X +1/+1 counters on him.\n"
                        "When Luis enters, he deals X damage to target opponent.",
    },
    "Makdee and Itla, Skysnarers": {
        "rarity": "rare",
        "mana_cost": "{1}{W/U}",
        "converted_mana_cost": 2,
        "types": ["Creature"],
        "subtypes": ["Spider", "Human", "Hero"],
        "can_attack": True,
        "power": 2,
        "toughness": 2,
        "oracle_text": "Flying\nArtifacts and creatures your opponents control enter tapped.",
    },
    "Nia, Skysail Storyteller": {
        "rarity": "mythic",
        "mana_cost": "{1}{R}",
        "converted_mana_cost": 2,
        "types": ["Creature"],
        "subtypes": ["Human", "Performer", "Hero"],
        "can_attack": True,
        "power": 2,
        "toughness": 1,
        "oracle_text": "When Nia enters, exile the top card of your library. "
                        "You may play that card for as long as you control this creature.\n"
                        "{2}: Transform Nia. Activate only as a sorcery.",
    },
    "Yera and Oski, Weaver and Guide": {
        "rarity": "rare",
        "mana_cost": "{2}{W}",
        "converted_mana_cost": 3,
        "types": ["Creature"],
        "subtypes": ["Spider", "Human", "Hero"],
        "can_attack": True,
        "power": 3,
        "toughness": 3,
        "oracle_text": "Enweb (You may cast this spell for its enweb cost if you also return a tapped "
                        "creature you control to its owner's hand.)\n"
                        "As Yera and Oski enters, look at an opponent's hand, then choose a card type "
                        "other than creature.\nSpells of the chosen type cost {1} more to cast.",
    },
}


def _get_manually_curated_features(name: str) -> Optional[dict]:
    data = MANUALLY_CURATED_CARDS.get(name)
    if data is None:
        return None
    return {"name": name, **data}


class TrainingDataBuilder:
    def __init__(self, sequence_builder: SequenceBuilder):
        self.mtgjson = MTGJson()
        self.card_encoder = CardEncoder()
        self.sequence_builder = sequence_builder

    # takes one pick (pack number, pick number, pack cards, picked card and pool)
    # and encodes it accordingly (ex: 14 bad example, 1 bad one)
    def encode_pick(
        self,
        pick: dict,
        set_cards: List[np.ndarray],
        name_to_features: Dict[str, dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int, float]]:
        #Get all the example possible out of the pick (candidate, other_pack_cards, pool, label, weight)
        examples = self.build_training_examples(pick)

        #encode the pool since all the example share the same pool
        pool_vectors = [self.card_encoder.encode(name_to_features[name]) for name in pick['pool']]

        #for each example, encode the candidate and all the other card in the pack and build the final sequence to send to the model as input
        encoded_examples = []
        for example in examples:
            candidate_vector = self.card_encoder.encode(name_to_features[example['candidate']])
            other_pack_vectors = [
                self.card_encoder.encode(name_to_features[name]) for name in example['other_pack_cards']
            ]

            sequence, mask = self.sequence_builder.build_full_sequence(
                set_cards, pool_vectors, candidate_vector, other_pack_vectors
            )

            #for each draft possibility, we are keeping the result either as a good example (label = 1) or a bad one,
            encoded_examples.append((sequence, mask, example['label'], example['weight']))

        return encoded_examples

    def encode_set_cards(self, set_code: str, name_to_features: Dict[str, dict]) -> List[np.ndarray]:
        start = time.perf_counter()
        card_list = self.unpack_csv_to_card_list(set_code)
        cards = [name_to_features[name] for name in card_list]
        encoded = list(self.card_encoder.encode_batch(cards))
        print(f"[TrainingDataBuilder] encoded {len(encoded)} set cards for {set_code} "
              f"in {time.perf_counter() - start:.2f}s")
        return encoded

    def get_name_to_features(self, set_code: str) -> Dict[str, dict]:
        card_list = self.unpack_csv_to_card_list(set_code)

        try:
            uuid_to_features = self.mtgjson.get_combined_uuid_lookup(set_code)
            name_to_features = {features['name']: features for features in uuid_to_features.values()}
        except requests.exceptions.HTTPError:
            # no real MTGJSON set page for this code (e.g. a 17lands Cube draft,
            # not an official Magic set) — resolve cards by name instead
            name_to_features = self.mtgjson.get_name_to_features_from_atomic(card_list)

            for name in card_list:
                if name not in name_to_features:
                    curated = _get_manually_curated_features(name)
                    if curated is not None:
                        name_to_features[name] = curated

            # even MTGJSON's full card database occasionally lags behind the very
            # newest cards — fill any remaining gaps with neutral placeholders
            # rather than blocking an entire cube over a handful of cards
            unresolved = sorted(name for name in card_list if name not in name_to_features)
            if unresolved:
                print(f"[TrainingDataBuilder] {len(unresolved)} card(s) in {set_code} not found even in "
                      f"MTGJSON's card database (likely too new) — using placeholder features: {unresolved}")
                for name in unresolved:
                    name_to_features[name] = _placeholder_card_features(name)

            return name_to_features

        unresolved = sorted(name for name in card_list if name not in name_to_features)

        if unresolved:
            raise ValueError(
                f"{len(unresolved)} card(s) in {set_code}'s card list did not resolve "
                f"to features: {unresolved}"
            )

        return name_to_features

    #get all the info about one draft, iterable
    def get_draft_pick_sequences(self, set_code: str, draft_ids: Optional[Set[str]] = None) -> Iterator[Tuple[str, List[dict]]]:
        df = self._load_seven_win_df(set_code)
        pack_cols = [c for c in df.columns if c.startswith('pack_card_')]

        if draft_ids is not None:
            df = df[df['draft_id'].isin(draft_ids)]

        #sort first by draft ids, then we check the pack number and the pick number to make sure its accurately sorted in order
        #if its not sorted in order it could "spoil" the model of which card to pick
        df = df.sort_values(['draft_id', 'pack_number', 'pick_number'])

        #for each draft, create a cleaned package to send to training with the pack and pick number, the pack cards and the picked card
        for draft_id, draft_rows in df.groupby('draft_id', sort=False):
            picks = []
            for _, row in draft_rows.iterrows():
                # pack_card_* values are COUNTS, not just 0/1 presence flags — a pack can
                # genuinely contain 2+ copies of the same card, so we repeat the name that
                # many times rather than checking == 1.
                pack_cards = []
                for col in pack_cols:
                    count = int(row[col])
                    if count > 0:
                        pack_cards.extend([col.replace('pack_card_', '')] * count)
                picks.append({
                    'pack_number': row['pack_number'],
                    'pick_number': row['pick_number'],
                    'pack_cards': pack_cards,
                    'picked_card': row['pick'],
                })
            yield draft_id, self.add_pool_history(picks)

    #this function makes the one pick become multiple training example, one for the positive card, and then the rest are negative examples
    def build_training_examples(self, pick: dict) -> List[dict]:
        pack_cards = pick['pack_cards']
        picked_card = pick['picked_card']

        num_positives = pack_cards.count(picked_card)
        num_negatives = len(pack_cards) - num_positives
        positive_weight = 1.0 / num_positives
        negative_weight = 1.0 / num_negatives if num_negatives > 0 else 0

        examples = []
        for i, candidate in enumerate(pack_cards):
            is_picked = candidate == picked_card
            examples.append({
                'candidate': candidate,
                'other_pack_cards': pack_cards[:i] + pack_cards[i + 1:],
                'pool': pick['pool'],
                'label': 1 if is_picked else 0,
                'weight': positive_weight if is_picked else negative_weight,
            })

        return examples

    def unpack_csv_to_card_list(self, set_code: str) -> Optional[Set[str]]:
        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            return None

        column_names = pd.read_csv(csv_path, nrows=0).columns

        card_names = set()
        for col in column_names:
            if col.startswith('pack_card_'):
                card_names.add(col.replace('pack_card_', ''))

        set_dir = os.path.join(DATA_DIR, set_code)
        file_name = os.path.join(set_dir, CARD_LIST_FILENAME)
        tmp_path = f"{file_name}.tmp.{os.getpid()}"
        with open(tmp_path, 'w', newline='') as jsonfile:
            json.dump(sorted(card_names), jsonfile)
        os.replace(tmp_path, file_name)

        return card_names

    def get_seven_win_draft_ids(self, set_code: str) -> Set[str]:
        cached = _read_parquet_cache(self._find_parquet_path(set_code), columns=['draft_id'])
        if cached is not None:
            return set(cached['draft_id'].unique())

        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            raise FileNotFoundError(f"No .csv.gz found for set {set_code} in {DATA_DIR}/{set_code}")

        df = pd.read_csv(csv_path, usecols=['draft_id', 'event_match_wins'])
        per_draft = df.groupby('draft_id')['event_match_wins'].max()
        return set(per_draft[per_draft == 7].index)

    def _load_seven_win_df(self, set_code: str) -> pd.DataFrame:
        parquet_path = self._find_parquet_path(set_code)
        start = time.perf_counter()
        cached = _read_parquet_cache(parquet_path)
        if cached is not None:
            print(f"[TrainingDataBuilder] loaded cached parquet for {set_code} "
                  f"({len(cached)} rows) in {time.perf_counter() - start:.2f}s")
            return cached

        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            raise FileNotFoundError(f"No .csv.gz found for set {set_code} in {DATA_DIR}/{set_code}")

        start = time.perf_counter()
        seven_win_ids = self.get_seven_win_draft_ids(set_code)

        pack_cols = [c for c in pd.read_csv(csv_path, nrows=0).columns if c.startswith('pack_card_')]
        usecols = ['draft_id', 'pack_number', 'pick_number', 'pick', 'event_match_wins'] + pack_cols
        dtype = {col: 'uint8' for col in pack_cols}

        chunks = []
        for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtype, chunksize=200_000):
            chunk = chunk[chunk['draft_id'].isin(seven_win_ids)]
            if not chunk.empty:
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = parquet_path.with_name(parquet_path.name + f".tmp.{os.getpid()}")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, parquet_path)
        print(f"[TrainingDataBuilder] built and cached parquet for {set_code} "
              f"({len(df)} rows) in {time.perf_counter() - start:.2f}s")
        return df

    def _find_parquet_path(self, set_code: str) -> Path:
        return Path(DATA_DIR) / set_code / FILTERED_DRAFT_DATA_FILENAME

    def add_pool_history(self, picks: List[dict]) -> List[dict]:
        """
        Given one draft's ordered picks, adds a 'pool' key to each pick, the
        cards already picked BEFORE that pick happened.
        """
        pool = []
        enriched_picks = []

        for pick in picks:
            enriched_pick = dict(pick)
            enriched_pick['pool'] = list(pool)  # snapshot — pool keeps growing after this
            enriched_picks.append(enriched_pick)
            pool.append(pick['picked_card'])

        return enriched_picks

    def _find_csv_path(self, set_code: str) -> Optional[Path]:
        set_dir = os.path.join(DATA_DIR, set_code)
        matches = list(Path(set_dir).glob("*.csv.gz"))
        if not matches:
            return None
        return matches[0]
