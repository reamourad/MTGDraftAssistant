import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from ..card_encoder.card_encoder import CardEncoder
from ..external_api.config import DATA_DIR, CARD_LIST_FILENAME
from ..external_api.mtgjson_data import MTGJson
from ..model.sequence_builder import SequenceBuilder


class TrainingDataBuilder:
    def __init__(self, sequence_builder: SequenceBuilder):
        # sequence_builder is passed in (not built here) because it holds the
        # shared, trainable OracleTextProjection — same reasoning as everywhere
        # else this session: trainable things get built once, externally, and
        # handed to whatever needs them.
        self.mtgjson = MTGJson()
        self.card_encoder = CardEncoder()
        self.sequence_builder = sequence_builder

#takes one pick (pack number, pick number, pack cards, picked card and pool) and encodes it accordingly (ex: 14 bad example, 1 bad one)
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

    def get_name_to_features(self, set_code: str) -> Dict[str, dict]:
        uuid_to_features = self.mtgjson.get_combined_uuid_lookup(set_code)
        name_to_features = {features['name']: features for features in uuid_to_features.values()}

        card_list = self.unpack_csv_to_card_list(set_code)
        unresolved = sorted(name for name in card_list if name not in name_to_features)

        if unresolved:
            raise ValueError(
                f"{len(unresolved)} card(s) in {set_code}'s card list did not resolve "
                f"to features: {unresolved}"
            )

        return name_to_features

    #get all the info about one draft, iterable
    def get_draft_pick_sequences(self, set_code: str) -> Iterator[Tuple[str, List[dict]]]:
        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            raise FileNotFoundError(f"No .csv.gz found for set {set_code} in {DATA_DIR}/{set_code}")

        #only get the pack columns to read + some extra like pick number
        pack_cols = [c for c in pd.read_csv(csv_path, nrows=0).columns if c.startswith('pack_card_')]
        usecols = ['draft_id', 'pack_number', 'pick_number', 'pick', 'event_match_wins'] + pack_cols
        df = pd.read_csv(csv_path, usecols=usecols)

        # 7-win filter, computed inline from this same read — a second call to
        # get_seven_win_draft_ids would mean reading this whole CSV a second time.
        wins_per_draft = df.groupby('draft_id')['event_match_wins'].max()
        seven_win_ids = set(wins_per_draft[wins_per_draft == 7].index)
        df = df[df['draft_id'].isin(seven_win_ids)]

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
        """
        Given one pool-enriched pick, builds one training example per card in the
        pack: positive (label=1) for the card actually chosen, negative (label=0)
        for every other card. Each example's 'other_pack_cards' is that pack minus
        only the ONE specific occurrence used as candidate in THIS example (by
        position, not by name) — a pack can genuinely hold 2+ copies of the same
        card, and excluding by name would wrongly hide every copy, not just this one.
        Weights are class-balanced: total positive weight (1) always equals total
        negative weight — split evenly if there's more than one positive occurrence
        (e.g. two copies of the card that got picked).
        """
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
        """
        Why: the pack-generation pipeline (sheets.json) needs to know which cards
        actually appear in real 17lands data, to filter out MTGJSON cards that
        technically exist but never show up in a real pack — this builds that
        reference list, once, from the CSV's own column names.
        """
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
        with open(file_name, 'w', newline='') as jsonfile:
            json.dump(sorted(card_names), jsonfile)

        return card_names

    def get_seven_win_draft_ids(self, set_code: str) -> Set[str]:
        """
        Why: the original, standalone version of the 7-win filter — written
        before get_draft_pick_sequences existed. That method now recomputes this
        same filter inline (to avoid reading the CSV twice), so this is mostly
        redundant for the main pipeline now — kept as a lightweight, standalone
        check for when you just need the draft IDs and nothing else.
        """
        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            raise FileNotFoundError(f"No .csv.gz found for set {set_code} in {DATA_DIR}/{set_code}")

        df = pd.read_csv(csv_path, usecols=['draft_id', 'event_match_wins'])
        per_draft = df.groupby('draft_id')['event_match_wins'].max()
        return set(per_draft[per_draft == 7].index)

    def add_pool_history(self, picks: List[dict]) -> List[dict]:
        """
        Why: kept separate from get_draft_pick_sequences (which calls this
        internally) so pool-tracking could be tested and understood on its own,
        apart from all the CSV-reading logic.

        Given one draft's ordered picks, adds a 'pool' key to each pick — the
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
        """
        Why: every method above needs to find the set's CSV, and the actual
        filenames are inconsistent (e.g. Powered_Cube's is named after "Cube_-_Powered",
        not the folder name) — centralizing the glob logic here means it's only
        written once, not copy-pasted into every method that needs it.
        """
        set_dir = os.path.join(DATA_DIR, set_code)
        matches = list(Path(set_dir).glob("*.csv.gz"))
        if not matches:
            return None
        return matches[0]
