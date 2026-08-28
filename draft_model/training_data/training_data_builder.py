import json
import os
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import pandas as pd

from ..external_api.config import DATA_DIR, CARD_LIST_FILENAME


class TrainingDataBuilder:
    def get_draft_pick_sequences(self, set_code: str) -> Iterator[Tuple[str, List[dict]]]:
        """
        Reads the set's CSV exactly once, filters to 7-win drafts, and yields
        (draft_id, ordered_picks) for each one — ordered_picks is a list of
        {pack_number, pick_number, pack_cards, picked_card} dicts, in draft order.
        """
        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            raise FileNotFoundError(f"No .csv.gz found for set {set_code} in {DATA_DIR}/{set_code}")

        pack_cols = [c for c in pd.read_csv(csv_path, nrows=0).columns if c.startswith('pack_card_')]
        usecols = ['draft_id', 'pack_number', 'pick_number', 'pick', 'event_match_wins'] + pack_cols
        df = pd.read_csv(csv_path, usecols=usecols)

        # 7-win filter, computed inline from this same read — a second call to
        # get_seven_win_draft_ids would mean reading this whole CSV a second time.
        wins_per_draft = df.groupby('draft_id')['event_match_wins'].max()
        seven_win_ids = set(wins_per_draft[wins_per_draft == 7].index)
        df = df[df['draft_id'].isin(seven_win_ids)]

        df = df.sort_values(['draft_id', 'pack_number', 'pick_number'])

        for draft_id, draft_rows in df.groupby('draft_id', sort=False):
            picks = []
            for _, row in draft_rows.iterrows():
                pack_cards = [col.replace('pack_card_', '') for col in pack_cols if row[col] == 1]
                picks.append({
                    'pack_number': row['pack_number'],
                    'pick_number': row['pick_number'],
                    'pack_cards': pack_cards,
                    'picked_card': row['pick'],
                })
            yield draft_id, picks

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
        with open(file_name, 'w', newline='') as jsonfile:
            json.dump(sorted(card_names), jsonfile)

        return card_names

    def get_seven_win_draft_ids(self, set_code: str) -> Set[str]:
        csv_path = self._find_csv_path(set_code)
        if csv_path is None:
            raise FileNotFoundError(f"No .csv.gz found for set {set_code} in {DATA_DIR}/{set_code}")

        df = pd.read_csv(csv_path, usecols=['draft_id', 'event_match_wins'])
        per_draft = df.groupby('draft_id')['event_match_wins'].max()
        return set(per_draft[per_draft == 7].index)


    def _find_csv_path(self, set_code: str) -> Optional[Path]:
        set_dir = os.path.join(DATA_DIR, set_code)
        matches = list(Path(set_dir).glob("*.csv.gz"))
        if not matches:
            return None
        return matches[0]
