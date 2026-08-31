import random

from sklearn.model_selection import KFold
from typing import List, Optional, Tuple

from .training_data_builder import TrainingDataBuilder


class DraftSplitter:
    def __init__(self, training_data_builder: TrainingDataBuilder):
        self.training_data_builder = training_data_builder

    #Chooses the set we will do validation on
    def choose_held_out_set(self, set_codes: List[str]) -> Tuple[str, List[str]]:
        held_out = random.choice(set_codes)
        remaining = [s for s in set_codes if s != held_out]
        return held_out, remaining


    def get_draft_folds(self, set_codes: List[str], k: int, max_drafts: Optional[int] = None) -> List[List[Tuple[str, str]]]:
        """
        splits real drafts into k folds for cross-validation, at the DRAFT
        level. every pick from the same draft always lands in the same fold.
        """
        draft_ids = []
        for set_code in set_codes:
            set_draft_ids = list(self.training_data_builder.get_seven_win_draft_ids(set_code))
            if max_drafts is not None:
                random.shuffle(set_draft_ids)
                set_draft_ids = set_draft_ids[:max_drafts]
            draft_ids.extend((set_code, draft_id) for draft_id in set_draft_ids)

        folds = []
        kfold = KFold(n_splits=k, shuffle=True)

        #fold indices is a list of positions from draft_ids
        for _, fold_indices in kfold.split(draft_ids):
            folds.append([draft_ids[i] for i in fold_indices])

        return folds