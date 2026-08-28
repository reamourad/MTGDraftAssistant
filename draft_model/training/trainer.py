import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from ..model.oracle_text_projection import OracleTextProjection
from ..model.pick_scorer import PickScorer
from .training_data_builder import TrainingDataBuilder


class DraftDataset(IterableDataset):
    def __init__(self, training_data_builder, fold, set_cards_by_set, name_to_features_by_set):
        self.training_data_builder = training_data_builder
        self.fold = fold
        self.set_cards_by_set = set_cards_by_set
        self.name_to_features_by_set = name_to_features_by_set

    def __iter__(self):
        draft_ids_by_set = {}
        for set_code, draft_id in self.fold:
            draft_ids_by_set.setdefault(set_code, set()).add(draft_id)

        for set_code, wanted_draft_ids in draft_ids_by_set.items():
            set_cards = self.set_cards_by_set[set_code]
            name_to_features = self.name_to_features_by_set[set_code]

            for draft_id, picks in self.training_data_builder.get_draft_pick_sequences(set_code):
                if draft_id not in wanted_draft_ids:
                    continue

                for pick in picks:
                    encoded_examples = self.training_data_builder.encode_pick(
                        pick, set_cards, name_to_features
                    )
                    for sequence, mask, label, weight in encoded_examples:
                        yield (
                            sequence,
                            mask,
                            torch.tensor(label, dtype=torch.float32),
                            torch.tensor(weight, dtype=torch.float32),
                        )


class Trainer:
    def __init__(
        self,
        training_data_builder: TrainingDataBuilder,
        pick_scorer: PickScorer,
        projection: OracleTextProjection,
        learning_rate: float = 1e-4,
    ):
        self.training_data_builder = training_data_builder
        self.pick_scorer = pick_scorer
        self.projection = projection
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adam(
            list(self.pick_scorer.parameters()) + list(self.projection.parameters()),
            lr=learning_rate,
        )

    def train_on_fold(self, fold, batch_size: int = 32) -> float:
        set_codes = sorted({set_code for set_code, _ in fold})

        name_to_features_by_set = {}
        set_cards_by_set = {}
        for set_code in set_codes:
            name_to_features = self.training_data_builder.get_name_to_features(set_code)
            name_to_features_by_set[set_code] = name_to_features
            set_cards_by_set[set_code] = self._encode_set_cards(set_code, name_to_features)

        dataset = DraftDataset(self.training_data_builder, fold, set_cards_by_set, name_to_features_by_set)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        self.pick_scorer.train()
        total_loss = 0.0
        num_batches = 0

        for sequences, masks, labels, weights in loader:
            self.optimizer.zero_grad()

            scores = self.pick_scorer(sequences, masks)
            per_example_loss = self.loss_fn(scores, labels)
            loss = (per_example_loss * weights).mean()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _encode_set_cards(self, set_code, name_to_features):
        card_list = self.training_data_builder.unpack_csv_to_card_list(set_code)
        return [
            self.training_data_builder.card_encoder.encode(name_to_features[name])
            for name in card_list
        ]
