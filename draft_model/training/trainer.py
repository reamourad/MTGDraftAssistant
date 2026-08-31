import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from ..model.oracle_text_projection import OracleTextProjection
from ..model.pick_scorer import PickScorer
from .training_data_builder import TrainingDataBuilder, group_fold_by_set


class DraftDataset(IterableDataset):
    def __init__(self, training_data_builder, fold, set_cards_by_set, name_to_features_by_set):
        self.training_data_builder = training_data_builder
        self.fold = fold
        self.set_cards_by_set = set_cards_by_set
        self.name_to_features_by_set = name_to_features_by_set

    def __iter__(self):
        draft_ids_by_set = group_fold_by_set(self.fold)

        for set_code, wanted_draft_ids in draft_ids_by_set.items():
            set_cards = self.set_cards_by_set[set_code]
            name_to_features = self.name_to_features_by_set[set_code]

            for draft_id, picks in self.training_data_builder.get_draft_pick_sequences(set_code, wanted_draft_ids):
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
        device: str = None,
    ):
        self.training_data_builder = training_data_builder
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pick_scorer = pick_scorer.to(self.device)
        # stays on CPU: SequenceBuilder runs this per-card during CPU-side data
        # loading (before batching), so its input is always a CPU tensor
        self.projection = projection
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.optimizer = torch.optim.Adam(
            list(self.pick_scorer.parameters()) + list(self.projection.parameters()),
            lr=learning_rate,
        )

    def train_on_fold(
        self,
        fold,
        batch_size: int = 32,
        log_every: int = 5,
        checkpoint_path: str = None,
        checkpoint_every: int = 10000,
    ) -> float:
        fold_start = time.perf_counter()
        set_codes = sorted({set_code for set_code, _ in fold})
        print(f"[Trainer] device={self.device}, training on {len(fold)} drafts across {set_codes}")

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        prep_start = time.perf_counter()
        name_to_features_by_set = {}
        set_cards_by_set = {}
        for set_code in set_codes:
            name_to_features = self.training_data_builder.get_name_to_features(set_code)
            name_to_features_by_set[set_code] = name_to_features
            set_cards_by_set[set_code] = self.training_data_builder.encode_set_cards(set_code, name_to_features)
        print(f"[Trainer] set/card features ready in {time.perf_counter() - prep_start:.2f}s, "
              f"starting batches (batch_size={batch_size})")

        dataset = DraftDataset(self.training_data_builder, fold, set_cards_by_set, name_to_features_by_set)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        self.pick_scorer.train()
        total_loss = 0.0
        num_batches = 0
        fetch_time = 0.0
        step_time = 0.0
        batches_start = time.perf_counter()

        loader_iter = iter(loader)
        while True:
            fetch_start = time.perf_counter()
            try:
                sequences, masks, labels, weights = next(loader_iter)
            except StopIteration:
                break
            fetch_time += time.perf_counter() - fetch_start

            step_start = time.perf_counter()
            total_loss += self.train_step(sequences, masks, labels, weights)
            step_time += time.perf_counter() - step_start
            num_batches += 1

            if checkpoint_path is not None and num_batches % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_path)

            if num_batches % log_every == 0:
                elapsed = time.perf_counter() - batches_start
                print(f"[Trainer] batch {num_batches}: running avg loss = {total_loss / num_batches:.4f} "
                      f"({elapsed:.1f}s elapsed | data-fetch {fetch_time:.1f}s [{fetch_time / num_batches:.3f}s/batch], "
                      f"train_step {step_time:.1f}s [{step_time / num_batches:.3f}s/batch])")

        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[Trainer] done: {num_batches} batches in {time.perf_counter() - batches_start:.2f}s "
              f"(data-fetch {fetch_time:.2f}s, train_step {step_time:.2f}s) "
              f"(fold total {time.perf_counter() - fold_start:.2f}s), final avg loss = {average_loss:.4f}")

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return average_loss

    def save_checkpoint(self, path: str):
        tmp_path = f"{path}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'pick_scorer': self.pick_scorer.state_dict(),
            'projection': self.projection.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.pick_scorer.load_state_dict(checkpoint['pick_scorer'])
            self.projection.load_state_dict(checkpoint['projection'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"[Trainer] resumed (warm-started) from checkpoint {path}")
        except Exception as e:
            print(f"[Trainer] checkpoint {path} is unreadable ({e}), starting fresh")

    def train_step(self, sequences, masks, labels, weights) -> float:
        sequences = sequences.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)
        weights = weights.to(self.device)

        self.optimizer.zero_grad()

        scores = self.pick_scorer(sequences, masks)
        per_example_loss = self.loss_fn(scores, labels)
        loss = (per_example_loss * weights).mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()
