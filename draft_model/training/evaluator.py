import time
from typing import Dict, List, Tuple

import torch

from ..model.pick_scorer import PickScorer
from .training_data_builder import TrainingDataBuilder, group_fold_by_set

TOP_K_VALUES = (1, 3, 5)


class Evaluator:
    def __init__(
        self,
        training_data_builder: TrainingDataBuilder,
        pick_scorer: PickScorer,
        device: str = None,
    ):
        self.training_data_builder = training_data_builder
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pick_scorer = pick_scorer.to(self.device)

    def evaluate_fold(self, fold: List[Tuple[str, str]], top_k_values=TOP_K_VALUES, log_every: int = 25) -> Dict[str, float]:
        fold_start = time.perf_counter()
        set_codes = sorted({set_code for set_code, _ in fold})
        print(f"[Evaluator] device={self.device}, evaluating on {len(fold)} drafts across {set_codes}")

        prep_start = time.perf_counter()
        name_to_features_by_set = {}
        set_cards_by_set = {}
        for set_code in set_codes:
            name_to_features = self.training_data_builder.get_name_to_features(set_code)
            name_to_features_by_set[set_code] = name_to_features
            set_cards_by_set[set_code] = self.training_data_builder.encode_set_cards(set_code, name_to_features)
        print(f"[Evaluator] set/card features ready in {time.perf_counter() - prep_start:.2f}s, scoring picks")

        draft_ids_by_set = group_fold_by_set(fold)

        self.pick_scorer.eval()
        ranks = []
        build_time = 0.0
        score_time = 0.0
        scoring_start = time.perf_counter()

        with torch.no_grad():
            for set_code, wanted_draft_ids in draft_ids_by_set.items():
                set_cards = set_cards_by_set[set_code]
                name_to_features = name_to_features_by_set[set_code]

                for draft_id, picks in self.training_data_builder.get_draft_pick_sequences(set_code, wanted_draft_ids):
                    for pick in picks:
                        if len(pick['pack_cards']) < 2:
                            continue

                        build_start = time.perf_counter()
                        encoded_examples = self.training_data_builder.encode_pick(pick, set_cards, name_to_features)
                        sequences = torch.stack([seq for seq, _, _, _ in encoded_examples]).to(self.device)
                        masks = torch.stack([mask for _, mask, _, _ in encoded_examples]).to(self.device)
                        labels = [label for _, _, label, _ in encoded_examples]
                        build_time += time.perf_counter() - build_start

                        score_start = time.perf_counter()
                        scores = self.pick_scorer(sequences, masks).tolist()
                        score_time += time.perf_counter() - score_start

                        ranks.append(self.rank_of_picked_card(scores, labels))
                        if len(ranks) % log_every == 0:
                            elapsed = time.perf_counter() - scoring_start
                            print(f"[Evaluator] scored {len(ranks)} picks so far "
                                  f"({elapsed:.1f}s elapsed | build {build_time:.1f}s [{build_time / len(ranks):.3f}s/pick], "
                                  f"score {score_time:.1f}s [{score_time / len(ranks):.3f}s/pick])")

        stats = self.accuracy_from_ranks(ranks, top_k_values)
        print(f"[Evaluator] done: {len(ranks)} picks scored in {time.perf_counter() - scoring_start:.2f}s "
              f"(build {build_time:.2f}s, score {score_time:.2f}s) "
              f"(fold total {time.perf_counter() - fold_start:.2f}s): {stats}")
        return stats

    @staticmethod
    def rank_of_picked_card(scores: List[float], labels: List[int]) -> int:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        picked_positions = [i for i, label in enumerate(labels) if label == 1]
        return min(order.index(i) for i in picked_positions) + 1

    @staticmethod
    def accuracy_from_ranks(ranks: List[int], top_k_values=TOP_K_VALUES) -> Dict[str, float]:
        total = len(ranks)
        result = {"total_picks": total}
        for k in top_k_values:
            hits = sum(1 for rank in ranks if rank <= k)
            result[f"top_{k}_accuracy"] = hits / total if total > 0 else 0.0
        return result
