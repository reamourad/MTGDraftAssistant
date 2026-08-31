import time
from collections import Counter

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("draft_model")
)

app = modal.App("mtg-draft-training", image=image)
volume = modal.Volume.from_name("mtg-draft-data", create_if_missing=True)


def fold_summary(fold) -> str:
    counts = Counter(set_code for set_code, _ in fold)
    return f"{len(fold)} drafts {dict(counts)}"


@app.function(gpu="T4", volumes={"/root/data": volume}, timeout=28800)
def train_and_evaluate_fold(fold_index, train_fold, eval_fold, learning_rate: float = 1e-4, batch_size: int = 32):
    import os
    os.chdir("/root")

    from draft_model.model.oracle_text_projection import OracleTextProjection
    from draft_model.model.pick_scorer import PickScorer
    from draft_model.model.sequence_builder import SequenceBuilder
    from draft_model.training.training_data_builder import TrainingDataBuilder
    from draft_model.training.trainer import Trainer
    from draft_model.training.evaluator import Evaluator

    projection = OracleTextProjection()
    sequence_builder = SequenceBuilder(projection)
    training_data_builder = TrainingDataBuilder(sequence_builder)
    pick_scorer = PickScorer()

    # per-fold path on the volume: if this container gets preempted/times out and
    # Modal (or we) retry the same fold, train_on_fold warm-starts from whatever
    # got checkpointed instead of from random-initialized weights
    checkpoint_path = f"/root/data/checkpoints/fold_{fold_index}.pt"

    run_start = time.perf_counter()
    trainer = Trainer(training_data_builder, pick_scorer, projection, learning_rate=learning_rate)
    loss = trainer.train_on_fold(train_fold, batch_size=batch_size, checkpoint_path=checkpoint_path, checkpoint_every=10000)

    evaluator = Evaluator(training_data_builder, pick_scorer, device=trainer.device)
    stats = evaluator.evaluate_fold(eval_fold)
    stats["loss"] = loss

    commit_start = time.perf_counter()
    volume.commit()
    print(f"[train_and_evaluate_fold] volume.commit() took {time.perf_counter() - commit_start:.2f}s, "
          f"whole fold took {time.perf_counter() - run_start:.2f}s")
    return stats


@app.local_entrypoint()
def main(set_codes: str = "MH3", k: int = 5, learning_rate: float = 1e-4, batch_size: int = 32, max_drafts: int = None):
    from draft_model.model.oracle_text_projection import OracleTextProjection
    from draft_model.model.sequence_builder import SequenceBuilder
    from draft_model.training.training_data_builder import TrainingDataBuilder
    from draft_model.training.draft_splitter import DraftSplitter

    projection = OracleTextProjection()
    sequence_builder = SequenceBuilder(projection)
    training_data_builder = TrainingDataBuilder(sequence_builder)
    splitter = DraftSplitter(training_data_builder)

    folds = splitter.get_draft_folds(set_codes.split(","), k, max_drafts=max_drafts)

    total_drafts = sum(len(fold) for fold in folds)
    print(f"Built {k} folds from {total_drafts} total drafts")
    for i, fold in enumerate(folds):
        print(f"  fold {i}: {fold_summary(fold)}")

    fold_args = []
    for i in range(k):
        eval_fold = folds[i]
        train_fold = [pair for j, fold in enumerate(folds) if j != i for pair in fold]
        print(f"  fold {i}: training on {fold_summary(train_fold)}, evaluating on {fold_summary(eval_fold)}")
        fold_args.append((i, train_fold, eval_fold, learning_rate, batch_size))

    print(f"=== launching {k} folds in parallel ===")
    run_start = time.perf_counter()
    all_results = list(train_and_evaluate_fold.starmap(fold_args, return_exceptions=True))
    print(f"all {k} folds done in {time.perf_counter() - run_start:.2f}s")

    all_stats = []
    for i, result in enumerate(all_results):
        if isinstance(result, Exception):
            print(f"fold {i}: FAILED - {result!r}")
        else:
            all_stats.append(result)
            print(f"fold {i}: {result}")

    if not all_stats:
        print("no folds completed successfully, nothing to average")
        return

    print(f"{len(all_stats)}/{k} folds completed successfully")
    for key in ["loss", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy"]:
        average = sum(s[key] for s in all_stats) / len(all_stats)
        print(f"average {key}: {average:.4f}")
