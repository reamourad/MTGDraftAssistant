"""
Runnable smoke test consolidating every check we've verified by hand this session,
against real MH3 data. Not a formal pytest suite yet — that's worth building once
the pipeline (through the training loop) stops actively changing shape. For now,
this is a "did I just break something" re-run: python -m draft_model.tests.smoke_test
"""
import math
import os
import tempfile
import time

import numpy as np
import torch

from ..card_encoder.card_encoder import CardEncoder
from ..external_api.mtgjson_data import MTGJson
from ..model.oracle_text_projection import OracleTextProjection
from ..model.pick_scorer import PickScorer
from ..model import sequence_builder as sequence_builder_module
from ..model.sequence_builder import SequenceBuilder
from ..training.training_data_builder import TrainingDataBuilder, _placeholder_card_features
from ..training.draft_splitter import DraftSplitter
from ..training.trainer import Trainer
from ..training.evaluator import Evaluator
from ..model.sequence_builder import TOKEN_DIM

SET_CODE = "MH3"


def check(condition: bool, description: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    assert condition, f"FAILED: {description}"


def test_card_encoder():
    print("CardEncoder")
    mtgjson = MTGJson()
    features = mtgjson.get_uuid_to_card_features(mtgjson.fetch_set_data(SET_CODE))
    card = next(iter(features.values()))

    encoder = CardEncoder(use_gpu=False)
    vector = encoder.encode(card)

    check(vector.shape == (encoder.TOTAL_DIM,), f"encoded vector shape is ({encoder.TOTAL_DIM},)")
    check(vector.dtype.name == "float32", "encoded vector is float32")

    all_cards = list(features.values())
    with_text = next(c for c in all_cards if c.get('oracle_text'))
    without_text = next((c for c in all_cards if not c.get('oracle_text')), None)
    batch_cards = [with_text] + ([without_text] if without_text is not None else [all_cards[1]])
    batch_cards = list({c['name']: c for c in batch_cards}.values())

    fresh_single_encoder = CardEncoder(use_gpu=False)
    expected = np.stack([fresh_single_encoder.encode(c) for c in batch_cards])

    batch_encoder = CardEncoder(use_gpu=False)
    batch_result = batch_encoder.encode_batch(batch_cards)

    check(batch_result.shape == (len(batch_cards), encoder.TOTAL_DIM), "encode_batch returns one row per card")
    check(np.allclose(batch_result, expected, atol=1e-5),
          "encode_batch matches independently per-card encoded results, in the same order")
    check(all(c['name'] in batch_encoder._cache for c in batch_cards),
          "encode_batch populates the cache for every card")

    mixed_encoder = CardEncoder(use_gpu=False)
    mixed_encoder.encode(batch_cards[0])
    cached_vector = mixed_encoder._cache[batch_cards[0]['name']].copy()
    fresh_card = next(c for c in all_cards if c['name'] not in {bc['name'] for bc in batch_cards})
    mixed_result = mixed_encoder.encode_batch([batch_cards[0], fresh_card])

    check(np.array_equal(mixed_result[0], cached_vector), "encode_batch reuses an existing cache hit unchanged")
    check(fresh_card['name'] in mixed_encoder._cache,
          "encode_batch populates the cache for a newly-seen card in a mixed batch")

    check(batch_encoder.encode_batch([]).shape == (0, encoder.TOTAL_DIM),
          "encode_batch on an empty list returns an empty array")


def test_sequence_builder():
    print("SequenceBuilder")
    mtgjson = MTGJson()
    features = mtgjson.get_uuid_to_card_features(mtgjson.fetch_set_data(SET_CODE))
    cards = list(features.values())[:5]

    encoder = CardEncoder(use_gpu=False)
    projection = OracleTextProjection()
    builder = SequenceBuilder(projection)
    vectors = encoder.encode_batch(cards)

    candidate = vectors[0]
    other_pack = list(vectors[1:3])
    pool = list(vectors[3:4])
    set_ctx = list(vectors[3:5])

    seq, mask = builder.build_full_sequence(set_ctx, pool, candidate, other_pack)
    m = sequence_builder_module
    expected_len = m.MAX_SET_SIZE + m.MAX_POOL_SIZE + m.MAX_OTHER_PACK_SIZE + 1

    check(seq.shape == (expected_len, 91), f"full sequence shape is ({expected_len}, 91)")
    check(mask.sum().item() == 6, "6 real tokens counted (2 set + 1 pool + 1 candidate + 2 other-pack)")
    check(mask[-1].item() is True, "candidate slot (last position) is always real")

    empty_pool_seq, empty_pool_mask = builder.build_full_sequence(set_ctx, [], candidate, other_pack)
    check(not torch.isnan(empty_pool_seq).any().item(), "empty pool produces no NaN")
    pool_region = empty_pool_mask[m.MAX_SET_SIZE:m.MAX_SET_SIZE + m.MAX_POOL_SIZE]
    check(not pool_region.any().item(), "empty pool's mask region is all False")

    empty_result, empty_mask_only = builder.build_padded_sequence([], m.ROLE_POOL, m.MAX_POOL_SIZE)
    check(empty_result.shape == (m.MAX_POOL_SIZE, 91) and not empty_mask_only.any().item(),
          "build_padded_sequence on an empty card list returns all-zero, all-False padding")

    other_pack_seq, other_pack_mask = builder.build_padded_sequence(other_pack, m.ROLE_OTHER_PACK, m.MAX_OTHER_PACK_SIZE)
    for i, card_vector in enumerate(other_pack):
        expected_structured = torch.from_numpy(card_vector[:CardEncoder.STRUCTURED_DIM]).float()
        expected_oracle = torch.from_numpy(card_vector[CardEncoder.STRUCTURED_DIM:]).float()
        with torch.no_grad():
            expected_compressed = projection(expected_oracle.unsqueeze(0)).squeeze(0)
        expected_token = torch.cat([expected_structured, expected_compressed, m.ROLE_OTHER_PACK])
        check(torch.allclose(other_pack_seq[i], expected_token, atol=1e-5),
              f"build_padded_sequence's vectorized row {i} matches an independently-computed reference token")


def test_pick_scorer():
    print("PickScorer")
    mtgjson = MTGJson()
    features = mtgjson.get_uuid_to_card_features(mtgjson.fetch_set_data(SET_CODE))
    cards = list(features.values())[:5]

    encoder = CardEncoder(use_gpu=False)
    projection = OracleTextProjection()
    builder = SequenceBuilder(projection)
    vectors = encoder.encode_batch(cards)

    candidate = vectors[0]
    other_pack = list(vectors[1:3])
    pool = list(vectors[3:4])
    set_ctx = list(vectors[3:5])

    seq, mask = builder.build_full_sequence(set_ctx, pool, candidate, other_pack)
    seq, mask = seq.unsqueeze(0), mask.unsqueeze(0)

    model = PickScorer()
    model.eval()
    torch.set_num_threads(1)

    with torch.no_grad():
        score = model(seq, mask)
    check(score.shape == (1,), "score has shape (1,)")
    # raw logit now, not a probability — sigmoid happens in the loss (or manually, for display)
    check(not torch.isnan(score).any().item(), "score is a real number, not NaN")
    check(0.0 <= torch.sigmoid(score).item() <= 1.0, "sigmoid(score) is a valid probability in [0, 1]")

    with torch.no_grad():
        model(seq, mask)  # warmup
        start = time.perf_counter()
        for _ in range(10):
            model(seq, mask)
        elapsed = (time.perf_counter() - start) / 10
    check(elapsed < 0.5, f"forward pass under 0.5s target (was {elapsed*1000:.1f}ms)")


def test_training_data_builder():
    print("TrainingDataBuilder")
    projection = OracleTextProjection()
    sequence_builder = SequenceBuilder(projection)
    tdb = TrainingDataBuilder(sequence_builder)

    card_list = tdb.unpack_csv_to_card_list(SET_CODE)
    check(card_list is not None and len(card_list) > 0, "unpack_csv_to_card_list returns a non-empty set")

    draft_id, picks = next(tdb.get_draft_pick_sequences(SET_CODE))

    pack_0_sizes = [len(p["pack_cards"]) for p in picks if p["pack_number"] == 0]
    check(pack_0_sizes == list(range(14, 0, -1)), "pack 0 shrinks 14 -> 1 across its picks")
    check(all(p["picked_card"] in p["pack_cards"] for p in picks), "picked card is always in its own pack")

    check(picks[0]["pool"] == [], "pool starts empty at pick 0")
    check(len(picks[5]["pool"]) == 5, "pool has grown to 5 cards by pick index 5")
    check(picks[0]["pool"] == [], "pick 0's pool snapshot is untouched by later picks (no aliasing)")

    examples = tdb.build_training_examples(picks[0])
    positives = [e for e in examples if e["label"] == 1]
    negatives = [e for e in examples if e["label"] == 0]
    check(len(positives) == 1 and len(negatives) == 13, "one positive, thirteen negatives for a 14-card pack")
    check(math.isclose(sum(e["weight"] for e in positives), 1.0), "total positive weight is 1.0")
    check(math.isclose(sum(e["weight"] for e in negatives), 1.0, rel_tol=1e-6), "total negative weight is 1.0")

    a_negative = negatives[0]
    check(picks[0]["picked_card"] in a_negative["other_pack_cards"],
          "a negative example's other_pack_cards includes the card that actually won")
    check(a_negative["candidate"] not in a_negative["other_pack_cards"],
          "a negative example's other_pack_cards excludes itself")

    last_pick_examples = tdb.build_training_examples(picks[-1])
    check(len(last_pick_examples) == 1 and last_pick_examples[0]["weight"] == 1.0,
          "single-card pack (forced pick) produces one example, no divide-by-zero crash")

    # regression check for the duplicate-copy bug: pack 1, pick 9 of this exact
    # draft genuinely contains 2 copies of "Eviscerator's Insight" (found via
    # pack_card_* value of 2, not 0/1 — the smoke test itself caught this the
    # first time it ran).
    dup_pick = next(p for p in picks if p["pack_number"] == 1 and p["pick_number"] == 9)
    check(dup_pick["pack_cards"].count("Eviscerator's Insight") == 2,
          "duplicate card (2 copies) is preserved in pack_cards, not silently dropped")
    dup_examples = tdb.build_training_examples(dup_pick)
    dup_positives = [e for e in dup_examples if e["label"] == 1]
    check(len(dup_positives) == 2, "both copies of the picked duplicate are labeled positive")
    check(math.isclose(sum(e["weight"] for e in dup_positives), 1.0),
          "total positive weight still sums to 1.0 when split across duplicate copies")
    check(dup_positives[0]["other_pack_cards"].count("Eviscerator's Insight") == 1,
          "excluding one occurrence as candidate leaves the OTHER copy visible in other_pack_cards")

    seven_win_ids = tdb.get_seven_win_draft_ids(SET_CODE)
    wanted = set(list(seven_win_ids)[:2])
    filtered_draft_ids = {d for d, _ in tdb.get_draft_pick_sequences(SET_CODE, wanted)}
    check(filtered_draft_ids == wanted,
          "get_draft_pick_sequences with a draft_ids filter yields exactly (and only) those drafts")

    name_to_features = tdb.get_name_to_features(SET_CODE)
    check(len(name_to_features) > 0, "get_name_to_features returns a non-empty lookup")
    check(all(name in name_to_features for name in card_list),
          "every real card name in card_list resolves to features (no silent gaps)")
    a_name = next(iter(card_list))
    check(name_to_features[a_name]["name"] == a_name, "a resolved entry's own name matches its lookup key")

    # full pipeline, end to end: raw CSV -> encode_pick -> straight into PickScorer
    some_names = list(name_to_features.keys())[:5]
    set_cards = [tdb.card_encoder.encode(name_to_features[n]) for n in some_names]
    encoded = tdb.encode_pick(picks[0], set_cards, name_to_features)
    check(len(encoded) == 14, "encode_pick produces one encoded example per card in the pack")

    seq, mask, label, weight = encoded[0]
    check(seq.shape == (660, 91) and mask.shape == (660,), "encoded example has the expected sequence/mask shape")

    model = PickScorer()
    model.eval()
    with torch.no_grad():
        score = model(seq.unsqueeze(0), mask.unsqueeze(0))
    check(0.0 <= torch.sigmoid(score).item() <= 1.0, "a real encoded example scores end to end (sigmoid applied manually)")


def test_trainer():
    print("Trainer")
    projection = OracleTextProjection()
    pick_scorer = PickScorer()
    trainer = Trainer(training_data_builder=None, pick_scorer=pick_scorer, projection=projection, device="cpu")

    check(trainer.device == "cpu", "trainer respects an explicit device override")
    check(next(trainer.pick_scorer.parameters()).device.type == "cpu", "pick_scorer parameters live on trainer.device")
    check(next(trainer.projection.parameters()).device.type == "cpu",
          "projection stays on CPU regardless of trainer.device (SequenceBuilder runs it during CPU-side data loading)")

    batch_size, seq_len = 4, 5
    sequences = torch.randn(batch_size, seq_len, TOKEN_DIM)
    masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
    weights = torch.ones(batch_size, dtype=torch.float32)

    trainer.pick_scorer.train()
    param_before = next(trainer.pick_scorer.parameters()).clone()
    loss = trainer.train_step(sequences, masks, labels, weights)
    param_after = next(trainer.pick_scorer.parameters())

    check(not math.isnan(loss), "train_step produces a real loss, not NaN")
    check(not torch.equal(param_before, param_after),
          "pick_scorer parameters actually update after train_step (gradients flow through optimizer.step)")

    checkpoint_path = os.path.join(tempfile.mkdtemp(), "fold_0.pt")
    trained_pick_scorer_param = next(trainer.pick_scorer.parameters()).clone()
    trained_projection_param = next(trainer.projection.parameters()).clone()
    trainer.save_checkpoint(checkpoint_path)
    check(os.path.exists(checkpoint_path), "save_checkpoint writes a file")

    fresh_trainer = Trainer(training_data_builder=None, pick_scorer=PickScorer(),
                             projection=OracleTextProjection(), device="cpu")
    check(not torch.equal(next(fresh_trainer.pick_scorer.parameters()), trained_pick_scorer_param),
          "a freshly-initialized trainer starts with different weights (sanity check before loading)")
    fresh_trainer.load_checkpoint(checkpoint_path)
    check(torch.equal(next(fresh_trainer.pick_scorer.parameters()), trained_pick_scorer_param),
          "load_checkpoint restores the exact pick_scorer weights that were saved")
    check(torch.equal(next(fresh_trainer.projection.parameters()), trained_projection_param),
          "load_checkpoint restores the exact projection weights that were saved")

    another_fresh_trainer = Trainer(training_data_builder=None, pick_scorer=PickScorer(),
                                     projection=OracleTextProjection(), device="cpu")
    another_fresh_trainer.load_checkpoint(os.path.join(tempfile.mkdtemp(), "does_not_exist.pt"))
    check(True, "load_checkpoint on a missing path is a no-op, not a crash")


def test_evaluator():
    print("Evaluator")

    check(Evaluator.rank_of_picked_card([0.1, 0.9, 0.5], [0, 1, 0]) == 1,
          "the highest-scored card, when it's the picked one, ranks 1")
    check(Evaluator.rank_of_picked_card([0.9, 0.1, 0.5], [0, 1, 0]) == 3,
          "the picked card ranks by its position in descending score order")
    check(Evaluator.rank_of_picked_card([0.9, 0.5, 0.1], [1, 1, 0]) == 1,
          "duplicate positives use the best (lowest) rank among them")

    accuracy = Evaluator.accuracy_from_ranks([1, 1, 2, 4, 5], top_k_values=(1, 3, 5))
    check(accuracy["total_picks"] == 5, "accuracy_from_ranks counts every rank")
    check(math.isclose(accuracy["top_1_accuracy"], 2 / 5), "top_1_accuracy only counts rank == 1")
    check(math.isclose(accuracy["top_3_accuracy"], 3 / 5), "top_3_accuracy counts ranks <= 3")
    check(math.isclose(accuracy["top_5_accuracy"], 1.0), "top_5_accuracy counts every rank here")

    empty_accuracy = Evaluator.accuracy_from_ranks([], top_k_values=(1,))
    check(empty_accuracy["top_1_accuracy"] == 0.0, "accuracy_from_ranks handles zero picks without dividing by zero")

    projection = OracleTextProjection()
    sequence_builder = SequenceBuilder(projection)
    tdb = TrainingDataBuilder(sequence_builder)
    pick_scorer = PickScorer()
    evaluator = Evaluator(tdb, pick_scorer, device="cpu")

    name_to_features = tdb.get_name_to_features(SET_CODE)
    some_names = list(name_to_features.keys())[:5]
    set_cards = [tdb.card_encoder.encode(name_to_features[n]) for n in some_names]

    _, picks = next(tdb.get_draft_pick_sequences(SET_CODE))
    encoded_examples = tdb.encode_pick(picks[0], set_cards, name_to_features)
    sequences = torch.stack([seq for seq, _, _, _ in encoded_examples])
    masks = torch.stack([mask for _, mask, _, _ in encoded_examples])
    labels = [label for _, _, label, _ in encoded_examples]

    evaluator.pick_scorer.eval()
    with torch.no_grad():
        scores = evaluator.pick_scorer(sequences, masks).tolist()
    rank = evaluator.rank_of_picked_card(scores, labels)
    check(1 <= rank <= len(labels), "a real pick's rank falls within the pack size")


def test_draft_splitter():
    print("DraftSplitter")
    projection = OracleTextProjection()
    sequence_builder = SequenceBuilder(projection)
    tdb = TrainingDataBuilder(sequence_builder)
    splitter = DraftSplitter(tdb)

    all_sets = ["MH3", "TDM", "FIN"]
    held_out, remaining = splitter.choose_held_out_set(all_sets)
    check(held_out in all_sets and held_out not in remaining, "held-out set is excluded from the remaining list")
    check(set(remaining) == set(all_sets) - {held_out}, "remaining sets are exactly all_sets minus the held-out one")

    folds = splitter.get_draft_folds(remaining, k=5)
    check(len(folds) == 5, "get_draft_folds returns the requested number of folds")
    all_drafts = [pair for fold in folds for pair in fold]
    check(len(all_drafts) == len(set(all_drafts)), "no (set_code, draft_id) pair appears in more than one fold")

    capped_folds = splitter.get_draft_folds(["MH3", "TDM"], k=2, max_drafts=3)
    capped_drafts = [pair for fold in capped_folds for pair in fold]
    capped_sets = {set_code for set_code, _ in capped_drafts}
    check(capped_sets == {"MH3", "TDM"}, "max_drafts caps per-set, not the combined list — both sets are represented")
    check(sum(1 for s, _ in capped_drafts if s == "MH3") <= 3 and sum(1 for s, _ in capped_drafts if s == "TDM") <= 3,
          "each set contributes at most max_drafts drafts")


def test_mtgjson():
    print("MTGJson")
    mtgjson = MTGJson()

    play_arena_booster = mtgjson.get_arena_booster({"booster": {"play-arena": {"sourceSetCodes": ["X"]}}}, "X")
    check(play_arena_booster == {"sourceSetCodes": ["X"]}, "get_arena_booster prefers 'play-arena' when present")

    older_set_booster = mtgjson.get_arena_booster({"booster": {"arena": {"sourceSetCodes": ["Y"]}}}, "Y")
    check(older_set_booster == {"sourceSetCodes": ["Y"]},
          "get_arena_booster falls back to 'arena' for older sets that predate the 'play-arena' key")

    try:
        mtgjson.get_arena_booster({"booster": {"collector": {}}}, "Z")
        check(False, "get_arena_booster raises when neither 'play-arena' nor 'arena' exists")
    except KeyError:
        check(True, "get_arena_booster raises when neither 'play-arena' nor 'arena' exists")

    atomic_names = mtgjson.get_name_to_features_from_atomic(["Ancestral Recall", "Archon of Cruelty", "Not A Real Card"])
    check(set(atomic_names.keys()) == {"Ancestral Recall", "Archon of Cruelty"},
          "get_name_to_features_from_atomic resolves real names and skips unresolvable ones")
    check(atomic_names["Ancestral Recall"]["name"] == "Ancestral Recall",
          "an atomic-resolved entry's own name matches its lookup key")
    check(atomic_names["Ancestral Recall"]["rarity"] == "common",
          "atomic data has no rarity field, so it defaults to 'common' like the normal set path already does")
    check(atomic_names["Archon of Cruelty"]["power"] == 6.0 and atomic_names["Archon of Cruelty"]["toughness"] == 6.0,
          "atomic-resolved power/toughness are correct")
    check("Flying" in atomic_names["Archon of Cruelty"]["oracle_text"],
          "atomic-resolved oracle text is correct")

    dfc_names = mtgjson.get_name_to_features_from_atomic(["Bonecrusher Giant", "Stomp"])
    check(set(dfc_names.keys()) == {"Bonecrusher Giant", "Stomp"},
          "modal double-faced cards resolve by their individual faceName")
    check(dfc_names["Bonecrusher Giant"]["power"] == 4.0, "the front face's own stats resolve, not the back face's")

    split_names = mtgjson.get_name_to_features_from_atomic(["Life // Death"])
    check("Life // Death" in split_names,
          "old-style split cards resolve by their combined name (the opposite convention from DFCs)")

    placeholder = _placeholder_card_features("Some Brand New Card")
    check(placeholder["name"] == "Some Brand New Card" and placeholder["rarity"] == "common"
          and placeholder["types"] == [] and placeholder["oracle_text"] == "",
          "placeholder features are neutral/uninformative, not silently wrong")

    projection = OracleTextProjection()
    sequence_builder = SequenceBuilder(projection)
    cube_tdb = TrainingDataBuilder(sequence_builder)
    cube_card_list = cube_tdb.unpack_csv_to_card_list("Powered_Cube")
    cube_name_to_features = cube_tdb.get_name_to_features("Powered_Cube")
    check(all(name in cube_name_to_features for name in cube_card_list),
          "Powered_Cube (no real MTGJSON set page) fully resolves via the atomic fallback + placeholders")
    cube_set_cards = cube_tdb.encode_set_cards("Powered_Cube", cube_name_to_features)
    check(len(cube_set_cards) == len(cube_card_list) and cube_set_cards[0].shape == (CardEncoder.TOTAL_DIM,),
          "Powered_Cube's full card list encodes end to end")

    omenpath_names = ["Ademi of the Silkchutes", "Goben, Gene-Splice Savant", "Luis, Pompous Pillager",
                       "Makdee and Itla, Skysnarers", "Nia, Skysail Storyteller", "Yera and Oski, Weaver and Guide"]
    check(all(cube_name_to_features[name]["oracle_text"] != "" for name in omenpath_names),
          "the 6 Through the Omenpaths cards use real manually-curated text, not placeholders")
    check(cube_name_to_features["Ademi of the Silkchutes"]["power"] == 3
          and cube_name_to_features["Ademi of the Silkchutes"]["toughness"] == 2,
          "manually-curated stats match the real Gatherer card (Ademi of the Silkchutes is 3/2)")


if __name__ == "__main__":
    test_mtgjson()
    test_card_encoder()
    test_sequence_builder()
    test_pick_scorer()
    test_training_data_builder()
    test_trainer()
    test_evaluator()
    test_draft_splitter()
    print("\nAll checks passed.")
