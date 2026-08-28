"""
Runnable smoke test consolidating every check we've verified by hand this session,
against real MH3 data. Not a formal pytest suite yet — that's worth building once
the pipeline (through the training loop) stops actively changing shape. For now,
this is a "did I just break something" re-run: python -m draft_model.tests.smoke_test
"""
import math
import time

import torch

from ..card_encoder.card_encoder import CardEncoder
from ..external_api.mtgjson_data import MTGJson
from ..model.oracle_text_projection import OracleTextProjection
from ..model.pick_scorer import PickScorer
from ..model import sequence_builder as sequence_builder_module
from ..model.sequence_builder import SequenceBuilder
from ..training_data.training_data_builder import TrainingDataBuilder

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
    check(0.0 <= score.item() <= 1.0, "score is a valid sigmoid output in [0, 1]")

    with torch.no_grad():
        model(seq, mask)  # warmup
        start = time.perf_counter()
        for _ in range(10):
            model(seq, mask)
        elapsed = (time.perf_counter() - start) / 10
    check(elapsed < 0.5, f"forward pass under 0.5s target (was {elapsed*1000:.1f}ms)")


def test_training_data_builder():
    print("TrainingDataBuilder")
    tdb = TrainingDataBuilder()

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


if __name__ == "__main__":
    test_card_encoder()
    test_sequence_builder()
    test_pick_scorer()
    test_training_data_builder()
    print("\nAll checks passed.")
