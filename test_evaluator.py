"""Test ModelEvaluator functionality."""

import torch
from app.training.evaluator import ModelEvaluator
from app.training.dataset import DraftDataset
from app.training import DraftSequence
from app.ml.experimental.two_tower_model import TwoTowerModel
from app.ml.experimental.card_encoder import CardEncoder
from torch.utils.data import DataLoader


def test_evaluator_initialization():
    """Test ModelEvaluator initialization."""
    print("Testing ModelEvaluator initialization...")
    
    # Create mock card data
    mock_cards = [
        {
            "name": "Card A",
            "rarity": "common",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Creature"],
            "subtypes": [],
            "power": 2.0,
            "toughness": 2.0,
            "can_attack": True,
            "keywords": [],
            "oracle_text": "Test"
        },
        {
            "name": "Card B",
            "rarity": "common",
            "colors": ["U"],
            "mana_cost": "{2}{U}",
            "converted_mana_cost": 3.0,
            "types": ["Instant"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test"
        }
    ]
    
    card_encoder = CardEncoder(card_list=mock_cards)
    
    sequences = [
        DraftSequence(
            draft_id="test_1",
            pick_number=1,
            pool=[],
            pack=["Card A", "Card B"],
            picked_card="Card A"
        )
    ]
    
    dataset = DraftDataset(sequences, card_encoder)
    val_loader = DataLoader(dataset, batch_size=1)
    
    model = TwoTowerModel()
    evaluator = ModelEvaluator(model, val_loader)
    
    assert evaluator.model is model
    assert evaluator.val_loader is val_loader
    assert evaluator.device is not None
    print(f"  Device: {evaluator.device}")
    
    print("✓ ModelEvaluator initialization passed\n")


def test_evaluate_metrics():
    """Test evaluate() method returns correct metrics."""
    print("Testing evaluate() metrics...")
    
    # Create mock card data
    mock_cards = [
        {
            "name": "Card A",
            "rarity": "common",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Creature"],
            "subtypes": [],
            "power": 2.0,
            "toughness": 2.0,
            "can_attack": True,
            "keywords": [],
            "oracle_text": "Test"
        },
        {
            "name": "Card B",
            "rarity": "common",
            "colors": ["U"],
            "mana_cost": "{2}{U}",
            "converted_mana_cost": 3.0,
            "types": ["Instant"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test"
        },
        {
            "name": "Card C",
            "rarity": "rare",
            "colors": ["R"],
            "mana_cost": "{3}{R}",
            "converted_mana_cost": 4.0,
            "types": ["Sorcery"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test"
        }
    ]
    
    card_encoder = CardEncoder(card_list=mock_cards)
    
    sequences = [
        DraftSequence(
            draft_id="test_1",
            pick_number=1,
            pool=[],
            pack=["Card A", "Card B", "Card C"],
            picked_card="Card A"
        ),
        DraftSequence(
            draft_id="test_2",
            pick_number=2,
            pool=["Card A"],
            pack=["Card B", "Card C"],
            picked_card="Card B"
        )
    ]
    
    dataset = DraftDataset(sequences, card_encoder)
    val_loader = DataLoader(dataset, batch_size=2, num_workers=0)
    
    model = TwoTowerModel()
    evaluator = ModelEvaluator(model, val_loader, device=torch.device('cpu'))
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'top1_acc' in metrics
    assert 'top3_acc' in metrics
    assert 'top5_acc' in metrics
    
    # Check metrics are valid
    assert isinstance(metrics['loss'], float)
    assert metrics['loss'] > 0
    assert 0.0 <= metrics['top1_acc'] <= 1.0
    assert 0.0 <= metrics['top3_acc'] <= 1.0
    assert 0.0 <= metrics['top5_acc'] <= 1.0
    
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Top-1 accuracy: {metrics['top1_acc']:.4f}")
    print(f"  Top-3 accuracy: {metrics['top3_acc']:.4f}")
    print(f"  Top-5 accuracy: {metrics['top5_acc']:.4f}")
    
    print("✓ evaluate() metrics test passed\n")


def test_top_k_accuracy():
    """Test compute_top_k_accuracy() method."""
    print("Testing compute_top_k_accuracy()...")
    
    # Create mock card data
    mock_cards = [
        {
            "name": "Card A",
            "rarity": "common",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Creature"],
            "subtypes": [],
            "power": 2.0,
            "toughness": 2.0,
            "can_attack": True,
            "keywords": [],
            "oracle_text": "Test"
        }
    ]
    
    card_encoder = CardEncoder(card_list=mock_cards)
    sequences = [
        DraftSequence(
            draft_id="test_1",
            pick_number=1,
            pool=[],
            pack=["Card A"],
            picked_card="Card A"
        )
    ]
    
    dataset = DraftDataset(sequences, card_encoder)
    val_loader = DataLoader(dataset, batch_size=1)
    
    model = TwoTowerModel()
    evaluator = ModelEvaluator(model, val_loader)
    
    # Test with mock scores
    # Batch of 2, 5 candidates each
    scores = torch.tensor([
        [0.9, 0.7, 0.5, 0.3, 0.1],  # Target is index 0 (highest score)
        [0.9, 0.7, 0.5, 0.3, 0.1]   # Target is index 2 (third highest score)
    ])
    target = torch.tensor([0, 2])
    
    top1, top3, top5 = evaluator.compute_top_k_accuracy(scores, target)
    
    # First sample: target at index 0 (highest score) → correct for all k
    # Second sample: target at index 2 (third highest score) → correct for k>=3
    # Expected: top1=0.5 (1/2), top3=1.0 (2/2), top5=1.0 (2/2)
    assert top1 == 0.5, f"Expected top1=0.5, got {top1}"
    assert top3 == 1.0, f"Expected top3=1.0, got {top3}"
    assert top5 == 1.0, f"Expected top5=1.0, got {top5}"
    
    print(f"  Top-1: {top1:.2f} (expected 0.50)")
    print(f"  Top-3: {top3:.2f} (expected 1.00)")
    print(f"  Top-5: {top5:.2f} (expected 1.00)")
    
    # Test edge case: pack size < k
    scores_small = torch.tensor([[0.9, 0.1]])  # Only 2 cards
    target_small = torch.tensor([0])
    
    top1, top3, top5 = evaluator.compute_top_k_accuracy(scores_small, target_small)
    
    # With only 2 cards, top-3 and top-5 should be same as top-2
    assert top1 == 1.0
    assert top3 == 1.0  # Target is in top-2, so also in "top-3"
    assert top5 == 1.0
    
    print(f"  Edge case (2 cards): Top-1={top1:.2f}, Top-3={top3:.2f}, Top-5={top5:.2f}")
    
    print("✓ compute_top_k_accuracy() test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ModelEvaluator")
    print("=" * 60 + "\n")
    
    test_evaluator_initialization()
    test_evaluate_metrics()
    test_top_k_accuracy()
    
    print("=" * 60)
    print("All evaluator tests passed!")
    print("=" * 60)
