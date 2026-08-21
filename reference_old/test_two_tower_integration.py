"""
Test script to verify TwoTowerModel integration.

This script tests:
1. Model instantiation
2. Forward pass with dummy data
3. Output shapes
4. Gradient flow
5. Checkpoint saving and loading
"""

import torch
import os
import tempfile
from app.ml.experimental.two_tower_model import TwoTowerModel


def test_model_instantiation():
    """Test that the model can be instantiated."""
    print("Test 1: Model Instantiation")
    model = TwoTowerModel(card_dim=407, hidden_dim=256, embedding_dim=128)
    print(f"✓ Model created successfully")
    print(f"  - CandidateTower: {model.candidate_tower}")
    print(f"  - ContextTower: {model.context_tower}")
    print(f"  - ScoringHead: {model.scoring_head}")
    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\nTest 2: Forward Pass")
    
    batch_size = 4
    num_candidates = 10
    num_picked = 5
    num_available = 10
    card_dim = 407
    
    # Create dummy data
    candidate_cards = torch.randn(batch_size, num_candidates, card_dim)
    pool_cards = torch.randn(batch_size, num_picked, card_dim)
    pack_cards = torch.randn(batch_size, num_available, card_dim)
    pick_number = torch.randint(1, 46, (batch_size, 1)).float()
    
    print(f"  Input shapes:")
    print(f"    - candidate_cards: {candidate_cards.shape}")
    print(f"    - pool_cards: {pool_cards.shape}")
    print(f"    - pack_cards: {pack_cards.shape}")
    print(f"    - pick_number: {pick_number.shape}")
    
    # Forward pass
    scores = model(candidate_cards, pool_cards, pack_cards, pick_number)
    
    print(f"  Output shape: {scores.shape}")
    expected_shape = (batch_size, num_candidates, 1)
    assert scores.shape == expected_shape, f"Expected {expected_shape}, got {scores.shape}"
    print(f"✓ Forward pass successful with correct output shape")
    
    return scores


def test_output_shapes():
    """Test output shapes for various input configurations."""
    print("\nTest 3: Output Shapes for Various Inputs")
    
    model = TwoTowerModel()
    
    test_cases = [
        {"batch": 1, "candidates": 5, "picked": 0, "available": 5, "desc": "First pick"},
        {"batch": 2, "candidates": 15, "picked": 10, "available": 15, "desc": "Mid-draft"},
        {"batch": 8, "candidates": 3, "picked": 40, "available": 3, "desc": "Late pick"},
    ]
    
    for case in test_cases:
        candidate_cards = torch.randn(case["batch"], case["candidates"], 407)
        pool_cards = torch.randn(case["batch"], case["picked"], 407)
        pack_cards = torch.randn(case["batch"], case["available"], 407)
        pick_number = torch.randint(1, 46, (case["batch"], 1)).float()
        
        scores = model(candidate_cards, pool_cards, pack_cards, pick_number)
        expected = (case["batch"], case["candidates"], 1)
        
        assert scores.shape == expected, f"Failed for {case['desc']}"
        print(f"  ✓ {case['desc']}: {scores.shape}")


def test_gradient_flow(model):
    """Test that gradients flow through all components."""
    print("\nTest 4: Gradient Flow")
    
    # Create dummy data
    candidate_cards = torch.randn(2, 5, 407, requires_grad=True)
    pool_cards = torch.randn(2, 3, 407)
    pack_cards = torch.randn(2, 5, 407)
    pick_number = torch.tensor([[10.0], [20.0]])
    
    # Forward pass
    scores = model(candidate_cards, pool_cards, pack_cards, pick_number)
    
    # Compute loss (simple sum for testing)
    loss = scores.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grad = {
        'candidate_tower': any(p.grad is not None for p in model.candidate_tower.parameters()),
        'context_tower': any(p.grad is not None for p in model.context_tower.parameters()),
        'scoring_head': any(p.grad is not None for p in model.scoring_head.parameters()),
    }
    
    print(f"  Gradients present:")
    for component, has_g in has_grad.items():
        status = "✓" if has_g else "✗"
        print(f"    {status} {component}")
    
    assert all(has_grad.values()), "Not all components have gradients"
    print(f"✓ Gradients flow through all components")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\nTest 5: Checkpoint Save/Load")
    
    # Create model on CPU for consistent testing
    device = torch.device('cpu')
    model1 = TwoTowerModel(card_dim=407, hidden_dim=256, embedding_dim=128)
    model1.to(device)
    
    # Create temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        
        # Save checkpoint with metadata
        metadata = {
            'epoch': 10,
            'loss': 0.5,
            'set_code': 'MH3'
        }
        model1.save_checkpoint(checkpoint_path, metadata)
        print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint on CPU
        model2 = TwoTowerModel.load_checkpoint(checkpoint_path, device=device)
        print(f"  ✓ Checkpoint loaded successfully")
        
        # Verify metadata
        loaded_metadata = model2.get_metadata(checkpoint_path)
        assert loaded_metadata == metadata, "Metadata mismatch"
        print(f"  ✓ Metadata preserved: {loaded_metadata}")
        
        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.allclose(param1, param2), f"Parameter values differ for {name1}"
        
        print(f"  ✓ All weights match between saved and loaded models")


def test_predict_pick():
    """Test the predict_pick convenience method."""
    print("\nTest 6: Predict Pick Method")
    
    model = TwoTowerModel()
    
    # Single draft state (no batch dimension)
    num_candidates = 8
    num_picked = 5
    num_available = 8
    
    candidate_cards = torch.randn(num_candidates, 407)
    pool_cards = torch.randn(num_picked, 407)
    pack_cards = torch.randn(num_available, 407)
    pick_number = 6
    
    scores = model.predict_pick(candidate_cards, pool_cards, pack_cards, pick_number)
    
    print(f"  Input: {num_candidates} candidates, pick #{pick_number}")
    print(f"  Output shape: {scores.shape}")
    assert scores.shape == (num_candidates,), f"Expected ({num_candidates},), got {scores.shape}"
    print(f"  ✓ Predict pick works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TwoTowerModel Integration Tests")
    print("=" * 60)
    
    try:
        # Test 1: Instantiation
        model = test_model_instantiation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Various input shapes
        test_output_shapes()
        
        # Test 4: Gradient flow
        test_gradient_flow(model)
        
        # Test 5: Checkpoint save/load
        test_checkpoint_save_load()
        
        # Test 6: Predict pick
        test_predict_pick()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
