"""Test model inference to debug NaN issue."""
import torch
import numpy as np
from app.ml.experimental.two_tower_model import TwoTowerModel

# Load model
print("Loading model...")
model = TwoTowerModel.load_checkpoint('app/models/general/best_model.pt', device=torch.device('cpu'))
model.eval()

# Create dummy input (14 cards in pack, empty pool)
batch_size = 1
num_cards = 14
card_dim = 407

pack_cards = torch.randn(batch_size, num_cards, card_dim)
pool_cards = torch.zeros(batch_size, 0, card_dim)  # Empty pool
pick_number = torch.tensor([[1]], dtype=torch.float32)

print(f"\nInput shapes:")
print(f"  pack_cards: {pack_cards.shape}")
print(f"  pool_cards: {pool_cards.shape}")
print(f"  pick_number: {pick_number.shape}")

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    try:
        scores = model.predict_pick(
            candidate_cards=pack_cards.squeeze(0),
            pool_cards=pool_cards.squeeze(0),
            pack_cards=pack_cards.squeeze(0),
            pick_number=1
        )
        
        print(f"\nScores shape: {scores.shape}")
        print(f"Scores: {scores}")
        print(f"Has NaN: {torch.isnan(scores).any()}")
        print(f"Has Inf: {torch.isinf(scores).any()}")
        print(f"Min: {scores.min().item():.4f}")
        print(f"Max: {scores.max().item():.4f}")
        print(f"Mean: {scores.mean().item():.4f}")
        
        # Try softmax
        probs = torch.softmax(scores, dim=0)
        print(f"\nProbabilities: {probs}")
        print(f"Probs sum: {probs.sum().item():.4f}")
        print(f"Probs has NaN: {torch.isnan(probs).any()}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
