import torch
import torch.nn as nn


class ContextTower(nn.Module):

    def __init__(self, card_dim=407, hidden_dim=256, output_dim=128, max_picks=45, dropout=0.2):
        super().__init__()

        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Pick number embedding layer (45 picks → 16 dims)
        self.pick_embedding = nn.Embedding(max_picks + 1, 16)

        # Context processing network
        # Input: pool_mean (407) + pack_mean (407) + pick_emb (16) = 830
        context_input_dim = card_dim * 2 + 16
        
        self.network = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim * 2),  # 830 → 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),  # 256 → 128
        )

    def forward(self, pool_cards, pack_cards, pick_number):

        batch_size = pool_cards.shape[0]
        device = pool_cards.device
        dtype = pool_cards.dtype

        # Step 1: Mean pooling over cards

        # Handle empty pool (first pick)
        if pool_cards.shape[1] == 0:
            pool_mean = torch.zeros(batch_size, self.card_dim, device=device, dtype=dtype)
        else:
            pool_mean = pool_cards.mean(dim=1)  # (batch, num_picked, 407) → (batch, 407)

        # Handle empty pack (shouldn't happen, but be safe)
        if pack_cards.shape[1] == 0:
            pack_mean = torch.zeros(batch_size, self.card_dim, device=device, dtype=dtype)
        else:
            pack_mean = pack_cards.mean(dim=1)  # (batch, num_available, 407) → (batch, 407)

        # Step 2: Embed pick number
        pick_emb = self.pick_embedding(pick_number.long().squeeze(-1))  # (batch, 16)

        # Step 3: Concatenate all context features
        context = torch.cat([
            pool_mean,      # 407 dims: "what I have"
            pack_mean,      # 407 dims: "what's available"
            pick_emb        # 16 dims: "where am I in the draft"
        ], dim=1)  # (batch, 830)

        # Step 4: Process through network to get context embedding
        context_embedding = self.network(context)  # (batch, 830) → (batch, 128)

        return context_embedding