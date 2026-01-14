import torch
import torch.nn as nn


class ContextTower(nn.Module):

    def __init__(self, card_dim=407, hidden_dim=256, output_dim=128):
        super().__init__()

        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pool_encoder = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),  # 407 → 256
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.pack_encoder = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),  # 407 → 256
            nn.ReLU(),
            nn.Dropout(0.2),
        )


        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  #  407 → 256
            nn.ReLU(),
            nn.Dropout(0.2),

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
            pool_mean = pool_cards.mean(dim=1)  # (batch, num_picked, 424) → (batch, 424)

        # Handle empty pack (shouldn't happen, but be safe)
        if pack_cards.shape[1] == 0:
            pack_mean = torch.zeros(batch_size, self.card_dim, device=device, dtype=dtype)
        else:
            pack_mean = pack_cards.mean(dim=1)  # (batch, num_available, 424) → (batch, 424)

        # Step 2: Encode pool and pack separately
        pool_features = self.pool_encoder(pool_mean)  # (batch, 424) → (batch, 256)
        pack_features = self.pack_encoder(pack_mean)  # (batch, 424) → (batch, 256)

        # Step 3: Concatenate all context information
        combined = torch.cat([
            pool_features,  # 256 dims: "what I have"
            pack_features,  # 256 dims: "what's available"
            pick_number / 45  # 1 dim: "where am I in the draft"
        ], dim=1)  # (batch, 513)

        # Step 4: Final transformation to context embedding
        context_embedding = self.combiner(combined)  # (batch, 513) → (batch, 128)

        return context_embedding