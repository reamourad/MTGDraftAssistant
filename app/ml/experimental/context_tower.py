import torch
import torch.nn as nn


class ContextTower(nn.Module):
    """
    Improved Context Tower with attention over pool and pack cards.
    
    FIX 2: Uses attention mechanism instead of mean pooling to better
    capture the importance of different cards in the pool and pack.
    """

    def __init__(self, card_dim=407, hidden_dim=256, output_dim=128, max_picks=45, dropout=0.2):
        super().__init__()

        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Pick number embedding layer (45 picks → 16 dims)
        self.pick_embedding = nn.Embedding(max_picks + 1, 16)

        # FIX 2: Project card embeddings to a dimension divisible by num_heads
        # 407 is not divisible by 4, so project to 408 (divisible by 4)
        attention_dim = 408
        self.pool_projection = nn.Linear(card_dim, attention_dim)
        self.pack_projection = nn.Linear(card_dim, attention_dim)
        
        # FIX 2: Attention over pool cards instead of mean pooling
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # FIX 2: Attention over pack cards
        self.pack_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # FIX 2: Learnable queries for attention
        self.pool_query = nn.Parameter(torch.randn(1, 1, attention_dim))
        self.pack_query = nn.Parameter(torch.randn(1, 1, attention_dim))

        # Context processing network
        # Input: pool_context (408) + pack_context (408) + pick_emb (16) = 832
        context_input_dim = attention_dim * 2 + 16
        
        self.network = nn.Sequential(
            nn.LayerNorm(context_input_dim),  # FIX 3: Add normalization
            nn.Linear(context_input_dim, hidden_dim * 2),  # 832 → 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),  # 256 → 128
        )

    def forward(self, pool_cards, pack_cards, pick_number):
        """
        Args:
            pool_cards: (batch, num_picked, 407)
            pack_cards: (batch, num_available, 407)
            pick_number: (batch, 1)
        
        Returns:
            context_embedding: (batch, 128)
        """
        batch_size = pool_cards.shape[0]
        device = pool_cards.device
        dtype = pool_cards.dtype

        # FIX 2: Use attention instead of mean pooling

        # Handle empty pool (first pick)
        if pool_cards.shape[1] == 0:
            # For empty pool, use zeros (no attention needed)
            pool_context = torch.zeros(batch_size, 408, device=device, dtype=dtype)
        else:
            # Project pool cards to attention dimension
            pool_projected = self.pool_projection(pool_cards)  # (batch, num_picked, 408)
            
            # Create mask for padded cards in pool
            pool_mask = (pool_cards.abs().sum(dim=-1) == 0)  # (batch, num_picked)
            
            # Check if ALL cards are masked (shouldn't happen but be safe)
            if pool_mask.all():
                pool_context = torch.zeros(batch_size, 408, device=device, dtype=dtype)
            else:
                # Expand query to batch size
                pool_query_expanded = self.pool_query.expand(batch_size, -1, -1)  # (batch, 1, 408)
                
                # Apply attention over pool
                pool_context, _ = self.pool_attention(
                    query=pool_query_expanded,
                    key=pool_projected,
                    value=pool_projected,
                    key_padding_mask=pool_mask
                )
                pool_context = pool_context.squeeze(1)  # (batch, 408)

        # Handle pack cards
        if pack_cards.shape[1] == 0:
            pack_context = torch.zeros(batch_size, 408, device=device, dtype=dtype)
        else:
            # Project pack cards to attention dimension
            pack_projected = self.pack_projection(pack_cards)  # (batch, num_available, 408)
            
            # Create mask for padded cards in pack
            pack_mask = (pack_cards.abs().sum(dim=-1) == 0)  # (batch, num_available)
            
            # Check if ALL cards are masked (shouldn't happen but be safe)
            if pack_mask.all():
                pack_context = torch.zeros(batch_size, 408, device=device, dtype=dtype)
            else:
                # Expand query to batch size
                pack_query_expanded = self.pack_query.expand(batch_size, -1, -1)
                
                # Apply attention over pack
                pack_context, _ = self.pack_attention(
                    query=pack_query_expanded,
                    key=pack_projected,
                    value=pack_projected,
                    key_padding_mask=pack_mask
                )
                pack_context = pack_context.squeeze(1)  # (batch, 408)

        # Embed pick number
        pick_emb = self.pick_embedding(pick_number.long().squeeze(-1))  # (batch, 16)

        # Concatenate all context features
        context = torch.cat([
            pool_context,   # 408 dims: weighted summary of pool
            pack_context,   # 408 dims: weighted summary of pack
            pick_emb        # 16 dims: where in the draft
        ], dim=1)  # (batch, 832)

        # Process through network to get context embedding
        context_embedding = self.network(context)  # (batch, 832) → (batch, 128)

        return context_embedding