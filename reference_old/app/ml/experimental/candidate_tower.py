# Problem: Cards are 407 dimensions
# - That's a LOT of numbers
# - Hard for the model to work with directly
# - Many dimensions might be redundant

# Solution: Compress to 128 dimensions with proper normalization
# - Keep the important information
# - Normalize inputs for stable training
# - Throw away the noise
# - Easier for the model to learn patterns

import torch
import torch.nn as nn

class CandidateTower(nn.Module):
    """
    Improved Candidate Tower with normalization.
    
    Encodes individual card features into a lower-dimensional embedding.
    """
    
    def __init__(self, input_dim=407, hidden_dim=256, output_dim=128, dropout=0.2):
        super().__init__()

        # FIX 3: Add input normalization for stable training
        self.input_norm = nn.LayerNorm(input_dim)

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # FIX 3: Add normalization after each layer
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # FIX 3: Add normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, card):
        """
        Args:
            card: (batch, 407) or (batch*num_candidates, 407)
        
        Returns:
            embedding: (batch, 128) or (batch*num_candidates, 128)
        """
        # FIX 3: Normalize input features
        out = self.input_norm(card)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out