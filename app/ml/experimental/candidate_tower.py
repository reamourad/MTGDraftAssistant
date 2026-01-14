# Problem: Cards are 424 dimensions
# - That's a LOT of numbers
# - Hard for the model to work with directly
# - Many dimensions might be redundant

# Solution: Compress to 128 dimensions
# - Keep the important information
# - Throw away the noise
# - Easier for the model to learn patterns

import torch
import torch.nn as nn

class CandidateTower(nn.Module):
    def __init__(self, input_dim=407, hidden_dim=256, output_dim=128):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, card):
        out = self.layer1(card)
        out = self.layer2(out)
        out = self.layer3(out)

        return out