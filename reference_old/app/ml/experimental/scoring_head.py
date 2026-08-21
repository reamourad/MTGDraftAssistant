import torch
import torch.nn as nn

class ScoringHead(nn.Module):
    # Combines candidate and context embeddings to score a pick.
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #network
        self.network = nn.Sequential(
            #Layer 1
            nn.Linear(input_dim, hidden_dim), #256 to 128
            nn.ReLU(),
            nn.Dropout(0.2),

            #Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2), #128 to 64
            nn.ReLU(),
            nn.Dropout(0.2),

            #Layer 3
            nn.Linear(hidden_dim // 2, 1) #64 -> 1
        )

    def forward(self, candidate_emb, context_emb):
        #We concat and then pass it through an MLP
        combined = torch.cat((candidate_emb, context_emb), dim=1)

        score = self.network(combined)

        return score