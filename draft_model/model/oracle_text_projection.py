import torch
import torch.nn as nn

from ..card_encoder.card_encoder import CardEncoder

INPUT_DIM = CardEncoder.ORACLE_TEXT_DIM
OUTPUT_DIM = 64


class OracleTextProjection(nn.Module):
    """
    Trainable compression of CardEncoder's raw 384-dim oracle text embedding
    down to OUTPUT_DIM dims. Applied per-card, before sequence assembly —
    not inside the transformer's forward pass — so padded sequences carry
    the compact vector, not the full 384-dim one.
    """

    OUTPUT_DIM = OUTPUT_DIM

    def __init__(self, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, output_dim)

    def forward(self, oracle_text_vec: torch.Tensor) -> torch.Tensor:
        return self.linear(oracle_text_vec)
