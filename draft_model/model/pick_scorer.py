import torch
import torch.nn as nn

from .sequence_builder import TOKEN_DIM

NUM_HEADS = 7  # TOKEN_DIM (91)
NUM_LAYERS = 2
FEEDFORWARD_DIM = 256
DROPOUT = 0.1


class PickScorer(nn.Module):
    """
    Scores one candidate card as good/bad, given the whole role-tagged sequence
    (set context + pool + other pack cards + candidate) built by SequenceBuilder.
    Pointwise: this only ever looks at one candidate at a time, with no memory
    of other calls for the same pack.
    """

    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TOKEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.scoring_head = nn.Linear(TOKEN_DIM, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, seq_len, TOKEN_DIM), mask: (batch, seq_len), True = real card
        # nn.TransformerEncoder's src_key_padding_mask uses the OPPOSITE convention
        # (True = ignore this position), so we have to flip it here.
        padding_mask = ~mask

        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)

        # the candidate is always the last token — see sequence_builder.build_full_sequence
        candidate_representation = encoded[:, -1, :]

        score = self.scoring_head(candidate_representation)
        return score.squeeze(-1)
