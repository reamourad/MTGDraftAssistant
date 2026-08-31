from typing import List, Tuple

import numpy as np
import torch

from ..card_encoder.card_encoder import CardEncoder
from .oracle_text_projection import OracleTextProjection

# Known maximums across all current sets (checked directly against the real data):
# set context: largest card_list.json is Powered_Cube at 545 cards; 600 leaves headroom.
# pool: largest draft is Powered_Cube at 3 packs x 15 picks = 45.
# pack: largest pack is Powered_Cube at 15 cards; "other pack" excludes the candidate itself.
MAX_SET_SIZE = 600
MAX_POOL_SIZE = 45
MAX_PACK_SIZE = 15
MAX_OTHER_PACK_SIZE = MAX_PACK_SIZE - 1

ROLE_SET = torch.tensor([1.0, 0.0, 0.0, 0.0])
ROLE_POOL = torch.tensor([0.0, 1.0, 0.0, 0.0])
ROLE_CANDIDATE = torch.tensor([0.0, 0.0, 1.0, 0.0])
ROLE_OTHER_PACK = torch.tensor([0.0, 0.0, 0.0, 1.0])

#this is the sum of the features dimensions, the oracle text dimension and the role dimensions
TOKEN_DIM = CardEncoder.STRUCTURED_DIM + OracleTextProjection.OUTPUT_DIM + ROLE_SET.shape[0]


class SequenceBuilder:
    def __init__(self, projection: OracleTextProjection):
        # projection's weights change during training, so we use a pointer of the class
        self.projection = projection

    #Get all cards
    def build_full_sequence(
        self, set_cards: List[np.ndarray], pool_cards: List[np.ndarray], candidate_card: np.ndarray, other_pack_cards: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        #get all the padded sequences
        set_seq, set_mask = self.build_padded_sequence(set_cards, ROLE_SET, MAX_SET_SIZE)
        pool_seq, pool_mask = self.build_padded_sequence(pool_cards, ROLE_POOL, MAX_POOL_SIZE)
        other_pack_seq, other_pack_mask = self.build_padded_sequence(other_pack_cards, ROLE_OTHER_PACK, MAX_OTHER_PACK_SIZE)
        # candidate goes last, unpadded (always exactly 1 token) — so its position in the final sequence is always just sequence[-1]
        candidate_seq, candidate_mask = self.build_padded_sequence([candidate_card], ROLE_CANDIDATE, 1)

        #The sequence and the mask has the same length of elements, mask just tells you if there is a card at x index
        full_sequence = torch.cat([set_seq, pool_seq, other_pack_seq, candidate_seq], dim=0)
        full_mask = torch.cat([set_mask, pool_mask, other_pack_mask, candidate_mask], dim=0)
        return full_sequence, full_mask


    def build_padded_sequence(self, card_vectors: List[np.ndarray], role: torch.Tensor, max_length: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(card_vectors) > max_length:
            raise ValueError(f"Got {len(card_vectors)} cards, but max_length is {max_length}")

        #this is a 2d matrix, each row is as long as TOKEN_DIM and there is max_length column, it represents all our cards
        padded = torch.zeros(max_length, TOKEN_DIM)

        #this is a list of max_length bools which tells us which column is used or not
        mask = torch.zeros(max_length, dtype=torch.bool)

        if not card_vectors:
            return padded, mask

        stacked = np.stack(card_vectors, axis=0)
        structured = torch.from_numpy(stacked[:, :CardEncoder.STRUCTURED_DIM]).float()
        oracle_text = torch.from_numpy(stacked[:, CardEncoder.STRUCTURED_DIM:]).float()
        compressed_text = self.projection(oracle_text)

        n = len(card_vectors)
        roles = role.unsqueeze(0).expand(n, -1)
        tokens = torch.cat([structured, compressed_text, roles], dim=1)

        padded[:n] = tokens
        mask[:n] = True

        return padded, mask
