"""
PyTorch Dataset for draft pick sequences.

This module provides a PyTorch Dataset implementation for loading
and preprocessing draft data for training the two-tower model.
"""

import logging
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from . import DraftSequence
from ..ml.experimental.card_encoder import CardEncoder, CardEncoderError


logger = logging.getLogger(__name__)


class DraftDataset(Dataset):
    """
    PyTorch Dataset for draft pick sequences.
    
    Each sample represents a single pick decision:
    - Pool: Cards already picked
    - Pack: Cards available to pick from
    - Pick number: Current pick in draft (1-45)
    - Target: Index of card that was actually picked
    """
    
    def __init__(
        self,
        draft_sequences: List[DraftSequence],
        card_encodings: Dict[str, Dict],
        card_name_to_uuid: Dict[str, str],
        max_pool_size: int = 45,
        max_pack_size: int = 15
    ):
        """
        Initialize the dataset.
        
        Args:
            draft_sequences: List of DraftSequence objects
            card_encodings: Pre-computed card encodings (UUID -> encoding dict)
            card_name_to_uuid: Mapping from card names to UUIDs
            max_pool_size: Maximum pool size for padding (default: 45)
            max_pack_size: Maximum pack size for padding (default: 15)
        """
        self.sequences = draft_sequences
        self.card_encodings = card_encodings
        self.card_name_to_uuid = card_name_to_uuid
        self.max_pool_size = max_pool_size
        self.max_pack_size = max_pack_size
        
        # Filter out sequences with encoding errors
        self.valid_indices = []
        self._validate_sequences()
        
        logger.info(
            f"DraftDataset initialized with {len(self.valid_indices)}/{len(draft_sequences)} "
            f"valid sequences"
        )
    
    def _validate_sequences(self):
        """
        Validate sequences and filter out those with missing card encodings.
        
        This pre-validates all sequences to avoid errors during training.
        """
        for idx, seq in enumerate(self.sequences):
            try:
                # Check if all cards in pack have encodings
                if seq.pack:
                    card_name = seq.pack[0]
                    if card_name not in self.card_name_to_uuid:
                        logger.debug(f"Skipping sequence {idx}: Card '{card_name}' not in name->UUID mapping")
                        continue
                    uuid = self.card_name_to_uuid[card_name]
                    if uuid not in self.card_encodings:
                        logger.debug(f"Skipping sequence {idx}: UUID '{uuid}' not in encodings")
                        continue
                self.valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"Skipping sequence {idx}: {e}")
                continue
    
    def __len__(self) -> int:
        """Return the number of valid sequences."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample using pre-computed encodings.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with:
                - pool_cards: (max_pool_size, 407) padded pool encodings
                - pack_cards: (max_pack_size, 407) padded pack encodings
                - pick_number: (1,) current pick number
                - target_idx: (1,) index of picked card in pack
        
        Raises:
            KeyError: If card encoding not found
        """
        # Get the actual sequence index
        seq_idx = self.valid_indices[idx]
        seq = self.sequences[seq_idx]
        
        try:
            # Encode pool cards using pre-computed encodings
            if seq.pool:
                pool_encoded = []
                for card_name in seq.pool:
                    uuid = self.card_name_to_uuid.get(card_name)
                    if uuid and uuid in self.card_encodings:
                        pool_encoded.append(self.card_encodings[uuid]['encoding'])
                    else:
                        logger.warning(f"Card '{card_name}' not found in encodings, using zero vector")
                        pool_encoded.append(np.zeros(407, dtype=np.float32))
                pool_encoded = np.array(pool_encoded, dtype=np.float32)
            else:
                # Handle empty pool (first pick)
                pool_encoded = np.zeros((0, 407), dtype=np.float32)
            
            # Encode pack cards using pre-computed encodings
            pack_encoded = []
            for card_name in seq.pack:
                uuid = self.card_name_to_uuid.get(card_name)
                if uuid and uuid in self.card_encodings:
                    pack_encoded.append(self.card_encodings[uuid]['encoding'])
                else:
                    logger.warning(f"Card '{card_name}' not found in encodings, using zero vector")
                    pack_encoded.append(np.zeros(407, dtype=np.float32))
            pack_encoded = np.array(pack_encoded, dtype=np.float32)
            
            # Pad to max sizes
            pool_padded = self._pad_cards(pool_encoded, self.max_pool_size)
            pack_padded = self._pad_cards(pack_encoded, self.max_pack_size)
            
            # Find target index
            target_idx = seq.pack.index(seq.picked_card)
            
            return {
                'pool_cards': torch.tensor(pool_padded, dtype=torch.float32),
                'pack_cards': torch.tensor(pack_padded, dtype=torch.float32),
                'pick_number': torch.tensor([seq.pick_number], dtype=torch.float32),
                'target_idx': torch.tensor([target_idx], dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Failed to process sequence {seq_idx}: {e}")
            raise
    
    def _pad_cards(self, cards: np.ndarray, max_size: int) -> np.ndarray:
        """
        Pad card encodings to a fixed size.
        
        Args:
            cards: Array of shape (num_cards, 407)
            max_size: Target size for padding
        
        Returns:
            Padded array of shape (max_size, 407)
        """
        num_cards = cards.shape[0]
        card_dim = 407  # Fixed encoding dimension
        
        if num_cards == 0:
            # Return all zeros if no cards
            return np.zeros((max_size, card_dim), dtype=np.float32)
        
        if num_cards > max_size:
            # Truncate if too many cards (shouldn't happen in normal drafts)
            logger.warning(f"Truncating {num_cards} cards to {max_size}")
            return cards[:max_size]
        
        if num_cards == max_size:
            return cards
        
        # Pad with zeros
        padding = np.zeros(
            (max_size - num_cards, card_dim),
            dtype=np.float32
        )
        return np.vstack([cards, padding])


def train_test_split(
    sequences: List[DraftSequence],
    train_ratio: float = 0.8,
    stratify_by_set: bool = True,
    random_seed: int = 42
) -> tuple[List[DraftSequence], List[DraftSequence]]:
    """
    Split draft sequences into training and validation sets.
    
    Args:
        sequences: List of DraftSequence objects
        train_ratio: Ratio of data to use for training (default: 0.8)
        stratify_by_set: Whether to stratify by set code in draft_id (default: True)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_sequences, val_sequences)
    """
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    if not stratify_by_set:
        # Simple random split
        shuffled = sequences.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_sequences = shuffled[:split_idx]
        val_sequences = shuffled[split_idx:]
        
        logger.info(
            f"Split {len(sequences)} sequences into "
            f"{len(train_sequences)} train / {len(val_sequences)} val"
        )
        
        return train_sequences, val_sequences
    
    # Stratified split by set code
    # Group sequences by set (extracted from draft_id)
    set_groups = {}
    for seq in sequences:
        # Extract set code from draft_id (assumes format like "MH3_...")
        set_code = seq.draft_id.split('_')[0] if '_' in seq.draft_id else 'unknown'
        
        if set_code not in set_groups:
            set_groups[set_code] = []
        set_groups[set_code].append(seq)
    
    # Split each set group
    train_sequences = []
    val_sequences = []
    
    for set_code, set_seqs in set_groups.items():
        # Shuffle within set
        shuffled = set_seqs.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * train_ratio)
        train_sequences.extend(shuffled[:split_idx])
        val_sequences.extend(shuffled[split_idx:])
        
        logger.info(
            f"Set {set_code}: split {len(set_seqs)} sequences into "
            f"{split_idx} train / {len(set_seqs) - split_idx} val"
        )
    
    # Shuffle combined sets
    random.shuffle(train_sequences)
    random.shuffle(val_sequences)
    
    logger.info(
        f"Total: {len(train_sequences)} train / {len(val_sequences)} val sequences"
    )
    
    return train_sequences, val_sequences


__all__ = ['DraftDataset', 'train_test_split']
