"""
Training infrastructure for PyTorch two-tower model.

This module provides data structures and utilities for training
the two-tower draft pick prediction model.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DraftSequence:
    """
    Represents a single pick decision in a draft.
    
    Each DraftSequence captures the state at one moment in a draft:
    - The cards already picked (pool)
    - The cards available to pick from (pack)
    - The card that was actually picked
    - Metadata about the draft and pick number
    """
    
    draft_id: str
    pick_number: int
    pool: List[str]  # Card names already picked
    pack: List[str]  # Card names available in current pack
    picked_card: str  # Card that was actually picked
    
    def validate(self) -> bool:
        """
        Validate that the picked card is in the pack.
        
        Returns:
            True if valid, False otherwise
        
        Raises:
            ValueError: If validation fails with details
        """
        if not self.picked_card:
            raise ValueError(f"Draft {self.draft_id} pick {self.pick_number}: picked_card is empty")
        
        if not self.pack:
            raise ValueError(f"Draft {self.draft_id} pick {self.pick_number}: pack is empty")
        
        if self.picked_card not in self.pack:
            raise ValueError(
                f"Draft {self.draft_id} pick {self.pick_number}: "
                f"picked_card '{self.picked_card}' not found in pack of {len(self.pack)} cards"
            )
        
        if self.pick_number < 1:
            raise ValueError(
                f"Draft {self.draft_id}: pick_number must be >= 1, got {self.pick_number}"
            )
        
        return True


__all__ = ['DraftSequence']

# Import submodules for easier access
from .config import TrainingConfig
from .dataset import DraftDataset, train_test_split
from .data_loader import DraftDataLoader
from .trainer import TwoTowerTrainer
from .evaluator import ModelEvaluator

__all__.extend([
    'TrainingConfig',
    'DraftDataset',
    'train_test_split',
    'DraftDataLoader',
    'TwoTowerTrainer',
    'ModelEvaluator'
])
