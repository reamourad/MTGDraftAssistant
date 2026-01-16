"""Training configuration for PyTorch two-tower model."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for training the two-tower model.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training and validation
        learning_rate: Learning rate for Adam optimizer
        patience: Number of epochs to wait for improvement before early stopping
        use_gpu: Whether to use GPU if available
        num_workers: Number of worker processes for data loading
        checkpoint_dir: Directory to save model checkpoints
        log_interval: Number of batches between logging updates
        save_best_only: Whether to save only the best checkpoint or all checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (None = keep all)
    """
    
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 5
    use_gpu: bool = True
    num_workers: int = 4
    checkpoint_dir: str = "app/models/general"
    log_interval: int = 10
    save_best_only: bool = True
    max_checkpoints: Optional[int] = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            TrainingConfig instance
        """
        # Filter out any keys that aren't valid TrainingConfig fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"TrainingConfig(\n"
            f"  epochs={self.epochs},\n"
            f"  batch_size={self.batch_size},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  patience={self.patience},\n"
            f"  use_gpu={self.use_gpu},\n"
            f"  num_workers={self.num_workers},\n"
            f"  checkpoint_dir='{self.checkpoint_dir}',\n"
            f"  log_interval={self.log_interval},\n"
            f"  save_best_only={self.save_best_only},\n"
            f"  max_checkpoints={self.max_checkpoints}\n"
            f")"
        )
