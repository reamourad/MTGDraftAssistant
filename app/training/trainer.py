"""Training infrastructure for PyTorch two-tower model."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .dataset import DraftDataset
from .evaluator import ModelEvaluator
from ..ml.experimental.two_tower_model import TwoTowerModel


logger = logging.getLogger(__name__)


class TwoTowerTrainer:
    """
    Handles training of the two-tower model.
    
    Manages the training loop, validation, checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        train_dataset: DraftDataset,
        val_dataset: DraftDataset,
        config: TrainingConfig
    ):
        """
        Initialize the trainer.
        
        Args:
            model: TwoTowerModel instance to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
        """
        self.model = model
        self.config = config
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.use_gpu
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.use_gpu
        )
        
        # Setup device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_top1_acc': [],
            'val_top3_acc': [],
            'val_top5_acc': [],
            'learning_rate': []
        }
        
        # Checkpoint management
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints = []
        
        # Create evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            val_loader=self.val_loader,
            device=self.device
        )
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary with training history (losses, metrics)
        """
        logger.info("Starting training")
        logger.info(f"Training config:\n{self.config}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_top1_acc'].append(val_metrics['top1_acc'])
            self.training_history['val_top3_acc'].append(val_metrics['top3_acc'])
            self.training_history['val_top5_acc'].append(val_metrics['top5_acc'])
            self.training_history['learning_rate'].append(self.config.learning_rate)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Top-1: {val_metrics['top1_acc']:.4f}, "
                f"Val Top-3: {val_metrics['top3_acc']:.4f}"
            )
            
            # Save checkpoint if improved
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **val_metrics
                }
                self._save_checkpoint(epoch, metrics)
                logger.info(f"New best validation loss: {val_metrics['loss']:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epoch(s) "
                    f"(patience: {self.config.patience})"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        logger.info("Training completed")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.training_history
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            pool_cards = batch['pool_cards'].to(self.device)
            pack_cards = batch['pack_cards'].to(self.device)
            pick_number = batch['pick_number'].to(self.device)
            target_idx = batch['target_idx'].to(self.device)
            
            # Forward pass
            # Model expects: (batch, num_candidates, card_dim)
            # pack_cards is already in this format
            scores = self.model(
                candidate_cards=pack_cards,
                pool_cards=pool_cards,
                pack_cards=pack_cards,
                pick_number=pick_number
            )
            
            # Reshape scores for loss computation
            # scores: (batch, num_candidates, 1) -> (batch, num_candidates)
            scores = scores.squeeze(-1)
            
            # target_idx: (batch, 1) -> (batch,)
            target = target_idx.squeeze(-1)
            
            # Compute loss
            loss = self.criterion(scores, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch {self.current_epoch + 1} "
                    f"[{batch_idx + 1}/{len(self.train_loader)}] - "
                    f"Loss: {avg_loss:.4f}"
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self) -> Dict[str, float]:
        """
        Run validation on the validation set.
        
        Uses ModelEvaluator to compute validation loss and accuracy metrics.
        
        Returns:
            Dictionary with validation metrics:
                - loss: Average validation loss
                - top1_acc: Top-1 accuracy
                - top3_acc: Top-3 accuracy
                - top5_acc: Top-5 accuracy
        """
        return self.evaluator.evaluate()
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with training metrics
        """
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare metadata
        metadata = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            **metrics
        }
        
        # Save using model's save_checkpoint method
        self.model.save_checkpoint(str(checkpoint_path), metadata)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Track saved checkpoints
        self.saved_checkpoints.append(checkpoint_path)
        
        # Clean up old checkpoints if needed
        if self.config.max_checkpoints is not None:
            self._cleanup_old_checkpoints()
        
        # Also save as "best" checkpoint
        best_path = self.checkpoint_dir / "best_model.pt"
        self.model.save_checkpoint(str(best_path), metadata)
        logger.info(f"Saved best model: {best_path}")
    
    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints to save disk space.
        
        Keeps only the most recent N checkpoints based on config.max_checkpoints.
        Does not delete the "best_model.pt" file.
        """
        if len(self.saved_checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoints_to_remove = self.saved_checkpoints[:-self.config.max_checkpoints]
        
        for checkpoint_path in checkpoints_to_remove:
            if checkpoint_path.exists() and checkpoint_path.name != "best_model.pt":
                try:
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
        
        # Update the list
        self.saved_checkpoints = self.saved_checkpoints[-self.config.max_checkpoints:]
    
    def save_training_history(self, path: Optional[str] = None):
        """
        Save training history to a file.
        
        Args:
            path: Path to save history (default: checkpoint_dir/training_history.json)
        """
        import json
        
        if path is None:
            path = self.checkpoint_dir / "training_history.json"
        else:
            path = Path(path)
        
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Saved training history: {path}")


__all__ = ['TwoTowerTrainer']

