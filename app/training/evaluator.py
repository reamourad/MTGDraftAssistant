"""
Model evaluation and metrics computation for PyTorch two-tower model.

This module provides evaluation functionality to assess model performance
on validation data during and after training.
"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..ml.experimental.two_tower_model import TwoTowerModel


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance on validation data.
    
    Computes various metrics including loss and top-k accuracy
    without updating model parameters.
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        val_loader: DataLoader,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: TwoTowerModel instance to evaluate
            val_loader: DataLoader for validation data
            device: Device to run evaluation on (default: auto-detect)
        """
        self.model = model
        self.val_loader = val_loader
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"ModelEvaluator initialized with device: {self.device}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Computes validation loss and accuracy metrics without gradients.
        
        Returns:
            Dictionary with metrics:
                - loss: Average validation loss
                - top1_acc: Top-1 accuracy (exact match)
                - top3_acc: Top-3 accuracy
                - top5_acc: Top-5 accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Accumulators for accuracy
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                pool_cards = batch['pool_cards'].to(self.device)
                pack_cards = batch['pack_cards'].to(self.device)
                pick_number = batch['pick_number'].to(self.device)
                target_idx = batch['target_idx'].to(self.device)
                
                # Forward pass
                scores = self.model(
                    candidate_cards=pack_cards,
                    pool_cards=pool_cards,
                    pack_cards=pack_cards,
                    pick_number=pick_number
                )
                
                # Reshape for loss and metrics
                # scores: (batch, num_candidates, 1) -> (batch, num_candidates)
                scores = scores.squeeze(-1)
                # target: (batch, 1) -> (batch,)
                target = target_idx.squeeze(-1)
                
                # Compute loss
                loss = self.criterion(scores, target)
                total_loss += loss.item()
                num_batches += 1
                
                # Compute top-k accuracy
                batch_size = scores.size(0)
                top1, top3, top5 = self.compute_top_k_accuracy(scores, target)
                
                top1_correct += top1 * batch_size
                top3_correct += top3 * batch_size
                top5_correct += top5 * batch_size
                total_samples += batch_size
        
        # Compute average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        top1_acc = top1_correct / total_samples if total_samples > 0 else 0.0
        top3_acc = top3_correct / total_samples if total_samples > 0 else 0.0
        top5_acc = top5_correct / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'top1_acc': top1_acc,
            'top3_acc': top3_acc,
            'top5_acc': top5_acc
        }
        
        logger.info(
            f"Validation metrics - "
            f"Loss: {avg_loss:.4f}, "
            f"Top-1: {top1_acc:.4f}, "
            f"Top-3: {top3_acc:.4f}, "
            f"Top-5: {top5_acc:.4f}"
        )
        
        return metrics
    
    def compute_top_k_accuracy(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        k_values: tuple = (1, 3, 5)
    ) -> tuple[float, float, float]:
        """
        Compute top-k accuracy for pick predictions.
        
        Args:
            scores: (batch, num_candidates) prediction scores
            target: (batch,) target indices
            k_values: Tuple of k values to compute (default: (1, 3, 5))
        
        Returns:
            Tuple of (top1_acc, top3_acc, top5_acc) as fractions
        """
        batch_size = scores.size(0)
        num_candidates = scores.size(1)
        
        # Get top-k predictions
        # topk returns (values, indices)
        max_k = max(k_values)
        
        # Handle edge case where pack size < k
        actual_k = min(max_k, num_candidates)
        _, top_k_indices = scores.topk(actual_k, dim=1, largest=True, sorted=True)
        
        # Expand target to compare with top-k predictions
        # target: (batch,) -> (batch, 1)
        target_expanded = target.unsqueeze(1)
        
        # Check if target is in top-k
        # (batch, k) boolean tensor
        correct = top_k_indices.eq(target_expanded)
        
        # Compute accuracy for each k
        accuracies = []
        for k in k_values:
            if k > num_candidates:
                # If k > pack size, accuracy is same as for pack size
                k_to_use = num_candidates
            else:
                k_to_use = k
            
            # Check if target is in top-k_to_use
            correct_at_k = correct[:, :k_to_use].any(dim=1).float()
            accuracy = correct_at_k.sum().item() / batch_size
            accuracies.append(accuracy)
        
        return tuple(accuracies)


__all__ = ['ModelEvaluator']
