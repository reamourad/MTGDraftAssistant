"""
Two-Tower Model for MTG Draft Pick Prediction

This module implements a unified two-tower architecture that combines:
1. CandidateTower: Encodes individual card features
2. ContextTower: Encodes draft context (pool, pack, pick number)
3. ScoringHead: Combines embeddings to score picks

Architecture:
    Card (407-dim) → CandidateTower → Candidate Embedding (128-dim)
    Context → ContextTower → Context Embedding (128-dim)
    [Candidate + Context] → ScoringHead → Score (1-dim)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import os

from .candidate_tower import CandidateTower
from .context_tower import ContextTower
from .scoring_head import ScoringHead


class TwoTowerModel(nn.Module):
    """
    Unified two-tower model for draft pick prediction.
    
    The model processes cards and draft context separately through two towers,
    then combines their embeddings to produce pick scores.
    """
    
    def __init__(
        self,
        card_dim: int = 407,
        hidden_dim: int = 256,
        embedding_dim: int = 128
    ):
        """
        Initialize the two-tower model.
        
        Args:
            card_dim: Dimension of card feature vectors (default: 407)
            hidden_dim: Hidden layer dimension (default: 256)
            embedding_dim: Output embedding dimension for both towers (default: 128)
        """
        super().__init__()
        
        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Initialize the three components
        self.candidate_tower = CandidateTower(
            input_dim=card_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        )
        
        self.context_tower = ContextTower(
            card_dim=card_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        )
        
        self.scoring_head = ScoringHead(
            input_dim=embedding_dim * 2,  # Concatenated embeddings
            hidden_dim=hidden_dim // 2
        )
    
    def forward(
        self,
        candidate_cards: torch.Tensor,
        pool_cards: torch.Tensor,
        pack_cards: torch.Tensor,
        pick_number: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the two-tower model.
        
        Args:
            candidate_cards: Tensor of shape (batch, num_candidates, card_dim)
                Cards to score for picking
            pool_cards: Tensor of shape (batch, num_picked, card_dim)
                Cards already picked in the draft
            pack_cards: Tensor of shape (batch, num_available, card_dim)
                Cards available in the current pack
            pick_number: Tensor of shape (batch, 1)
                Current pick number in the draft (1-45)
        
        Returns:
            scores: Tensor of shape (batch, num_candidates, 1)
                Predicted scores for each candidate card
        """
        batch_size, num_candidates, _ = candidate_cards.shape
        
        # Encode context once (same for all candidates)
        context_embedding = self.context_tower(
            pool_cards, pack_cards, pick_number
        )  # (batch, embedding_dim)
        
        # Encode each candidate card
        # Reshape to process all candidates at once
        candidates_flat = candidate_cards.reshape(batch_size * num_candidates, self.card_dim)
        candidate_embeddings = self.candidate_tower(candidates_flat)
        candidate_embeddings = candidate_embeddings.reshape(batch_size, num_candidates, self.embedding_dim)
        
        # Expand context to match number of candidates
        context_expanded = context_embedding.unsqueeze(1).expand(
            batch_size, num_candidates, self.embedding_dim
        )
        
        # Score each candidate with the context
        # Reshape for scoring head
        candidate_flat = candidate_embeddings.reshape(batch_size * num_candidates, self.embedding_dim)
        context_flat = context_expanded.reshape(batch_size * num_candidates, self.embedding_dim)
        
        scores_flat = self.scoring_head(candidate_flat, context_flat)
        scores = scores_flat.reshape(batch_size, num_candidates, 1)
        
        return scores
    
    def predict_pick(
        self,
        candidate_cards: torch.Tensor,
        pool_cards: torch.Tensor,
        pack_cards: torch.Tensor,
        pick_number: int
    ) -> torch.Tensor:
        """
        Predict pick scores for a single draft state.
        
        Args:
            candidate_cards: Tensor of shape (num_candidates, card_dim)
            pool_cards: Tensor of shape (num_picked, card_dim)
            pack_cards: Tensor of shape (num_available, card_dim)
            pick_number: Current pick number (1-45)
        
        Returns:
            scores: Tensor of shape (num_candidates,)
                Predicted scores for each candidate
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            candidate_cards = candidate_cards.unsqueeze(0)
            pool_cards = pool_cards.unsqueeze(0)
            pack_cards = pack_cards.unsqueeze(0)
            pick_number_tensor = torch.tensor([[pick_number]], dtype=torch.float32, device=candidate_cards.device)
            
            scores = self.forward(candidate_cards, pool_cards, pack_cards, pick_number_tensor)
            return scores.squeeze(0).squeeze(-1)  # Remove batch and score dimensions
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """
        Save model checkpoint with metadata.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional dictionary with training metadata
                (e.g., epoch, loss, set_code)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'card_dim': self.card_dim,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim
            }
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> 'TwoTowerModel':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model onto (default: CPU)
        
        Returns:
            Loaded TwoTowerModel instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        # Extract config
        config = checkpoint.get('config', {})
        
        # Create model with saved config
        model = cls(
            card_dim=config.get('card_dim', 407),
            hidden_dim=config.get('hidden_dim', 256),
            embedding_dim=config.get('embedding_dim', 128)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
    
    def get_metadata(self, checkpoint_path: str) -> Optional[Dict]:
        """
        Load metadata from checkpoint without loading the full model.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Metadata dictionary or None
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('metadata')
