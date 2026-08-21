"""
PyTorch Prediction Service for Two-Tower Model

This module provides the prediction service for the PyTorch two-tower architecture,
handling card encoding, model inference, and result formatting for the API.
"""

import torch
import numpy as np
import logging
from typing import List, Optional
from dataclasses import dataclass

from app.ml.experimental.model_loader import PyTorchModelLoader, ModelLoadError
from app.ml.experimental.card_encoder import CardEncoder, CardEncoderError


logger = logging.getLogger(__name__)


@dataclass
class CardPrediction:
    """Prediction result for a single card."""
    card_name: str
    probability: float


class PyTorchPredictionError(Exception):
    """Raised when prediction fails."""
    pass


class PyTorchPredictionService:
    """
    Prediction service using PyTorch two-tower model.
    
    This service handles:
    - Loading the general PyTorch model
    - Encoding cards using CardEncoder
    - Running inference through the two-tower model
    - Converting scores to probabilities
    - Formatting results for API responses
    """
    
    def __init__(
        self,
        model_loader: PyTorchModelLoader,
        card_encoder: CardEncoder
    ):
        """
        Initialize the prediction service.
        
        Args:
            model_loader: PyTorchModelLoader instance for loading models
            card_encoder: CardEncoder instance for encoding cards
        """
        self.model_loader = model_loader
        self.encoder = card_encoder
        logger.info("PyTorchPredictionService initialized")
    
    def predict_picks(
        self,
        set_code: str,
        deck: List[str],
        pack: List[str]
    ) -> List[CardPrediction]:
        """
        Predict best picks using PyTorch model.
        
        Args:
            set_code: MTG set code (may be ignored for general model)
            deck: List of card names in pool
            pack: List of card names in current pack
        
        Returns:
            List of predictions sorted by score (highest first)
        
        Raises:
            PyTorchPredictionError: If prediction fails
        """
        try:
            # Load the general model
            logger.debug(f"Loading model for prediction (set_code: {set_code})")
            try:
                model = self.model_loader.load_model("general")
            except ModelLoadError:
                raise PyTorchPredictionError(
                    "General PyTorch model not available. Please train the model first."
                )
            
            # Validate inputs
            if not pack:
                raise PyTorchPredictionError("Pack cannot be empty")
            
            # Encode cards
            logger.debug(f"Encoding {len(deck)} pool cards and {len(pack)} pack cards")
            try:
                pool_encoded = self.encoder.encode_batch_by_names(deck) if deck else None
                pack_encoded = self.encoder.encode_batch_by_names(pack)
                
                # Check for NaN in encodings
                if pool_encoded is not None:
                    if np.isnan(pool_encoded).any():
                        logger.error("NaN detected in pool encodings")
                        raise PyTorchPredictionError("Invalid pool card encodings")
                
                if np.isnan(pack_encoded).any():
                    logger.error("NaN detected in pack encodings")
                    raise PyTorchPredictionError("Invalid pack card encodings")
                    
            except CardEncoderError as e:
                raise PyTorchPredictionError(f"Failed to encode cards: {str(e)}")
            
            # Convert to tensors
            device = next(model.parameters()).device
            pack_tensor = torch.tensor(pack_encoded, dtype=torch.float32, device=device)
            
            if pool_encoded is not None:
                pool_tensor = torch.tensor(pool_encoded, dtype=torch.float32, device=device)
            else:
                # Empty pool - create zero tensor
                pool_tensor = torch.zeros((1, 407), dtype=torch.float32, device=device)
            
            # Calculate pick number (1-indexed)
            pick_number = len(deck) + 1
            
            # Get predictions from model
            logger.debug(f"Running inference for pick {pick_number}")
            with torch.no_grad():
                scores = model.predict_pick(
                    pack_tensor,
                    pool_tensor,
                    pack_tensor,  # candidates are the pack cards
                    pick_number
                )
            
            # Check for NaN or Inf values
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                logger.error(f"Model produced invalid scores (NaN or Inf). This usually means the model architecture has changed.")
                raise PyTorchPredictionError(
                    "Model produced invalid predictions. The model may be incompatible with the current code. "
                    "Please retrain the model: python scripts/train_pytorch.py --sets TLA TDM MH3 FIN --epochs 50 --lr 0.0003"
                )
            
            # Log raw scores for debugging
            logger.debug(f"Raw scores - min: {scores.min().item():.4f}, max: {scores.max().item():.4f}, mean: {scores.mean().item():.4f}")
            
            # Convert scores to probabilities using softmax (not sigmoid!)
            # The model outputs logits for CrossEntropyLoss
            probabilities = torch.softmax(scores, dim=0).cpu().numpy()
            
            # Check probabilities for NaN
            if np.isnan(probabilities).any():
                logger.error("Softmax produced NaN probabilities")
                raise PyTorchPredictionError("Invalid probability computation")
            
            # Create predictions
            predictions = []
            for idx, prob in enumerate(probabilities):
                # Ensure probability is a valid float
                prob_value = float(prob)
                if not np.isfinite(prob_value):
                    prob_value = 0.0
                    
                predictions.append(CardPrediction(
                    card_name=pack[idx],
                    probability=prob_value
                ))
            
            # Sort by probability (highest first)
            predictions.sort(key=lambda x: x.probability, reverse=True)
            
            logger.info(f"Generated {len(predictions)} predictions for pick {pick_number}")
            logger.debug(f"Top 3 predictions: {[(p.card_name, f'{p.probability:.4f}') for p in predictions[:3]]}")
            return predictions
            
        except PyTorchPredictionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
            raise PyTorchPredictionError(f"Prediction failed: {str(e)}")
    
    def is_model_available(self, set_code: str = "general") -> bool:
        """
        Check if a model is available for predictions.
        
        Args:
            set_code: Set code to check (default: "general")
        
        Returns:
            True if model is available, False otherwise
        """
        return self.model_loader.is_model_available(set_code)
