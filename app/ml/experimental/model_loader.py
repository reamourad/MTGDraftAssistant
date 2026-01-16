"""
PyTorch Model Loader for Two-Tower Architecture

This module handles loading, caching, and managing PyTorch model checkpoints
for the two-tower draft prediction system.
"""

import torch
import os
from typing import Optional, Dict
from pathlib import Path
import logging

from .two_tower_model import TwoTowerModel


logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class PyTorchModelLoader:
    """
    Manages loading and caching of PyTorch two-tower models.
    
    Features:
    - Automatic GPU/CPU device selection
    - Model caching to avoid repeated loading
    - Proper error handling for missing/corrupt checkpoints
    - Memory-efficient model management
    """
    
    def __init__(self, models_dir: str = "app/models", use_gpu: bool = True):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Base directory containing model checkpoints
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.models_dir = Path(models_dir)
        self.use_gpu = use_gpu
        self.device = self._get_device()
        self._model_cache: Dict[str, TwoTowerModel] = {}
        
        logger.info(f"PyTorchModelLoader initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """
        Determine the appropriate device for model loading.
        
        Returns:
            torch.device: CUDA device if available and use_gpu=True, else CPU
        """
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            if self.use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but not available, using CPU")
            else:
                logger.info("Using CPU")
        
        return device
    
    def load_model(self, set_code: str, force_reload: bool = False) -> TwoTowerModel:
        """
        Load a PyTorch model for the specified set.
        
        Args:
            set_code: MTG set code (e.g., 'MH3', 'BLB')
            force_reload: If True, bypass cache and reload from disk
        
        Returns:
            TwoTowerModel: Loaded model in eval mode
        
        Raises:
            ModelLoadError: If model checkpoint doesn't exist or loading fails
        """
        # Check cache first
        if not force_reload and set_code in self._model_cache:
            logger.debug(f"Returning cached model for {set_code}")
            return self._model_cache[set_code]
        
        # Construct checkpoint path
        checkpoint_path = self._get_checkpoint_path(set_code)
        
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise ModelLoadError(
                f"Model checkpoint not found for set '{set_code}' at {checkpoint_path}"
            )
        
        try:
            logger.info(f"Loading model for {set_code} from {checkpoint_path}")
            
            # Load model using TwoTowerModel's class method
            model = TwoTowerModel.load_checkpoint(str(checkpoint_path), device=self.device)
            
            # Ensure model is in eval mode
            model.eval()
            
            # Cache the model
            self._model_cache[set_code] = model
            
            logger.info(f"Successfully loaded model for {set_code}")
            return model
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model for set '{set_code}': {str(e)}"
            ) from e
    
    def get_cached_model(self, set_code: str) -> Optional[TwoTowerModel]:
        """
        Get a model from cache without loading from disk.
        
        Args:
            set_code: MTG set code
        
        Returns:
            Cached model or None if not in cache
        """
        return self._model_cache.get(set_code)
    
    def is_model_available(self, set_code: str) -> bool:
        """
        Check if a model checkpoint exists for the given set.
        
        Args:
            set_code: MTG set code
        
        Returns:
            True if checkpoint exists, False otherwise
        """
        checkpoint_path = self._get_checkpoint_path(set_code)
        return checkpoint_path.exists()
    
    def _get_checkpoint_path(self, set_code: str) -> Path:
        """
        Get the checkpoint path for a given set.
        
        Args:
            set_code: MTG set code
        
        Returns:
            Path to checkpoint file
        """
        # Look for best_model.pt (saved by trainer)
        return self.models_dir / set_code / "best_model.pt"
    
    def clear_cache(self, set_code: Optional[str] = None):
        """
        Clear model cache to free memory.
        
        Args:
            set_code: If provided, clear only this model. Otherwise clear all.
        """
        if set_code:
            if set_code in self._model_cache:
                del self._model_cache[set_code]
                logger.info(f"Cleared cache for {set_code}")
                
                # Force garbage collection if on GPU
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            self._model_cache.clear()
            logger.info("Cleared all model cache")
            
            # Force garbage collection if on GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def get_model_metadata(self, set_code: str) -> Optional[Dict]:
        """
        Load metadata from a model checkpoint without loading the full model.
        
        Args:
            set_code: MTG set code
        
        Returns:
            Metadata dictionary or None if not available
        """
        checkpoint_path = self._get_checkpoint_path(set_code)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint.get('metadata')
        except Exception as e:
            logger.error(f"Failed to load metadata for {set_code}: {e}")
            return None
    
    def list_available_models(self) -> list[str]:
        """
        List all sets with available model checkpoints.
        
        Returns:
            List of set codes with available models
        """
        available = []
        
        if not self.models_dir.exists():
            return available
        
        for set_dir in self.models_dir.iterdir():
            if set_dir.is_dir():
                checkpoint_path = set_dir / "best_model.pt"
                if checkpoint_path.exists():
                    available.append(set_dir.name)
        
        return sorted(available)
    
    def preload_models(self, set_codes: list[str]):
        """
        Preload multiple models into cache.
        
        Useful for warming up the cache at application startup.
        
        Args:
            set_codes: List of set codes to preload
        """
        logger.info(f"Preloading {len(set_codes)} models")
        
        for set_code in set_codes:
            try:
                self.load_model(set_code)
            except ModelLoadError as e:
                logger.warning(f"Failed to preload {set_code}: {e}")
        
        logger.info(f"Preloading complete. {len(self._model_cache)} models cached")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_models': list(self._model_cache.keys()),
            'cache_size': len(self._model_cache),
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'gpu_memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
