#Chnage this whole thing with the new model 


"""
Dependency injection for API layer.

This module handles dependency injection for services and manages
model/data caching that was previously in the API layer.
"""

from typing import Dict, Tuple
from fastapi import HTTPException
from app.ml.current.draft_data import DraftData
from app.ml.current.model_builder import ModelBuilder, TransformerBlock, PositionalEmbedding
from tensorflow.keras.models import load_model
import os
import json


class ModelCache:
    """
    Manages caching of loaded models and draft data.
    
    This removes global caches from the API layer and provides
    a clean interface for model management.
    """
    
    def __init__(self):
        self._model_cache: Dict[str, ModelBuilder] = {}
        self._draft_data_cache: Dict[str, DraftData] = {}
    
    def get_model(self, set_code: str) -> Tuple[ModelBuilder, DraftData]:
        """
        Load model and draft data for a specific set, with caching.
        
        Args:
            set_code: The set code (e.g., 'MH3', 'BLB')
            
        Returns:
            Tuple of (ModelBuilder, DraftData)
            
        Raises:
            HTTPException: If model or training data not found
        """
        set_code = set_code.upper()
        
        # Return from cache if already loaded
        if set_code in self._model_cache:
            print(f"✓ Using cached {set_code} model")
            return self._model_cache[set_code], self._draft_data_cache[set_code]
        
        # Load card list from training_cards.json
        model_dir = f"app/models/{set_code}"
        training_cards_path = f"{model_dir}/training_cards.json"
        
        if not os.path.exists(training_cards_path):
            raise HTTPException(
                status_code=404,
                detail=f"No training_cards.json found for set {set_code}. Train the model first."
            )
        
        print(f"Loading card list for {set_code} from {training_cards_path}...")
        with open(training_cards_path, 'r', encoding='utf-8') as f:
            card_list = json.load(f)
        
        # Load draft data (lightweight mode)
        draft_data = DraftData(card_list=card_list)
        
        # Load model
        model_path = f"{model_dir}/{set_code.lower()}_model.keras"
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for set {set_code}. Train with /train?set={set_code}"
            )
        
        print(f"Loading model for {set_code} from {model_path}...")
        model_builder = ModelBuilder(draft_data)
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEmbedding': PositionalEmbedding
        }
        
        # Try loading as .keras first, fallback to .h5 for legacy HDF5 models
        try:
            model_builder._model = load_model(model_path, custom_objects=custom_objects)
        except ValueError as e:
            if "zip file" in str(e).lower():
                # Model is in old HDF5 format, load it as .h5
                print(f"⚠ Legacy HDF5 model detected, using .h5 loader...")
                import shutil
                h5_path = model_path.replace('.keras', '.h5')
                shutil.copy(model_path, h5_path)
                try:
                    model_builder._model = load_model(h5_path, custom_objects=custom_objects)
                    os.remove(h5_path)
                except Exception as h5_error:
                    if os.path.exists(h5_path):
                        os.remove(h5_path)
                    raise h5_error
            else:
                raise
        
        # Cache
        self._model_cache[set_code] = model_builder
        self._draft_data_cache[set_code] = draft_data
        
        print(f"✓ Loaded {set_code} model with {len(draft_data.cards)} cards")
        return model_builder, draft_data
    
    def clear_cache(self, set_code: str = None):
        """
        Clear cached models.
        
        Args:
            set_code: Optional set code to clear. If None, clears all caches.
        """
        if set_code:
            set_code = set_code.upper()
            self._model_cache.pop(set_code, None)
            self._draft_data_cache.pop(set_code, None)
            print(f"✓ Cleared cache for {set_code}")
        else:
            self._model_cache.clear()
            self._draft_data_cache.clear()
            print("✓ Cleared all model caches")


# Global instance for dependency injection
_model_cache_instance = ModelCache()


def get_model_cache() -> ModelCache:
    """
    Dependency injection function for ModelCache.
    
    Returns:
        The global ModelCache instance
    """
    return _model_cache_instance