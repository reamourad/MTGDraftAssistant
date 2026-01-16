"""
FastAPI application and routing logic.

This module contains the main FastAPI app initialization, CORS setup,
and route handlers for the MTG Draft Assistant API.
"""

import logging
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.models import PredictRequest
from app.api.dependencies import ModelCache, get_model_cache
from app.core.booster import generate_booster
from app.core.pytorch_prediction import PyTorchPredictionService, PyTorchPredictionError
from app.ml.experimental.model_loader import PyTorchModelLoader
from app.ml.experimental.card_encoder import CardEncoder
import os
import glob
import json


logger = logging.getLogger(__name__)


# Initialize FastAPI application
app = FastAPI(
    title="Lotus Draft Assistant API",
    description="API for Magic: The Gathering draft pick predictions and booster generation",
    version="1.0.0"
)

# Configure CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize PyTorch prediction service (lazy loading)
_pytorch_service = None
_pytorch_service_error = None


def get_pytorch_service() -> PyTorchPredictionService:
    """
    Get or initialize the PyTorch prediction service.
    
    Returns:
        PyTorchPredictionService instance
    
    Raises:
        HTTPException: If service initialization fails
    """
    global _pytorch_service, _pytorch_service_error
    
    if _pytorch_service is not None:
        return _pytorch_service
    
    if _pytorch_service_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"PyTorch service unavailable: {_pytorch_service_error}"
        )
    
    try:
        # Load card data from all preprocessed sets in app/models/
        card_data = []
        seen_names = set()
        models_dir = "app/models"
        
        if not os.path.exists(models_dir):
            raise Exception("Models directory not found")
        
        for set_dir in os.listdir(models_dir):
            set_path = os.path.join(models_dir, set_dir)
            if not os.path.isdir(set_path):
                continue
            
            # Skip the 'general' directory (that's where trained models go)
            if set_dir == 'general':
                continue
            
            card_file = os.path.join(set_path, "cards.json")
            if os.path.exists(card_file):
                try:
                    with open(card_file, 'r') as f:
                        cards = json.load(f)
                    
                    # Add unique cards only
                    for card in cards:
                        name = card.get('name')
                        if name and name not in seen_names:
                            card_data.append(card)
                            seen_names.add(name)
                    
                    logger.info(f"Loaded {len(cards)} cards from {set_dir}")
                except Exception as e:
                    logger.warning(f"Failed to load cards from {card_file}: {e}")
        
        if not card_data:
            raise Exception("No card data available. Please run preprocessing first: python preprocess_cards.py <SET>")
        
        logger.info(f"Loaded {len(card_data)} total unique cards from {len(seen_names)} sets")
        
        # Initialize components
        model_loader = PyTorchModelLoader(models_dir="app/models", use_gpu=True)
        card_encoder = CardEncoder(card_list=card_data, use_gpu=True)
        
        _pytorch_service = PyTorchPredictionService(model_loader, card_encoder)
        logger.info("PyTorch service initialized successfully")
        return _pytorch_service
        
    except Exception as e:
        _pytorch_service_error = str(e)
        logger.error(f"Failed to initialize PyTorch service: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize PyTorch service: {str(e)}"
        )


#Checked
@app.get("/")
def root():
    """Root endpoint returning welcome message."""
    return {"message": "Welcome to the Lotus Draft Assistant API"}

#Checked
@app.get("/sets")
def get_supported_sets():
    """
    Get all sets that have trained models available.

    Returns a list of sets with their metadata (code, name) and whether they have a model.
    """
    sets = []
    models_dir = "app/models"

    if not os.path.exists(models_dir):
        raise HTTPException(status_code=500, detail="Models directory not found")

    # Scan all subdirectories in models folder
    for folder in os.listdir(models_dir):

        folder_path = os.path.join(models_dir, folder)

        # Skip non-directories and README
        if not os.path.isdir(folder_path) or folder == "README.md":
            continue

        # Read config.json if available
        config_path = os.path.join(folder_path, "config.json")
        set_info = {
            "code": folder.upper(),
            "name": folder.upper()
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    set_info["code"] = config.get("code", folder.upper())
                    set_info["name"] = config.get("name", folder.upper())
            except json.JSONDecodeError:
                pass  # Use defaults if config is invalid

        # Check if icon exists
        icon_path = os.path.join(folder_path, "icon.png")
        set_info["has_icon"] = os.path.exists(icon_path)

        # Check if this directory has a model file
        model_files = glob.glob(os.path.join(folder_path, "*.keras"))
        set_info["has_model"] = (model_files != [])
        
        sets.append(set_info)

    # Sort by set code
    sets.sort(key=lambda x: x["code"])

    return {"sets": sets, "count": len(sets)}

#Checked
@app.get("/sets/{set_code}/icon")
def get_set_icon(set_code: str):
    """
    Get the icon image for a specific set.

    Returns the icon.png file for the requested set.
    """
    set_code = set_code.upper()
    icon_path = os.path.join("app/models", set_code, "icon.png")

    if not os.path.exists(icon_path):
        raise HTTPException(
            status_code=404,
            detail=f"Icon not found for set {set_code}"
        )

    return FileResponse(
        icon_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"}  # Cache for 1 day
    )


#Checked
@app.get("/booster")
def get_booster(set: str = Query("MH3", description="Set code (e.g., 'MH3', 'BLB')")):
    """
    Generate a draft booster pack using MTGJson rules.

    Returns pack as card names. Use /predict to get AI recommendations.
    """
    try:
        card_names = generate_booster(set)
        return {
            "pack": card_names,
            "set": set.upper(),
            "count": len(card_names)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate booster: {str(e)}")


#Todo: to change when I create the new model 
@app.post("/predict")
def predict_next_card(
    req: PredictRequest,
    model_cache: ModelCache = Depends(get_model_cache)
):
    """
    Predict the best card to pick from a pack.

    Request body:
    - set: Set code (e.g., "MH3")
    - deck: List of card names already in your pool
    - pack: List of card names available in current pack (from /booster)

    Returns:
    - predictions: List of cards with probabilities, sorted by recommendation
    """
    try:
        # Load the appropriate model for this set using dependency injection
        model_builder, draft_data = model_cache.get_model(req.set)

        # Convert card names to integers
        deck_ids = []
        for card_name in req.deck:
            if card_name in draft_data.cards_to_int:
                deck_ids.append(draft_data.cards_to_int[card_name])
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Card '{card_name}' not found in {req.set} set"
                )

        pack_ids = []
        for card_name in req.pack:
            if card_name in draft_data.cards_to_int:
                pack_ids.append(draft_data.cards_to_int[card_name])
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Card '{card_name}' not found in {req.set} set"
                )

        # Get predictions from model
        predictions = model_builder.predict(deck_ids, pack_ids)

        return {
            "set": req.set.upper(),
            "predictions": predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_pytorch")
def predict_next_card_pytorch(req: PredictRequest):
    """
    Predict the best card to pick from a pack using PyTorch two-tower model.

    Request body:
    - set: Set code (e.g., "MH3") - ignored for general model
    - deck: List of card names already in your pool
    - pack: List of card names available in current pack (from /booster)

    Returns:
    - predictions: List of cards with probabilities, sorted by recommendation
    - model_type: "pytorch" to indicate which model was used
    """
    try:
        # Get PyTorch prediction service
        pytorch_service = get_pytorch_service()
        
        # Get predictions
        predictions = pytorch_service.predict_picks(
            set_code=req.set,
            deck=req.deck,
            pack=req.pack
        )
        
        # Format predictions for API response
        formatted_predictions = [
            {
                "card_name": pred.card_name,
                "probability": pred.probability
            }
            for pred in predictions
        ]
        
        return {
            "set": req.set.upper(),
            "predictions": formatted_predictions,
            "model_type": "pytorch"
        }
    
    except PyTorchPredictionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")