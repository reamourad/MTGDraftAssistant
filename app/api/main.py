"""
FastAPI application and routing logic.

This module contains the main FastAPI app initialization, CORS setup,
and route handlers for the MTG Draft Assistant API.
"""

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.models import PredictRequest
from app.api.dependencies import ModelCache, get_model_cache
from app.booster.generator import generate_booster
import os
import glob
import json


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