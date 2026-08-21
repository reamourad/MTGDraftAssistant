"""
LEGACY COMPATIBILITY LAYER - TensorFlow to PyTorch Migration

This file maintains backward compatibility with the existing TensorFlow-based
prediction system while preparing for migration to the PyTorch two-tower architecture.

Migration Path:
1. Current: /predict endpoint uses TensorFlow models (ml/current/)
2. Future: /predict_v2 endpoint will use PyTorch 2-tower model (ml/experimental/)
3. Once PyTorch model is trained and validated, /predict will be updated to use it

Architecture:
- TensorFlow System (CURRENT): app/ml/current/ - Production-ready, serving predictions
- PyTorch System (FUTURE): app/ml/experimental/ - Two-tower architecture, awaiting training

Note: Both systems coexist during transition. This file will be deprecated once
migration to app/api/main.py is complete.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.ml.current.draft_data import DraftData
from app.ml.current.model_builder import ModelBuilder, TransformerBlock, PositionalEmbedding
from tensorflow.keras.models import load_model
import os
import glob
import json
from fastapi.middleware.cors import CORSMiddleware
from app.core.booster import generate_booster
import uvicorn

app = FastAPI(
    title="Lotus Draft Assistant API",
    description="MTG Draft Assistant - TensorFlow (legacy) and PyTorch (future) prediction systems"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# TENSORFLOW SYSTEM (CURRENT/LEGACY)
# ============================================================================

# Cache for loaded TensorFlow models and draft data
_model_cache = {}
_draft_data_cache = {}


def load_set_model(set_code: str):
    """Load model and draft data for a specific set, with caching."""
    set_code = set_code.upper()

    # Return from cache if already loaded
    if set_code in _model_cache:
        print(f"✓ Using cached {set_code} model")
        return _model_cache[set_code], _draft_data_cache[set_code]

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
        raise HTTPException(status_code=404, detail=f"No trained model found for set {set_code}. Train with /train?set={set_code}")

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
    _model_cache[set_code] = model_builder
    _draft_data_cache[set_code] = draft_data

    print(f"✓ Loaded {set_code} model with {len(draft_data.cards)} cards")
    return model_builder, draft_data


class PredictRequest(BaseModel):
    set: str
    deck: list[str]  # Card names, not IDs
    pack: list[str]  # Card names from /booster

@app.get("/")
def root():
    return {"message": "Welcome to the Lotus Draft Assistant API"}


@app.get("/status")
def get_system_status():
    """
    Get information about available prediction systems.
    
    Returns information about TensorFlow (legacy) and PyTorch (experimental) systems.
    """
    # Check if PyTorch general model is available
    from app.ml.experimental.model_loader import PyTorchModelLoader
    
    pytorch_status = "not_ready"
    try:
        model_loader = PyTorchModelLoader(models_dir="app/models", use_gpu=False)
        if model_loader.is_model_available("general"):
            pytorch_status = "active"
    except Exception:
        pass
    
    return {
        "systems": {
            "tensorflow": {
                "status": "active",
                "endpoint": "/predict",
                "description": "Current production system using TensorFlow models",
                "location": "app/ml/current/"
            },
            "pytorch": {
                "status": pytorch_status,
                "endpoint": "/predict_pytorch",
                "description": "Two-tower architecture using PyTorch",
                "location": "app/ml/experimental/"
            }
        },
        "migration": {
            "current_phase": "Phase 2: PyTorch Model Available" if pytorch_status == "active" else "Phase 1: Dual System Operation",
            "next_phase": "Phase 3: Full Migration to PyTorch" if pytorch_status == "active" else "Phase 2: PyTorch Model Training",
            "timeline": "Ready for testing" if pytorch_status == "active" else "TBD - awaiting PyTorch model training completion"
        }
    }


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
    for entry in os.listdir(models_dir):
        set_path = os.path.join(models_dir, entry)

        # Skip non-directories and README
        if not os.path.isdir(set_path) or entry == "README.md":
            continue



        # Read config.json if available
        config_path = os.path.join(set_path, "config.json")
        set_info = {
            "code": entry.upper(),
            "name": entry.upper()
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    set_info["code"] = config.get("code", entry.upper())
                    set_info["name"] = config.get("name", entry.upper())
            except json.JSONDecodeError:
                pass  # Use defaults if config is invalid

        # Check if icon exists
        icon_path = os.path.join(set_path, "icon.png")
        set_info["has_icon"] = os.path.exists(icon_path)

                # Check if this directory has a model file
        model_files = glob.glob(os.path.join(set_path, "*.keras"))
        set_info["has_model"] = (model_files != [])
        
        sets.append(set_info)

    # Sort by set code
    sets.sort(key=lambda x: x["code"])

    return {
        "sets": sets,
        "count": len(sets)
    }


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
def predict_next_card(req: PredictRequest):
    """
    [TENSORFLOW/LEGACY] Predict the best card to pick from a pack.
    
    This endpoint uses the TensorFlow-based prediction system (ml/current/).
    For the future PyTorch two-tower model, use /predict_v2 (when available).

    Request body:
    - set: Set code (e.g., "MH3")
    - deck: List of card names already in your pool
    - pack: List of card names available in current pack (from /booster)

    Returns:
    - predictions: List of cards with probabilities, sorted by recommendation
    """
    try:
        # Load the appropriate TensorFlow model for this set
        model_builder, draft_data = load_set_model(req.set)

        # Convert card names to integers
        deck_ids = []
        for card_name in req.deck:
            if card_name in draft_data.cards_to_int:
                deck_ids.append(draft_data.cards_to_int[card_name])
            else:
                raise HTTPException(status_code=400, detail=f"Card '{card_name}' not found in {req.set} set")

        pack_ids = []
        for card_name in req.pack:
            if card_name in draft_data.cards_to_int:
                pack_ids.append(draft_data.cards_to_int[card_name])
            else:
                raise HTTPException(status_code=400, detail=f"Card '{card_name}' not found in {req.set} set")

        # Get predictions from TensorFlow model
        predictions = model_builder.predict(deck_ids, pack_ids)

        return {
            "set": req.set.upper(),
            "predictions": predictions,
            "model_type": "tensorflow"  # Indicate which model was used
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# PYTORCH SYSTEM (FUTURE/EXPERIMENTAL)
# ============================================================================

@app.post("/predict_v2")
def predict_next_card_v2(req: PredictRequest):
    """
    [PYTORCH/EXPERIMENTAL] Predict the best card to pick using two-tower architecture.
    
    This endpoint will use the PyTorch two-tower model (ml/experimental/) once trained.
    Currently returns a placeholder response indicating the model is not yet available.
    
    Request body:
    - set: Set code (e.g., "MH3")
    - deck: List of card names already in your pool
    - pack: List of card names available in current pack (from /booster)

    Returns:
    - predictions: List of cards with probabilities, sorted by recommendation
    
    Status: NOT YET IMPLEMENTED - PyTorch model training in progress
    """
    # TODO: Implement PyTorch two-tower prediction once model is trained
    # This will use:
    # - app/ml/experimental/two_tower_model.py for model architecture
    # - app/ml/experimental/model_loader.py for loading trained checkpoints
    # - app/ml/experimental/card_encoder.py for encoding cards (407-dim)
    # - app/core/prediction.py for orchestration
    
    raise HTTPException(
        status_code=501,
        detail=(
            "PyTorch two-tower model not yet available. "
            "Use /predict endpoint for TensorFlow-based predictions. "
            "The PyTorch model will be available after training is complete."
        )
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)