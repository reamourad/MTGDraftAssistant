from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.DraftData import DraftData
from app.ModelBuilder import ModelBuilder
from tensorflow.keras.models import load_model
import os
import glob
import json
from app.ModelBuilder import TransformerBlock, PositionalEmbedding
from fastapi.middleware.cors import CORSMiddleware
from app.booster.generator import generate_booster
import uvicorn

app = FastAPI(title="Lotus Draft Assistant API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models and draft data
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

        # Check if this directory has a model file
        model_files = glob.glob(os.path.join(set_path, "*.keras"))
        if not model_files:
            continue

        # Read config.json if available
        config_path = os.path.join(set_path, "config.json")
        set_info = {
            "code": entry.upper(),
            "name": entry.upper(),
            "has_model": True
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
    Predict the best card to pick from a pack.

    Request body:
    - set: Set code (e.g., "MH3")
    - deck: List of card names already in your pool
    - pack: List of card names available in current pack (from /booster)

    Returns:
    - predictions: List of cards with probabilities, sorted by recommendation
    """
    try:
        # Load the appropriate model for this set
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

        # Get predictions
        predictions = model_builder.predict(deck_ids, pack_ids)

        return {
            "set": req.set.upper(),
            "predictions": predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)