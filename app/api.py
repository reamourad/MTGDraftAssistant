from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from app.DraftData import DraftData
from app.ModelBuilder import ModelBuilder
from tensorflow.keras.models import load_model
import os
import glob
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
        return _model_cache[set_code], _draft_data_cache[set_code]

    # Find data path
    data_dir = f"data/{set_code}"
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail=f"No data found for set {set_code}")

    csv_files = glob.glob(f"{data_dir}/*.csv.gz") or glob.glob(f"{data_dir}/*.csv")
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No training data found for set {set_code}")

    data_path = csv_files[0]

    # Load draft data
    print(f"Loading draft data for {set_code} from {data_path}...")
    draft_data = DraftData(data_path)

    # Load model
    model_path = f"app/models/{set_code}/{set_code.lower()}_model.keras"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"No trained model found for set {set_code}. Train with /train?set={set_code}")

    print(f"Loading model for {set_code} from {model_path}...")
    model_builder = ModelBuilder(draft_data)
    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'PositionalEmbedding': PositionalEmbedding
    }
    model_builder._model = load_model(model_path, custom_objects=custom_objects)

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

        # Filter out cards with extremely low probabilities (< 0.01%)
        # This removes noise while keeping relevant options
        filtered_predictions = [
            p for p in predictions
            if p['probability'] > 0.0001
        ]

        return {
            "set": req.set.upper(),
            "predictions": filtered_predictions,
            "total_cards_in_pack": len(predictions),
            "cards_shown": len(filtered_predictions)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)