from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.DraftData import DraftData
from app.ModelBuilder import ModelBuilder
from tensorflow.keras.models import load_model
import os
from app.ModelBuilder import TransformerBlock, PositionalEmbedding
from fastapi.middleware.cors import CORSMiddleware
from app.booster.generator import generate_booster
import uvicorn

app = FastAPI(title="Lotus Draft Assistant API")


DATA_PATH  = "data/MH3/draft_data_public.MH3.PremierDraft.csv.gz"
MODEL_PATH = "app/models/MH3/mh3_model.keras"

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # The domains/origins allowed to make requests
    allow_credentials=True,       # Allow cookies (if needed)
    allow_methods=["*"],          # Allow all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],          # Allow all headers
)

if os.path.exists(DATA_PATH):
    draft_data = DraftData(DATA_PATH)
    model_builder = ModelBuilder(draft_data)

    # Try to load an existing model
    if os.path.exists(MODEL_PATH):
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEmbedding': PositionalEmbedding
        }
        model_builder._model = load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Loaded existing trained model.")
    else:
        print("No trained model found, use /train endpoint to train one.")
else:
    print("No draft data found, everything is broken o-o")

class PredictRequest(BaseModel):
    deck: list[int]
    pack: list[int]

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
        # Generate booster using cached MTGJson data
        card_names = generate_booster(set)

        return {
            "pack": card_names,
            "set": set.upper(),
            "count": len(card_names)
        }
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": f"Failed to generate booster: {str(e)}"}, 500

@app.post("/train")
def train_model(epochs: int = 3):
    model_builder.train_model(epochs)
    model_builder._model.save(MODEL_PATH)
    return {"message": f"Training complete ({epochs} epochs)"}

@app.post("/predict")
def predict_next_card(req: PredictRequest):
    predictions = model_builder.predict(req.deck, req.pack)
    return {"prediction": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)