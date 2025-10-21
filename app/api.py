from fastapi import FastAPI
from pydantic import BaseModel
from app.DraftData import DraftData
from app.ModelBuilder import ModelBuilder
from tensorflow.keras.models import load_model
import os
from app.ModelBuilder import TransformerBlock, PositionalEmbedding 
from fastapi.middleware.cors import CORSMiddleware # Needed for the browser to allow the request
import uvicorn
import requests

app = FastAPI(title="Lotus Draft Assistant API")

def ensure_file(path: str, url: str):
    """Download a file if it does not exist locally."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"🔽 Downloading {os.path.basename(path)} from Hugging Face …")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        print(f"✅ Saved {path}")

DATA_PATH  = "app/data/MH3_clean.csv"
MODEL_PATH = "app/model/best_model.keras"

DATA_URL  = "https://huggingface.co/datasets/SkyeMourad/MH3_clean"
ensure_file(DATA_PATH, DATA_URL)

origins = [
    # Allows requests from any origin (*). This is the simplest option for
    # local testing, but in production, you should restrict this to your
    # known domain(s).
    "*"
    
    # If your HTML page was hosted on a specific domain, you would list it here:
    # "http://localhost:3000",
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
def get_booster():
    return {"pack": draft_data.boosterCreater()}

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