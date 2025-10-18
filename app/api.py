from fastapi import FastAPI
from pydantic import BaseModel
from app.DraftData import DraftData
from app.ModelBuilder import ModelBuilder
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Lotus Draft Assistant API")

DATA_PATH = "app/data/first_1000.csv"      # Update this later
MODEL_PATH = "app/model/trained_model.keras"

draft_data = DraftData(DATA_PATH)
model_builder = ModelBuilder(draft_data)

# Try to load an existing model
if os.path.exists(MODEL_PATH):
    model_builder._model = load_model(MODEL_PATH)
    print("Loaded existing trained model.")
else:
    print("No trained model found, use /train endpoint to train one.")

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
