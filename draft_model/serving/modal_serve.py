import modal

# ============================================================================
# Which trained model to serve. Change this path to swap models — everything
# else in this file adapts automatically. Must point to a checkpoint already
# uploaded to the "mtg-draft-data" volume (see upload instructions in chat).
# ============================================================================
MODEL_CHECKPOINT_PATH = "/root/data/models/fold_0.pt"

SET_CODES = ["TLA", "TDM", "MH3", "FIN", "EOE", "NEO", "MSH", "Powered_Cube"]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fastapi[standard]")
    .add_local_python_source("draft_model")
)

app = modal.App("mtg-draft-serving", image=image)
volume = modal.Volume.from_name("mtg-draft-data", create_if_missing=True)


@app.function(image=image, volumes={"/root/data": volume})
@modal.asgi_app()
def fastapi_app():
    import os
    os.chdir("/root")

    from typing import List

    import torch
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    from draft_model.model.oracle_text_projection import OracleTextProjection
    from draft_model.model.pick_scorer import PickScorer
    from draft_model.model.sequence_builder import SequenceBuilder
    from draft_model.training.training_data_builder import TrainingDataBuilder
    from draft_model.pack_generator.pack_generator import PackGenerator

    web_app = FastAPI(title="Lotus Draft Assistant API")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        set: str
        deck: List[str]
        pack: List[str]

    print(f"[serving] loading checkpoint: {MODEL_CHECKPOINT_PATH}")
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    projection = OracleTextProjection()
    pick_scorer = PickScorer()
    projection.load_state_dict(checkpoint["projection"])
    pick_scorer.load_state_dict(checkpoint["pick_scorer"])
    pick_scorer.eval()

    sequence_builder = SequenceBuilder(projection)
    training_data_builder = TrainingDataBuilder(sequence_builder)
    pack_generator = PackGenerator()

    # cached per set_code once a container is warm — same set/card features
    # every request, no reason to rebuild them each time
    set_context_cache = {}

    def get_set_context(set_code: str):
        if set_code not in set_context_cache:
            name_to_features = training_data_builder.get_name_to_features(set_code)
            set_cards = training_data_builder.encode_set_cards(set_code, name_to_features)
            set_context_cache[set_code] = (set_cards, name_to_features)
        return set_context_cache[set_code]

    @web_app.get("/")
    def root():
        return {"message": "Welcome to the Lotus Draft Assistant API"}

    @web_app.get("/sets")
    def get_sets():
        sets = [{"code": code, "name": code, "has_icon": False, "has_model": True} for code in SET_CODES]
        sets.sort(key=lambda s: s["code"])
        return {"sets": sets, "count": len(sets)}

    @web_app.get("/booster")
    def get_booster(set: str = Query("MH3", description="Set code (e.g., 'MH3')")):
        try:
            pack = pack_generator.generate(set.upper())
            card_names = [card["name"] for card in pack]
            return {"pack": card_names, "set": set.upper(), "count": len(card_names)}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate booster: {e}")

    @web_app.post("/predict")
    def predict(req: PredictRequest):
        set_code = req.set.upper()

        try:
            set_cards, name_to_features = get_set_context(set_code)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No data available for set {set_code}: {e}")

        unknown = [name for name in req.pack + req.deck if name not in name_to_features]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown card(s) for set {set_code}: {unknown}")

        pool_vectors = [training_data_builder.card_encoder.encode(name_to_features[name]) for name in req.deck]

        sequences = []
        masks = []
        for i, candidate_name in enumerate(req.pack):
            candidate_vector = training_data_builder.card_encoder.encode(name_to_features[candidate_name])
            # exclude by POSITION not name, so duplicate copies of the same
            # card in a real pack don't accidentally exclude each other
            other_pack_vectors = [
                training_data_builder.card_encoder.encode(name_to_features[name])
                for j, name in enumerate(req.pack) if j != i
            ]
            seq, mask = sequence_builder.build_full_sequence(set_cards, pool_vectors, candidate_vector, other_pack_vectors)
            sequences.append(seq)
            masks.append(mask)

        sequences = torch.stack(sequences)
        masks = torch.stack(masks)

        with torch.no_grad():
            scores = pick_scorer(sequences, masks)
            probabilities = torch.sigmoid(scores).tolist()

        predictions = [
            {"card_name": name, "probability": prob}
            for name, prob in zip(req.pack, probabilities)
        ]
        predictions.sort(key=lambda p: -p["probability"])

        return {"set": set_code, "predictions": predictions}

    return web_app
