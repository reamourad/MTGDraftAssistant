import modal

# ============================================================================
# Which trained model to serve. Change this path to swap models — everything
# else in this file adapts automatically. Must point to a checkpoint already
# uploaded to the "mtg-draft-data" volume (see upload instructions in chat).
# ============================================================================
MODEL_CHECKPOINT_PATH = "/root/data/models/fold_0.pt"

SET_CODES = ["TLA", "TDM", "MH3", "FIN", "EOE", "NEO", "MSH", "Powered_Cube"]

def _download_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fastapi[standard]")
    # bake the model weights into the image so a cold-starting container
    # never has to fetch them from Hugging Face over the network
    .run_function(_download_sentence_transformer)
    .add_local_python_source("draft_model")
)

app = modal.App("mtg-draft-serving", image=image)
volume = modal.Volume.from_name("mtg-draft-data", create_if_missing=True)


# min_containers=1 keeps one container always warm so requests never hit a
# cold start, at the cost of continuous billing even while idle.
# scaledown_window covers any extra containers Modal spins up under load.
@app.function(image=image, volumes={"/root/data": volume}, min_containers=1, scaledown_window=600)
@modal.asgi_app()
def fastapi_app():
    import os
    os.chdir("/root")

    from typing import List

    import threading
    import time

    import requests
    import torch
    from fastapi import FastAPI, HTTPException, Query, Response
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

    SCRYFALL_HEADERS = {"User-Agent": "LotusDraftAssistant/1.0", "Accept": "*/*"}
    SCRYFALL_CACHE_TTL_SECONDS = 60 * 60 * 24
    SCRYFALL_MIN_REQUEST_INTERVAL = 0.1  # Scryfall recommends 50-100ms between requests
    scryfall_json_cache = {}
    scryfall_rate_lock = threading.Lock()
    scryfall_last_request_time = [0.0]

    def scryfall_get(params: dict):
        """GET api.scryfall.com/cards/named, throttled and with 429 retry."""
        for attempt in range(3):
            with scryfall_rate_lock:
                wait = SCRYFALL_MIN_REQUEST_INTERVAL - (time.time() - scryfall_last_request_time[0])
                if wait > 0:
                    time.sleep(wait)
                scryfall_last_request_time[0] = time.time()

            resp = requests.get(
                "https://api.scryfall.com/cards/named",
                params=params,
                headers=SCRYFALL_HEADERS,
            )
            if resp.status_code != 429:
                return resp
            retry_after = float(resp.headers.get("Retry-After", 1))
            time.sleep(retry_after)
        return resp

    def normalize_set_code(set_code: str) -> str:
        # Match case-insensitively against SET_CODES and return the
        # canonical casing. Real MTGJSON codes are all-uppercase so this is
        # a no-op for them, but "Powered_Cube" isn't a real set code and its
        # on-disk directory (and Modal's Linux filesystem is case-sensitive,
        # unlike the dev machine) only matches that exact mixed case.
        for code in SET_CODES:
            if code.upper() == set_code.upper():
                return code
        raise HTTPException(status_code=404, detail=f"Unknown set: {set_code}")

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
        set_code = normalize_set_code(set)
        try:
            pack = pack_generator.generate(set_code)
            _, name_to_features = get_set_context(set_code)

            cards = []
            for card in pack:
                features = name_to_features.get(card["name"], {})
                cards.append({
                    "name": card["name"],
                    "mana_cost": features.get("mana_cost", ""),
                    "cmc": features.get("converted_mana_cost", 0),
                    "types": features.get("types", []),
                    "subtypes": features.get("subtypes", []),
                    "rarity": features.get("rarity", "common"),
                    "power": features.get("power"),
                    "toughness": features.get("toughness"),
                    "oracle_text": features.get("oracle_text", ""),
                })

            return {"pack": cards, "set": set_code, "count": len(cards)}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate booster: {e}")

    @web_app.get("/scryfall")
    def get_scryfall_card(
        cardName: str = Query(..., description="Exact card name"),
        set: str = Query(None, description="Set code (unused by Scryfall lookup, kept for cache-key parity)"),
    ):
        cache_key = f"{cardName}-{set}"
        cached = scryfall_json_cache.get(cache_key)
        if cached and time.time() - cached[0] < SCRYFALL_CACHE_TTL_SECONDS:
            return cached[1]

        resp = scryfall_get({"exact": cardName})
        if not resp.ok:
            raise HTTPException(status_code=resp.status_code, detail=f"Scryfall API error: {resp.status_code}")

        data = resp.json()
        scryfall_json_cache[cache_key] = (time.time(), data)
        return data

    @web_app.get("/card-image")
    def get_card_image(
        cardName: str = Query(..., description="Exact card name"),
        version: str = Query("png", description="Scryfall image version"),
    ):
        resp = scryfall_get({"exact": cardName, "format": "image", "version": version})
        if not resp.ok:
            raise HTTPException(status_code=resp.status_code, detail=f"Scryfall API error: {resp.status_code}")

        content_type = resp.headers.get("content-type", "image/jpeg")
        return Response(
            content=resp.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @web_app.post("/predict")
    def predict(req: PredictRequest):
        set_code = normalize_set_code(req.set)

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
            # softmax over the pack so probabilities sum to 100% (relative
            # preference among these candidates, not independent per-card odds)
            probabilities = torch.softmax(scores, dim=0).tolist()

        predictions = [
            {"card_name": name, "probability": prob}
            for name, prob in zip(req.pack, probabilities)
        ]
        predictions.sort(key=lambda p: -p["probability"])

        return {"set": set_code, "predictions": predictions}

    return web_app
