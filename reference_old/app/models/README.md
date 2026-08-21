# Trained Models

This directory contains trained models and set-specific assets.

## Directory Structure

Each set has its own folder with the trained model and optional metadata:

```
app/models/
├── MH3/
│   ├── mh3_model.keras          # Trained model (required)
│   ├── config.json              # Set metadata (optional)
│   └── icon.png                 # Set icon (optional)
├── BLB/
│   └── blb_model.keras
└── EOE/
    └── (ready for model)
```

## Files Per Set

**Required:**
- `{set}_model.keras` - The trained Transformer model (TensorFlow/Keras)
- `booster_config.json` - Booster generation structure and rules
- `sheets.json` - Filtered card sheets for booster generation
- `card_encodings.pkl` - Pre-computed 407-dim card encodings

**Optional:**
- `config.json` - Set metadata (name, code, release date, etc.)
- `icon.png` - Set icon/symbol for UI
- `training_cards.json` - List of card names used in training

**Note:** As of the codebase cleanup, `cards.json` is no longer generated or used. 
Card data is accessed through `card_encodings.pkl` for predictions and `sheets.json` 
for booster generation, eliminating redundant data storage.

## Training a Model

Use the training CLI to create a model for a set:

```bash
python train.py --set MH3
```

This will:
1. Read training data from `data/MH3/*.csv.gz`
2. Train the model
3. Save to `app/models/MH3/mh3_model.keras`
