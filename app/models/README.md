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
- `{set}_model.keras` - The trained Transformer model

**Optional:**
- `config.json` - Set metadata (name, code, release date, etc.)
- `icon.png` - Set icon/symbol for UI
- `cards.json` - Cached card list for inference
- `booster_rules.json` - Booster generation configuration

## Training a Model

Use the training CLI to create a model for a set:

```bash
python train.py --set MH3
```

This will:
1. Read training data from `data/MH3/*.csv.gz`
2. Train the model
3. Save to `app/models/MH3/mh3_model.keras`
