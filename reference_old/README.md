# MTG Draft Assistant

AI agent to predict optimal card picks during Magic: The Gathering (MTG) drafts based on real player data from 17Lands. The model learns from high-skilled players with 60%+ win rates.

Built with Keras, TensorFlow, and FastAPI. Training data sourced from [17Lands](https://www.17lands.com).

## Version History

- **v4.0**: PyTorch two-tower model with general set-agnostic architecture
- **v3.0**: Set-centric architecture, multi-set support
- **v2.0**: Upgraded to Transformer-based sequence model
- **v1.5**: Added MTG drafting rules, model acts as drafter
- **v1.0**: Optimized LSTM architecture
- **v0.1**: Initial deck building model

## Features

- **PyTorch Two-Tower Model**: General-purpose model that works across all MTG sets
- **Set-Agnostic Architecture**: Unified 407-dimensional card encoding for any set
- **TensorFlow Legacy Models**: Per-set Transformer models (v3.0)
- **Dual Model Support**: Choose between PyTorch general model or TensorFlow set-specific models
- **FastAPI Backend**: RESTful API for draft assistance with both model types

## Architecture

```
MTGDraftAssistant/
├── data/                           # Training data (gitignored)
│   ├── MH3/
│   │   ├── game_data_public.MH3.PremierDraft.csv.gz
│   │   └── MH3_cards.json
│   ├── BLB/
│   └── FIN/
│
├── app/
│   ├── models/                     # Trained models
│   │   ├── general/                # PyTorch general model
│   │   │   ├── best_model.pt
│   │   │   └── training_history.json
│   │   ├── MH3/                    # TensorFlow set-specific models
│   │   │   ├── mh3_model.keras
│   │   │   └── config.json
│   │   └── BLB/
│   │
│   ├── ml/
│   │   ├── experimental/           # PyTorch two-tower architecture
│   │   │   ├── two_tower_model.py
│   │   │   ├── candidate_tower.py
│   │   │   ├── context_tower.py
│   │   │   ├── scoring_head.py
│   │   │   └── card_encoder.py
│   │   └── current/                # TensorFlow models (legacy)
│   │
│   ├── training/                   # PyTorch training infrastructure
│   │   ├── dataset.py
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   │
│   ├── core/
│   │   └── pytorch_prediction.py   # PyTorch prediction service
│   │
│   └── api/
│       └── main.py                 # FastAPI endpoints
│
├── scripts/
│   └── train_pytorch.py            # PyTorch training CLI
│
├── docs/
│   └── TRAINING.md                 # PyTorch training guide
│
└── requirements.txt
```

### Model Architecture Comparison

| Feature | PyTorch Two-Tower (v4.0) | TensorFlow Transformer (v3.0) |
|---------|--------------------------|-------------------------------|
| **Architecture** | Two-tower with candidate + context encoding | Transformer sequence model |
| **Set Support** | General (all sets) | Per-set models |
| **Card Encoding** | 407-dim unified features | Set-specific embeddings |
| **Training Data** | Multi-set combined | Single set |
| **Inference Speed** | Fast (~10ms) | Moderate (~20ms) |
| **Accuracy** | 35-42% top-1 | 40-45% top-1 |
| **Use Case** | Production, new sets | Set-specific optimization |

## Requirements

- **Python 3.10** (required for TensorFlow compatibility)

## Setup

### 1. Create Virtual Environment

**Windows:**
```bash
python3.10 -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Training Models

### PyTorch Two-Tower Model (Recommended)

Train a general-purpose model that works across all MTG sets:

```bash
# Train on multiple sets (general model)
python scripts/train_pytorch.py --sets MH3 BLB FIN --epochs 30

# Train on single set (faster)
python scripts/train_pytorch.py --sets MH3 --epochs 20

# CPU-only training
python scripts/train_pytorch.py --sets MH3 --no-gpu --batch-size 16
```

**See [docs/TRAINING.md](docs/TRAINING.md) for comprehensive training guide including:**
- Hardware requirements and GPU recommendations
- Data preparation steps
- All training arguments and hyperparameters
- Example commands for different scenarios
- Troubleshooting common issues

### TensorFlow Set-Specific Models (Legacy)

Train a Transformer model for a specific set:

```bash
python train_model.py --set MH3 --epochs 10
```

**Note:** TensorFlow models are set-specific and require separate training for each set. The PyTorch two-tower model is recommended for new deployments.

## Running the API

Start the FastAPI server:

```bash
uvicorn app.api:app --reload
```

The API will be available at `http://localhost:8000`

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### `GET /`
Get API information.

**Response:**
```json
{
  "message": "Welcome to the Lotus Draft Assistant API"
}
```

### `GET /status`
Get system status including available models.

**Response:**
```json
{
  "status": "healthy",
  "tensorflow": {
    "status": "active",
    "sets": ["MH3", "BLB", "EOE"]
  },
  "pytorch": {
    "status": "active",
    "model_type": "general"
  }
}
```

### `GET /sets`
Get all available sets with trained models.

**Response:**
```json
{
  "sets": [
    {
      "code": "EOE",
      "name": "Edges of Eternities",
      "has_model": true,
      "has_icon": true
    },
    {
      "code": "MH3",
      "name": "Modern Horizons 3",
      "has_model": true,
      "has_icon": true
    }
  ],
  "count": 2
}
```

### `GET /sets/{set_code}/icon`
Get the icon image for a specific set (e.g., `/sets/MH3/icon`).

Returns a PNG image file with cache headers for optimal performance.

### `GET /booster?set=MH3`
Generate a draft booster pack for a specific set using MTGJson rules.

**Response:**
```json
{
  "pack": ["Lightning Bolt", "Counterspell", "Giant Growth", ...],
  "set": "MH3",
  "count": 14
}
```

### `POST /predict`
Get AI draft pick recommendations using TensorFlow set-specific model.

**Request:**
```json
{
  "set": "MH3",
  "deck": ["Lightning Bolt", "Counterspell"],
  "pack": ["Giant Growth", "Shock", "Cancel", "Grizzly Bears"]
}
```

**Response:**
```json
{
  "set": "MH3",
  "model_type": "tensorflow",
  "predictions": [
    {
      "card": "Giant Growth",
      "probability": 0.85
    },
    {
      "card": "Shock",
      "probability": 0.12
    },
    {
      "card": "Cancel",
      "probability": 0.02
    },
    {
      "card": "Grizzly Bears",
      "probability": 0.01
    }
  ]
}
```

### `POST /predict_pytorch`
Get AI draft pick recommendations using PyTorch general model.

**Request:**
```json
{
  "set": "MH3",
  "deck": ["Lightning Bolt", "Counterspell"],
  "pack": ["Giant Growth", "Shock", "Cancel", "Grizzly Bears"]
}
```

**Response:**
```json
{
  "set": "MH3",
  "model_type": "pytorch",
  "predictions": [
    {
      "card": "Giant Growth",
      "probability": 0.82
    },
    {
      "card": "Shock",
      "probability": 0.15
    },
    {
      "card": "Cancel",
      "probability": 0.02
    },
    {
      "card": "Grizzly Bears",
      "probability": 0.01
    }
  ]
}
```

**Note:** The PyTorch endpoint works for any set, even those without TensorFlow models.

## Adding a New Set

### For PyTorch General Model

The PyTorch model works with any set automatically - no additional training needed! Just ensure card data is available:

1. **Add card data:**
   ```bash
   mkdir -p data/NEW_SET
   # Add NEW_SET_cards.json with card information
   ```

2. **Use immediately:**
   ```bash
   curl -X POST http://localhost:8000/predict_pytorch \
     -H "Content-Type: application/json" \
     -d '{"set": "NEW_SET", "deck": [...], "pack": [...]}'
   ```

### For TensorFlow Set-Specific Model

1. **Create set directory:**
   ```bash
   mkdir -p data/NEW_SET app/models/NEW_SET
   ```

2. **Add config:**
   ```json
   // app/models/NEW_SET/config.json
   {
     "code": "NEW_SET",
     "name": "New Set Name"
   }
   ```

3. **Download training data** to `data/NEW_SET/`

4. **Train model:**
   ```bash
   python train.py --set NEW_SET
   ```



## Data Sources

Training data comes from [17Lands](https://www.17lands.com/public_datasets), which collects draft logs from Magic Arena players. Data is excluded from git due to file size (stored in `data/` directory).

## Documentation

- **[TRAINING.md](docs/TRAINING.md)** - Comprehensive PyTorch training guide
  - Hardware requirements and GPU recommendations
  - Data preparation and organization
  - Training commands and hyperparameters
  - Monitoring and evaluation
  - Troubleshooting common issues

- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when server is running)
