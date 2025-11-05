# MTG Draft Assistant

AI agent to predict optimal card picks during Magic: The Gathering (MTG) drafts based on real player data from 17Lands. The model learns from high-skilled players with 60%+ win rates.

Built with Keras, TensorFlow, and FastAPI. Training data sourced from [17Lands](https://www.17lands.com).

## Version History

- **v3.0**: Set-centric architecture, multi-set support
- **v2.0**: Upgraded to Transformer-based sequence model
- **v1.5**: Added MTG drafting rules, model acts as drafter
- **v1.0**: Optimized LSTM architecture
- **v0.1**: Initial deck building model

## Features

- **Per-Set Models**: Train separate models for each Magic set (MH3, BLB, EOE, etc.)
- **Transformer Architecture**: Advanced sequence model for draft pick predictions
- **Set-Centric Organization**: All set-specific files (models, configs, icons) in one place
- **FastAPI Backend**: RESTful API for draft assistance

## Architecture

```
MTGDraftAssistant/
├── data/                           # Training data (gitignored)
│   ├── MH3/
│   │   └── game_data_public.MH3.PremierDraft.csv.gz
│   ├── BLB/
│   └── EOE/
│
├── app/
│   ├── models/                     # Trained models (committed to git)
│   │   ├── MH3/
│   │   │   ├── mh3_model.keras
│   │   │   └── config.json
│   │   ├── BLB/
│   │   └── EOE/
│   ├── api.py
│   ├── DraftData.py
│   └── ModelBuilder.py
│
├── train.py                        # CLI training script
└── requirements.txt
```

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

## Training a Model

### Step 1: Download Training Data

1. Go to https://www.17lands.com/public_datasets
2. Download the Premier Draft data for your set (e.g., `game_data_public.MH3.PremierDraft.csv.gz`)
3. Place it in the corresponding set folder:
   ```
   data/MH3/game_data_public.MH3.PremierDraft.csv.gz
   ```
4. **Keep the `.csv.gz` compressed format**, pandas reads it directly, no need to unzip!

### Step 2: Train the Model

```bash
python train.py --set MH3 --epochs 10
```

The script will:
1. Find training data in `data/MH3/*.csv.gz`
2. Read the compressed file directly
3. Train a Transformer model
4. Save to `app/models/MH3/mh3_model.keras`

**Training Options:**
```bash
# Train with custom epochs
python train.py --set MH3 --epochs 15

# Train a different set
python train.py --set BLB --epochs 10

# Process only first N rows (for testing)
python train.py --set MH3 --limit 100000
```

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
Get AI draft pick recommendations.

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

## Adding a New Set

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
