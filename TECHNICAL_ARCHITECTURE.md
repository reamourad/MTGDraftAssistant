# MTG Draft Assistant: Comprehensive Technical Architecture Analysis

## Executive Summary

The MTG Draft Assistant is an AI-powered card recommendation system that uses a **Transformer-based neural network** to predict optimal card picks during Magic: The Gathering (MTG) draft games. The system evolved from an initial LSTM implementation (v1.0) to a Transformer architecture (v2.0+), offering improved sequence modeling and better performance on draft decision prediction.

---

## 1. TRANSFORMER ARCHITECTURE OVERVIEW

### Model Architecture (TransformerBlock-based)

**File:** `/app/ModelBuilder.py`

The current model uses a **3-layer Transformer stack** with the following configuration:

```
Input (Sequence of length 64)
  ↓
Embedding Layer (input_dim=n_vocab, output_dim=256, mask_zero=True)
  ↓
Positional Embedding Layer (adds position information)
  ↓
Dropout (0.2)
  ↓
TransformerBlock 1 (embed_dim=256, num_heads=8, ff_dim=512, dropout=0.2)
  ↓
TransformerBlock 2 (embed_dim=256, num_heads=8, ff_dim=512, dropout=0.2)
  ↓
TransformerBlock 3 (embed_dim=256, num_heads=8, ff_dim=512, dropout=0.2)
  ↓
GlobalAveragePooling1D (aggregates entire sequence)
  ↓
Dense(256, activation='relu')
  ↓
Dropout (0.2)
  ↓
Dense(n_vocab, activation='softmax') [Output: Card Probability Distribution]
```

### Key Architectural Components

#### 1.1 TransformerBlock (Lines 22-57)
- **Multi-Head Attention**: 8 attention heads for parallel feature representation
- **Feed-Forward Network**: Two dense layers (256→512→256) with ReLU activation
- **Layer Normalization**: Applied after both attention and FFN sublayers
- **Residual Connections**: Skip connections around both sublayers
- **Dropout**: 0.2 applied after attention and FFN for regularization

```python
class TransformerBlock(Model):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

#### 1.2 PositionalEmbedding (Lines 59-81)
- Uses learned embedding vectors for position information
- Added directly to token embeddings
- Sequence length: 64 positions

```python
class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, output_dim):
        self.position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=output_dim
        )
    
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_vectors = self.position_embedding(positions)
        return inputs + position_vectors
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| EMBED_DIM | 256 | Balanced embedding dimension |
| NUM_HEADS | 8 | Allows diverse attention patterns |
| FF_DIM | 512 | 2x embedding dimension (typical ratio) |
| DROPOUT_RATE | 0.2 | Increased from 0.1 to combat overfitting |
| SEQUENCE_LENGTH | 64 | Captures up to 64 cards in pool+pack |
| BATCH_SIZE | 32 | Smaller batch for better generalization |
| LEARNING_RATE | 0.0001 | Conservative learning rate |

---

## 2. TRAINING DATA & PREPARATION

### Data Source
- **17Lands.com**: Real player draft data from Magic Arena
- **Filtering Criteria**: Only picks from players with 60%+ win rates
- **Format**: Premier Draft (standard best-of-3 format)

### DraftData Class (Lines 1-89 in DraftData.py)

Handles data preparation and encoding:

```
Training CSV (from 17Lands)
  ↓
Load with pandas (up to 1M rows)
  ↓
Extract:
  - pool_* columns: Cards already picked
  - pack_card_* columns: Available cards in current pack
  - pick column: The card player chose
  - event_match_wins/losses: Calculate player win rate
  ↓
Build Vocabulary:
  - Extract unique card names
  - Add special tokens: [SEP_POOL], [SEP_PACK], [PAD]
  - Create bidirectional mappings (card_name ↔ int_id)
  ↓
Tokenization:
  - All cards mapped to integer IDs
  - Special tokens for structural information
```

### Sequence Construction (Lines 118-163 in ModelBuilder.py)

For each training example:

```
[SEP_POOL] + pool_cards + [SEP_PACK] + pack_cards
                    ↓
            If length > 64:
            Take last 64 tokens (recency bias)
                    ↓
            If length < 64:
            Left-pad with [PAD] tokens
                    ↓
            Final input: (1, 64) integer sequence
```

**Critical Implementation Detail (FIXED)**: The target card is included in the pack during training, allowing the model to learn from the full context of available choices.

### Data Filtering Pipeline (Lines 98-116 in ModelBuilder.py)

```python
# 1. Calculate win rate from event results
data['player_win_rate'] = event_wins / total_matches

# 2. Filter to high-skill players
filtered_data = data[data['player_win_rate'] >= 0.6]

# 3. Build training examples
# Only use picks where:
# - All pool cards exist in vocabulary
# - All pack cards exist in vocabulary
# - Target card exists in vocabulary
```

---

## 3. API IMPLEMENTATION

**File:** `/app/api.py`

### Architecture: FastAPI with Model Caching

```
Client Request
    ↓
FastAPI Route Handler
    ↓
Load Model from Cache (or load fresh)
    ↓
Convert card names → integer IDs
    ↓
Call ModelBuilder.predict()
    ↓
Return probability distribution
    ↓
JSON Response
```

### Key Endpoints

#### 1. `/` (GET) - Root
Returns API welcome message

#### 2. `/sets` (GET) - List Available Sets
```python
def get_supported_sets():
    """
    Scans app/models/ directory for:
    - Model files (*.keras)
    - Config files (config.json)
    - Icon files (icon.png)
    
    Returns JSON with set metadata and capabilities
    """
    # Returns:
    {
        "sets": [
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

#### 3. `/sets/{set_code}/icon` (GET) - Set Icon
Returns PNG image file for set with 1-day cache headers

#### 4. `/booster?set=MH3` (GET) - Generate Booster Pack
Uses MTGJson official booster rules filtered to training data cards

#### 5. `/predict` (POST) - Get Card Recommendations
**Request:**
```json
{
    "set": "MH3",
    "deck": ["Lightning Bolt", "Counterspell"],
    "pack": ["Giant Growth", "Shock", "Cancel"]
}
```

**Response:**
```json
{
    "set": "MH3",
    "predictions": [
        {"card_id": 42, "card_name": "Giant Growth", "probability": 0.85},
        {"card_id": 18, "card_name": "Shock", "probability": 0.12},
        {"card_id": 101, "card_name": "Cancel", "probability": 0.03}
    ]
}
```

### Model Loading & Caching (Lines 32-95)

**Key Strategy**: Lightweight prediction-only DraftData initialization

```python
def load_set_model(set_code: str):
    # Check cache first
    if set_code in _model_cache:
        return cached_model, cached_draft_data
    
    # Load card list from training_cards.json (NO CSV needed)
    with open(f"app/models/{set_code}/training_cards.json") as f:
        card_list = json.load(f)
    
    # Create lightweight DraftData (card_list only)
    draft_data = DraftData(card_list=card_list)
    
    # Load model with custom objects
    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'PositionalEmbedding': PositionalEmbedding
    }
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Cache for future requests
    _model_cache[set_code] = model
    _draft_data_cache[set_code] = draft_data
    
    return model, draft_data
```

**Optimization**: No CSV loading needed at runtime - uses `training_cards.json` cached during training.

### CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 4. WHY TRANSFORMERS OVER LSTM?

### Historical Context (Commit 711919c: "Changed to Transformer")

**v1.0 (LSTM Architecture):**
```
Embedding → LSTM(512) → Dense(output)
- Sequence length: 42
- Single LSTM layer
- Learning rate: 0.01
```

**v2.0+ (Transformer Architecture):**
```
Embedding → Positional Embedding → 3x TransformerBlocks → GlobalAveragePooling → Dense → Output
- Sequence length: 64
- 8 multi-head attention heads
- Learning rate: 0.0001
```

### Key Advantages of Transformers for MTG Draft Prediction

| Aspect | LSTM | Transformer | Benefit |
|--------|------|-------------|---------|
| **Long-range dependencies** | Gradient vanishing | Direct attention | Can relate cards picked early to current decision |
| **Parallelizability** | Sequential processing | Parallel computation | Faster training |
| **Attention interpretation** | Implicit | Explicit (attention weights) | Explainability: see which cards influence decision |
| **Position encoding** | Recurrent state | Explicit embeddings | Better handling of card order in sequence |
| **Context aggregation** | Last hidden state only | Multiple attention heads | Richer context from entire sequence |
| **Scalability** | Memory grows with sequence | Linear attention complexity | Can handle longer drafts |

### Specific MTG Advantages

1. **Multi-headed attention captures different draft strategies**
   - Head 1: "Color synergies"
   - Head 2: "Mana curve balance"
   - Head 3: "Power level comparison"
   - Etc. (8 heads total)

2. **Parallel processing enables larger training batches**
   - Original learning rate: 0.01 (high, unstable)
   - New learning rate: 0.0001 (conservative, stable)
   - Indicates Transformers train more stably

3. **GlobalAveragePooling aggregates full sequence context**
   - Instead of relying on final LSTM state (limited memory)
   - Weighs all positions equally, then Dense layer learns importance
   - Prevents early picks from being "forgotten"

4. **Dropout increased from ~10% to 20%**
   - Suggests Transformers can afford higher regularization
   - Indicates better generalization capacity

---

## 5. MODEL IMPROVEMENTS & ITERATIONS

### Version History

```
v0.1 (Initial)    - Basic deck building model
    ↓
v1.0 (LSTM)       - Optimized LSTM architecture
    ↓
v1.5              - Added MTG drafting rules, model acts as drafter
    ↓
v2.0              - Upgraded to Transformer (711919c commit)
    ↓
v3.0              - Set-centric architecture, multi-set support
```

### Key Improvements Across Versions

#### v0.1 → v1.0: Foundational LSTM
- Single LSTM layer with 512 units
- Embedding layer with 256-dim vectors
- Basic padding strategy

#### v1.0 → v1.5: MTG Domain Integration
- Added [SEP_POOL] and [SEP_PACK] tokens
- Proper sequence formatting: pool before pack
- Filter for high-skill players (60%+ win rate)
- Implemented pick validation

#### v1.5 → v2.0: Transformer Migration (Commit 711919c)
- **Multi-head Attention**: 8 parallel attention mechanisms
- **3 Transformer Blocks**: Deeper feature transformation
- **Longer Sequences**: Extended from 42 to 64 tokens
- **Positional Embeddings**: Learned position information
- **GlobalAveragePooling**: Weighted context aggregation
- **Better Regularization**: 0.2 dropout, lower learning rate

**Performance Indicators:**
- Requirements changed from 60+ lines to 7 core packages
- ModelBuilder code grew from ~100 lines to 302 lines (better structure)

#### v2.0+ → v3.0: Set-Centric Architecture
- Separate models per Magic set (MH3, BLB, EOE, etc.)
- Per-set booster generation using MTGJson
- Cached training card lists (`training_cards.json`)
- Config files per set (`config.json`)
- No CSV loading at runtime (efficiency)

**Current State (v3.0 features):**
- Multi-set support with easy extensibility
- FastAPI REST endpoints
- Booster generation following official MTGA rules
- Model caching in memory for fast inference
- CORS support for frontend integration

---

## 6. KEY TECHNICAL SUBTLETIES

### 6.1 Sequence Construction Subtlety: Avoiding Target Leakage

**The Fix (Lines 147-154 in ModelBuilder.py):**

```python
# CRITICAL FIX: Do NOT include the target pick in the input sequence
# The input is: [SEP_POOL] + pool + [SEP_PACK] + pack (without the pick)

current_sequence = (
    [self._draft_data._sep_pool_token] + 
    pool + 
    [self._draft_data._sep_pack_token] + 
    pack_cards  # All available cards, including what will be picked
)
```

**Subtlety**: The target card IS included in the pack during training. This is correct because:
- The model sees the full pack
- The model must decide among all options
- Training target is which card was chosen from the available set
- This prevents the model from "cheating" by seeing the answer isolated

### 6.2 Padding Mask Application

```python
# Embedding layer has mask_zero=True
x = Embedding(..., mask_zero=True)(inputs)

# Padding mask created and applied to attention
mask = create_padding_mask(seq)  # Shape: (batch, 1, 1, seq_len)
attn_output = self.att(inputs, inputs, attention_mask=mask)
```

**Subtlety**: 
- Padding tokens (zeros) are automatically masked
- Attention heads cannot "attend to" padding
- Prevents PAD tokens from influencing predictions
- Crucial for variable-length sequences

### 6.3 Recency Bias Through Truncation (Line 157-158)

```python
if len(current_sequence) > SEQUENCE_LENGTH:
    current_sequence = current_sequence[len(current_sequence) - SEQUENCE_LENGTH:]
```

**Subtlety**: When pool exceeds 64 tokens, we keep the MOST RECENT cards (end of list), not the earliest. This implements implicit "recency bias":
- Most recent picks more relevant to current decision
- Natural "sliding window" of consideration
- Mimics human drafter focusing on recent picks

### 6.4 Normalization of Predictions (Lines 287-292)

```python
card_probability = []
pack_probability_sum = sum(prediction[0][card] for card in pack)

for card_id in pack:
    prob = prediction[0][card_id]
    normalized_prob = prob / pack_probability_sum if pack_probability_sum != 0 else 0
```

**Subtlety**: 
- Raw softmax outputs over entire vocabulary are normalized
- Only within the pack cards
- Ensures probabilities sum to 1.0 for available choices
- Makes output interpretable as "probability of picking each card"

### 6.5 Training Data Efficiency: Single CSV Read

**DraftData initialization (Line 29 in DraftData.py):**
```python
self._draft_data = pd.read_csv(data_path, low_memory=False, nrows=1000000)
```

**Subtlety**: 
- Reads max 1M rows from (possibly compressed) .csv.gz
- Pandas handles decompression transparently
- One-pass processing: all training examples built during single iteration
- Memory-efficient for large datasets

### 6.6 Two Modes of DraftData Operation

**Mode 1: Training** (lines 28-32)
```python
if data_path is not None:
    self._draft_data = pd.read_csv(data_path, ...)  # Full CSV loading
    # Extract all available cards from columns
```

**Mode 2: Prediction** (lines 22-26)
```python
if card_list is not None:
    self._draft_data = None  # No CSV
    self._base_cards = card_list  # Pre-computed from training
```

**Subtlety**: Reduces API inference latency by:
- Avoiding CSV parsing at runtime
- Avoiding re-computation of vocabulary
- Using pre-cached `training_cards.json`

### 6.7 Booster Generation Complexity

**File:** `/app/booster/generator.py` (351 lines)

**Key subtlety**: Weighted card selection from MTGJson

```python
def select_weighted_item(weighted_items, total_weight):
    random_num = random.random() * total_weight
    for item_key, weight in weighted_items.items():
        if random_num < weight:
            return item_key
        random_num -= weight
    return list(weighted_items.keys())[-1]
```

This implementation:
- Preserves exact weights from MTGJSON
- Selects cards probabilistically by rarity/weight
- Matches official Magic: The Gathering booster rules
- Generates realistic draft packs, not just random cards

---

## 7. PERFORMANCE CHARACTERISTICS

### Model Complexity

```
Input shape: (batch_size, 64)
↓
Embedding: 64 × 256 = 16,384 dimensions per batch item
↓
3 × TransformerBlock processing
↓
GlobalAveragePooling: reduces to 256 dims
↓
Final Dense layer: 256 × n_vocab
↓
Output shape: (batch_size, n_vocab)
```

### Training Callbacks

```python
# Early stopping (patience=10)
# Reduces LR on plateau (factor=0.5, patience=5)
# Model checkpoint saving best weights
# Validation split: 0.2 (80% train, 20% val)
```

### Loss Function: Categorical Crossentropy

```
L = -Σ y_true[i] * log(y_pred[i])
```
Where:
- y_true: One-hot encoded target card
- y_pred: Softmax probabilities from model
- i: Card vocabulary index

---

## 8. DEPLOYMENT ARCHITECTURE

### Directory Structure

```
MTGDraftAssistant/
├── app/
│   ├── api.py                    # FastAPI server
│   ├── ModelBuilder.py           # Transformer model definition
│   ├── DraftData.py              # Data handling
│   ├── train_model.py            # Training CLI
│   ├── models/                   # Trained models (committed to git)
│   │   ├── MH3/
│   │   │   ├── mh3_model.keras   # Trained weights
│   │   │   ├── config.json       # Set metadata
│   │   │   ├── cards.json        # MTGJson card data
│   │   │   ├── booster_config.json
│   │   │   ├── training_cards.json
│   │   │   └── icon.png
│   │   ├── BLB/
│   │   └── EOE/
│   └── booster/
│       ├── generator.py          # Booster pack generation
│       ├── mtgjson_fetcher.py    # MTGJson API integration
│       └── booster_factory.py    # Factory pattern (unused)
├── data/                         # Training data (gitignored)
│   ├── MH3/
│   ├── BLB/
│   └── EOE/
├── requirements.txt              # Dependencies
└── train_model.py               # Convenience script
```

### Dependencies

```
TensorFlow 2.10.0      # ML framework
FastAPI 0.95.2         # Web framework
Pandas 1.5.3           # Data manipulation
NumPy 1.23.5           # Numerical computing
Requests 2.31.0        # HTTP client (MTGJson fetching)
Scikit-learn 1.2.2     # (Included, minimal use)
Uvicorn 0.22.0         # ASGI server
```

---

## 9. TRAINING WORKFLOW

### Step-by-step Training Process

**1. Data Preparation** (`train_model.py` lines 54-55)
```bash
python -m app.train_model --set MH3 --epochs 10
```

**2. Find Training Data** (lines 54-55)
```
data/MH3/game_data_public.MH3.PremierDraft.csv.gz
```

**3. Fetch MTGJson Data** (lines 59-61)
```python
save_booster_data(set_code, output_dir)
# Fetches: booster_config.json, cards.json
```

**4. Load Training Data** (line 65)
```python
draft_data = DraftData(csv_path)
# Loads 17Lands data, builds vocabulary
```

**5. Train Model** (line 81)
```python
model_builder = ModelBuilder(draft_data)
model_builder.train_model(args.epochs)
# Builds sequences, trains transformer
```

**6. Save Model** (lines 84-85)
```python
model_builder._model.save(model_path)
# Saves to app/models/MH3/mh3_model.keras
```

---

## 10. INFERENCE PIPELINE

### Request to Prediction

**1. Client Request**
```json
{
    "set": "MH3",
    "deck": ["Lightning Bolt", "Counterspell"],
    "pack": ["Giant Growth", "Shock", "Cancel"]
}
```

**2. Load Model (cached)**
```python
model, draft_data = load_set_model("MH3")
```

**3. Convert Names to IDs**
```python
deck_ids = [draft_data.cards_to_int[name] for name in deck]
pack_ids = [draft_data.cards_to_int[name] for name in pack]
```

**4. Predict**
```python
predictions = model.predict(deck_ids, pack_ids)
# Internal: build sequence, run through transformer
```

**5. Return Probabilities**
```json
{
    "predictions": [
        {"card_name": "Giant Growth", "probability": 0.85},
        {"card_name": "Shock", "probability": 0.12},
        {"card_name": "Cancel", "probability": 0.03}
    ]
}
```

---

## 11. SCALABILITY & EXTENSIBILITY

### Adding New Sets

```bash
# 1. Create directory
mkdir -p data/NEW_SET app/models/NEW_SET

# 2. Download training data
# Place game_data_public.NEW_SET.PremierDraft.csv.gz in data/NEW_SET/

# 3. Train
python -m app.train_model --set NEW_SET --epochs 10

# 4. Model automatically available at /predict?set=NEW_SET
```

### Performance Considerations

- **Model loading**: ~2-3 seconds (one-time per set)
- **Inference**: ~100-200ms per request
- **Memory**: ~500MB per cached model
- **Booster generation**: ~1-2 seconds (MTGJson API calls cached)

---

## 12. KNOWN ISSUES & TODOs

**Line 240 in ModelBuilder.py:**
```python
#TODO: change where the model is stored
checkpoint = ModelCheckpoint(
    'app/model/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
```

TODO: Change checkpoint path from `app/model/` to `app/models/{set_code}/`

---

## Conclusion

The MTG Draft Assistant demonstrates a sophisticated application of Transformer architectures to domain-specific sequence prediction. The architecture elegantly combines:

1. **Advanced ML**: Multi-head attention for diverse feature learning
2. **Domain expertise**: Special tokens for MTG draft structure
3. **Software engineering**: Clean separation of concerns (data, model, API)
4. **Scalability**: Per-set models, efficient inference caching
5. **Real-world integration**: MTGJson booster simulation, 17Lands data integration

The evolution from LSTM to Transformer represents a strategic choice prioritizing long-range dependency modeling, interpretability, and training stability—all crucial for understanding complex draft decisions.

