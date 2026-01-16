# Design Document

## Overview

The MTG Draft Assistant codebase cleanup will transform the current mixed-framework, tightly-coupled architecture into a clean, layered system with clear separation of concerns. The cleanup addresses framework inconsistencies (TensorFlow vs PyTorch), excessive JSON file generation, unclear code organization, and tight coupling between components.

## Architecture

### Current State Analysis

**Framework Inconsistencies:**
- TensorFlow/Keras: `ModelBuilder.py`, `DraftData.py`, `api.py` (current production system)
- PyTorch: `CandidateTower.py`, `ContextTower.py`, `ScoringHead.py`, `CardEncoder.py` (future two-tower architecture)
- Mixed usage creates confusion but represents transition to new architecture

**JSON File Proliferation:**
- `cards.json` - MTGJson card data (essential for API)
- `booster_config.json` - Booster generation rules (essential for API)
- `sheets.json` - Filtered booster sheets (essential for API)
- `training_cards.json` - Card list for training (essential for training)
- Multiple intermediate JSON files during preprocessing

**Organizational Issues:**
- Mixed responsibilities in single files
- Unclear module boundaries
- Inconsistent naming patterns
- Dead/unused code (PyTorch components)

### Target Architecture

```
app/
├── api/                    # API Layer
│   ├── __init__.py
│   ├── main.py            # FastAPI app and routing
│   ├── models.py          # Pydantic request/response models
│   └── dependencies.py    # Dependency injection
├── core/                  # Core Business Logic
│   ├── __init__.py
│   ├── prediction.py      # Prediction orchestration
│   └── booster.py         # Booster generation
├── ml/                    # Machine Learning Layer
│   ├── __init__.py
│   ├── current/           # Current TensorFlow-based system
│   │   ├── model_builder.py
│   │   ├── model_loader.py
│   │   └── transformer.py
│   └── experimental/      # Future PyTorch two-tower architecture
│       ├── candidate_tower.py
│       ├── context_tower.py
│       ├── scoring_head.py
│       └── card_encoder.py
├── data/                  # Data Layer
│   ├── __init__.py
│   ├── draft_data.py      # Draft data handling
│   └── repositories.py    # Data access patterns
├── services/              # External Services
│   ├── __init__.py
│   └── mtgjson.py         # MTGJson API integration
└── utils/                 # Utilities
    ├── __init__.py
    ├── config.py          # Configuration management
    └── cache.py           # Caching utilities
```

## Components and Interfaces

### 1. API Layer (`app/api/`)

**Responsibilities:**
- HTTP request/response handling
- Input validation
- Response formatting
- CORS and middleware configuration

**Key Interfaces:**
```python
# app/api/main.py
class PredictionAPI:
    def __init__(self, prediction_service: PredictionService)
    
    async def predict(self, request: PredictRequest) -> PredictResponse
    async def get_booster(self, set_code: str) -> BoosterResponse
    async def get_sets() -> SetsResponse

# app/api/models.py
class PredictRequest(BaseModel):
    set: str
    deck: List[str]
    pack: List[str]

class PredictResponse(BaseModel):
    set: str
    predictions: List[CardPrediction]
```

### 2. Core Business Logic (`app/core/`)

**Responsibilities:**
- Orchestrate prediction workflow
- Coordinate between ML and data layers
- Business rule enforcement

**Key Interfaces:**
```python
# app/core/prediction.py
class PredictionService:
    def __init__(self, model_loader: ModelLoader, data_repo: DataRepository)
    
    def predict_picks(self, set_code: str, deck: List[str], pack: List[str]) -> List[CardPrediction]

# app/core/booster.py
class BoosterService:
    def __init__(self, data_repo: DataRepository)
    
    def generate_booster(self, set_code: str) -> List[str]
```

### 3. Machine Learning Layer (`app/ml/`)

**Responsibilities:**
- Model training, loading, and inference
- TensorFlow-specific operations
- Model caching and optimization

**Key Interfaces:**
```python
# app/ml/model_loader.py
class ModelLoader:
    def load_model(self, set_code: str) -> TensorFlowModel
    def get_cached_model(self, set_code: str) -> Optional[TensorFlowModel]

# app/ml/model_builder.py
class ModelBuilder:
    def train_model(self, draft_data: DraftData) -> TensorFlowModel
    def save_model(self, model: TensorFlowModel, path: str)
```

### 4. Data Layer (`app/data/`)

**Responsibilities:**
- Data loading and preprocessing
- Card encoding and transformation
- File I/O operations

**Key Interfaces:**
```python
# app/data/repositories.py
class DataRepository:
    def get_set_data(self, set_code: str) -> SetData
    def get_card_encodings(self, set_code: str) -> Dict[str, np.ndarray]
    def get_booster_config(self, set_code: str) -> BoosterConfig

# app/data/card_encoder.py
class CardEncoder:
    def encode_card(self, card: Card) -> np.ndarray
    def encode_batch(self, cards: List[Card]) -> np.ndarray
```

## Data Models

### Core Data Structures

```python
@dataclass
class Card:
    name: str
    uuid: str
    rarity: str
    colors: List[str]
    types: List[str]
    mana_cost: str
    oracle_text: str
    power: Optional[int] = None
    toughness: Optional[int] = None

@dataclass
class SetData:
    code: str
    name: str
    cards: List[Card]
    has_model: bool
    has_icon: bool

@dataclass
class CardPrediction:
    card_name: str
    probability: float

@dataclass
class BoosterConfig:
    play: Dict[str, Any]
    sheets: Dict[str, Dict[str, float]]
```

### File Structure Optimization

**Essential JSON Files (Keep):**
- `app/models/{SET}/config.json` - Set metadata
- `app/models/{SET}/training_cards.json` - Cards used in training
- `app/models/{SET}/booster_config.json` - Booster generation rules
- `app/models/{SET}/sheets.json` - Processed booster sheets

**Remove/Consolidate:**
- `app/models/{SET}/cards.json` - Merge into booster_config or eliminate
- Intermediate JSON files during preprocessing
- Duplicate card data storage

## Error Handling

### Centralized Error Management

```python
# app/utils/exceptions.py
class MTGDraftError(Exception):
    """Base exception for MTG Draft Assistant"""

class ModelNotFoundError(MTGDraftError):
    """Raised when a model for a set is not found"""

class InvalidSetError(MTGDraftError):
    """Raised when an invalid set code is provided"""

class PredictionError(MTGDraftError):
    """Raised when prediction fails"""
```

### Error Handling Strategy

1. **API Layer**: Catch all exceptions and return appropriate HTTP status codes
2. **Service Layer**: Raise domain-specific exceptions with clear messages
3. **Data Layer**: Handle file I/O errors and data validation errors
4. **ML Layer**: Handle model loading and prediction errors

## Testing Strategy

### Test Organization

```
tests/
├── unit/
│   ├── test_api/
│   ├── test_core/
│   ├── test_ml/
│   └── test_data/
├── integration/
│   ├── test_prediction_flow.py
│   └── test_booster_generation.py
└── fixtures/
    ├── sample_cards.json
    └── mock_models/
```

### Testing Approach

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Mock External Dependencies**: MTGJson API, file system operations
4. **Test Data**: Use minimal, focused test datasets

## Migration Strategy

### Phase 1: Framework Organization
1. Organize current TensorFlow system and experimental PyTorch components
2. Create clear separation between production and experimental code
3. Update dependencies to support both frameworks during transition

### Phase 2: Layer Separation
1. Create new directory structure
2. Extract API logic from business logic
3. Separate ML operations from data operations

### Phase 3: Interface Definition
1. Define clear interfaces between layers
2. Implement dependency injection
3. Add type hints and documentation

### Phase 4: Data Optimization
1. Audit JSON file usage
2. Consolidate or eliminate unnecessary files
3. Optimize data loading patterns

### Phase 5: Testing and Validation
1. Ensure all functionality works after refactoring
2. Add comprehensive tests
3. Performance validation

## Performance Considerations

### Caching Strategy
- Model caching to avoid repeated loading
- Data caching for frequently accessed sets
- HTTP response caching for static data

### Memory Management
- Lazy loading of models and data
- Proper cleanup of TensorFlow resources
- Efficient data structures for card encodings

### Scalability
- Stateless service design
- Configurable batch sizes
- Resource pooling for concurrent requests