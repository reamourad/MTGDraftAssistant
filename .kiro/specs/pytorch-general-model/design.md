# Design Document

## Overview

This design outlines the implementation of a complete PyTorch two-tower model training pipeline for MTG draft pick prediction. The system will train a general-purpose, set-agnostic model using data from multiple MTG sets. The architecture consists of three main neural network components (CandidateTower, ContextTower, ScoringHead), a training pipeline for processing 17Lands data, and integration with the existing API infrastructure.

## Architecture

### System Components

```
app/
├── ml/
│   └── experimental/              # PyTorch two-tower architecture
│       ├── two_tower_model.py     # Main model (already exists)
│       ├── candidate_tower.py     # NEW: Encodes card features
│       ├── context_tower.py       # NEW: Encodes draft context
│       ├── scoring_head.py        # NEW: Combines embeddings
│       ├── card_encoder.py        # Existing: 407-dim encoding
│       └── model_loader.py        # Existing: Model management
│
├── training/                      # NEW: Training infrastructure
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset for draft data
│   ├── data_loader.py             # Data loading from 17Lands
│   ├── trainer.py                 # Training loop implementation
│   ├── evaluator.py               # Validation and metrics
│   └── config.py                  # Training configuration
│
├── core/
│   └── pytorch_prediction.py      # NEW: PyTorch prediction service
│
└── api/
    └── main.py                    # Update: Add PyTorch endpoint

scripts/
└── train_pytorch.py               # NEW: CLI training script
```

### Data Flow

```
17Lands CSV → DataLoader → CardEncoder → PyTorch Dataset → Training Loop
                                                                  ↓
                                                            TwoTowerModel
                                                                  ↓
                                                          Model Checkpoint
                                                                  ↓
                                                          API Integration
```

## Components and Interfaces

### 1. Neural Network Components

#### CandidateTower
Encodes individual card features into embeddings.

```python
# app/ml/experimental/candidate_tower.py
class CandidateTower(nn.Module):
    """
    Encodes card features into embeddings.
    
    Architecture:
        Input (407) → Linear(256) → ReLU → Dropout(0.2) →
        Linear(256) → ReLU → Dropout(0.2) →
        Linear(128) → Output Embedding
    """
    
    def __init__(
        self,
        input_dim: int = 407,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, card_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            card_features: (batch, 407) card feature vectors
        Returns:
            embeddings: (batch, 128) card embeddings
        """
        return self.network(card_features)
```

#### ContextTower
Encodes draft context (pool, pack, pick number) into embeddings.

```python
# app/ml/experimental/context_tower.py
class ContextTower(nn.Module):
    """
    Encodes draft context into embeddings.
    
    Architecture:
        Pool Cards → Mean Pooling → (407)
        Pack Cards → Mean Pooling → (407)
        Pick Number → Embedding → (16)
        Concatenate → (407 + 407 + 16 = 830)
        → Linear(512) → ReLU → Dropout(0.2)
        → Linear(256) → ReLU → Dropout(0.2)
        → Linear(128) → Output Embedding
    """
    
    def __init__(
        self,
        card_dim: int = 407,
        hidden_dim: int = 256,
        output_dim: int = 128,
        max_picks: int = 45,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Pick number embedding
        self.pick_embedding = nn.Embedding(max_picks + 1, 16)
        
        # Context processing network
        context_input_dim = card_dim * 2 + 16  # pool + pack + pick
        self.network = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        pool_cards: torch.Tensor,
        pack_cards: torch.Tensor,
        pick_number: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pool_cards: (batch, num_picked, 407) cards in pool
            pack_cards: (batch, num_available, 407) cards in pack
            pick_number: (batch, 1) current pick number
        Returns:
            embeddings: (batch, 128) context embeddings
        """
        # Aggregate pool and pack using mean pooling
        pool_agg = pool_cards.mean(dim=1)  # (batch, 407)
        pack_agg = pack_cards.mean(dim=1)  # (batch, 407)
        
        # Embed pick number
        pick_emb = self.pick_embedding(pick_number.long().squeeze(-1))  # (batch, 16)
        
        # Concatenate all context features
        context = torch.cat([pool_agg, pack_agg, pick_emb], dim=-1)
        
        return self.network(context)
```

#### ScoringHead
Combines candidate and context embeddings to produce pick scores.

```python
# app/ml/experimental/scoring_head.py
class ScoringHead(nn.Module):
    """
    Combines candidate and context embeddings to produce scores.
    
    Architecture:
        [Candidate(128) + Context(128)] → (256)
        → Linear(128) → ReLU → Dropout(0.2)
        → Linear(64) → ReLU → Dropout(0.2)
        → Linear(1) → Score
    """
    
    def __init__(
        self,
        input_dim: int = 256,  # candidate + context
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        candidate_embedding: torch.Tensor,
        context_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            candidate_embedding: (batch, 128) candidate embeddings
            context_embedding: (batch, 128) context embeddings
        Returns:
            scores: (batch, 1) pick scores
        """
        combined = torch.cat([candidate_embedding, context_embedding], dim=-1)
        return self.network(combined)
```

### 2. Training Infrastructure

#### DraftDataset
PyTorch Dataset for draft sequences.

```python
# app/training/dataset.py
class DraftDataset(Dataset):
    """
    PyTorch Dataset for draft pick sequences.
    
    Each sample represents a single pick decision:
    - Pool: Cards already picked
    - Pack: Cards available to pick from
    - Pick number: Current pick in draft (1-45)
    - Target: Index of card that was actually picked
    """
    
    def __init__(
        self,
        draft_sequences: List[DraftSequence],
        card_encoder: CardEncoder,
        max_pool_size: int = 45,
        max_pack_size: int = 15
    ):
        self.sequences = draft_sequences
        self.encoder = card_encoder
        self.max_pool_size = max_pool_size
        self.max_pack_size = max_pack_size
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'pool_cards': (max_pool_size, 407),
                'pack_cards': (max_pack_size, 407),
                'pick_number': (1,),
                'target_idx': (1,)  # Index of picked card in pack
            }
        """
        seq = self.sequences[idx]
        
        # Encode cards
        pool_encoded = self.encoder.encode_batch_by_names(seq.pool)
        pack_encoded = self.encoder.encode_batch_by_names(seq.pack)
        
        # Pad to max sizes
        pool_padded = self._pad_cards(pool_encoded, self.max_pool_size)
        pack_padded = self._pad_cards(pack_encoded, self.max_pack_size)
        
        # Find target index
        target_idx = seq.pack.index(seq.picked_card)
        
        return {
            'pool_cards': torch.tensor(pool_padded, dtype=torch.float32),
            'pack_cards': torch.tensor(pack_padded, dtype=torch.float32),
            'pick_number': torch.tensor([seq.pick_number], dtype=torch.float32),
            'target_idx': torch.tensor([target_idx], dtype=torch.long)
        }
```

#### DataLoader
Loads and preprocesses 17Lands data.

```python
# app/training/data_loader.py
class DraftDataLoader:
    """
    Loads draft data from 17Lands CSV files.
    
    Processes game logs into individual pick sequences.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_set_data(
        self,
        set_code: str,
        min_win_rate: float = 0.60,
        limit: Optional[int] = None
    ) -> List[DraftSequence]:
        """
        Load draft sequences from a set's CSV file.
        
        Args:
            set_code: MTG set code (e.g., 'MH3')
            min_win_rate: Minimum player win rate to include
            limit: Maximum number of rows to process
        
        Returns:
            List of DraftSequence objects
        """
        csv_path = self._find_csv_file(set_code)
        
        # Read CSV (handles .gz compression automatically)
        df = pd.read_csv(csv_path, nrows=limit)
        
        # Filter by win rate
        df = df[df['user_game_win_rate_bucket'] >= min_win_rate]
        
        # Extract draft sequences
        sequences = self._extract_sequences(df)
        
        return sequences
    
    def load_multi_set_data(
        self,
        set_codes: List[str],
        **kwargs
    ) -> List[DraftSequence]:
        """Load data from multiple sets for general model training."""
        all_sequences = []
        
        for set_code in set_codes:
            sequences = self.load_set_data(set_code, **kwargs)
            all_sequences.extend(sequences)
        
        return all_sequences
```

#### Trainer
Training loop implementation.

```python
# app/training/trainer.py
class TwoTowerTrainer:
    """
    Handles training of the two-tower model.
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        train_dataset: DraftDataset,
        val_dataset: DraftDataset,
        config: TrainingConfig
    ):
        self.model = model
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        self.config = config
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu'
        )
        self.model.to(self.device)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training history (losses, metrics)
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_top1_acc': [],
            'val_top3_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Log metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_top1_acc'].append(val_metrics['top1_acc'])
            history['val_top3_acc'].append(val_metrics['top3_acc'])
            
            # Save checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move to device
            pool = batch['pool_cards'].to(self.device)
            pack = batch['pack_cards'].to(self.device)
            pick_num = batch['pick_number'].to(self.device)
            target = batch['target_idx'].to(self.device)
            
            # Forward pass
            scores = self.model(pack, pool, pack, pick_num)
            scores = scores.squeeze(-1)  # (batch, num_cards)
            
            # Compute loss
            loss = self.criterion(scores, target.squeeze(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
```

### 3. API Integration

#### PyTorch Prediction Service

```python
# app/core/pytorch_prediction.py
class PyTorchPredictionService:
    """
    Prediction service using PyTorch two-tower model.
    """
    
    def __init__(
        self,
        model_loader: PyTorchModelLoader,
        card_encoder: CardEncoder
    ):
        self.model_loader = model_loader
        self.encoder = card_encoder
    
    def predict_picks(
        self,
        set_code: str,
        deck: List[str],
        pack: List[str]
    ) -> List[CardPrediction]:
        """
        Predict best picks using PyTorch model.
        
        Args:
            set_code: MTG set code (may be ignored for general model)
            deck: List of card names in pool
            pack: List of card names in current pack
        
        Returns:
            List of predictions sorted by score
        """
        # Load model
        model = self.model_loader.load_model("general")  # General model
        
        # Encode cards
        pool_encoded = self.encoder.encode_batch_by_names(deck)
        pack_encoded = self.encoder.encode_batch_by_names(pack)
        
        # Convert to tensors
        pool_tensor = torch.tensor(pool_encoded, dtype=torch.float32)
        pack_tensor = torch.tensor(pack_encoded, dtype=torch.float32)
        pick_number = len(deck) + 1
        
        # Get predictions
        scores = model.predict_pick(
            pack_tensor,
            pool_tensor,
            pack_tensor,
            pick_number
        )
        
        # Sort and format
        predictions = []
        for idx, score in enumerate(scores):
            predictions.append(CardPrediction(
                card_name=pack[idx],
                probability=float(torch.sigmoid(score))
            ))
        
        predictions.sort(key=lambda x: x.probability, reverse=True)
        return predictions
```

## Data Models

```python
@dataclass
class DraftSequence:
    """Represents a single pick in a draft."""
    draft_id: str
    pick_number: int
    pool: List[str]  # Card names already picked
    pack: List[str]  # Card names available
    picked_card: str  # Card that was picked

@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 5
    use_gpu: bool = True
    num_workers: int = 4
    checkpoint_dir: str = "app/models/general"

@dataclass
class CardPrediction:
    """Prediction result for a single card."""
    card_name: str
    probability: float
```

## Error Handling

```python
class PyTorchTrainingError(Exception):
    """Raised when training fails."""
    pass

class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass

class EncodingError(Exception):
    """Raised when card encoding fails."""
    pass
```

## Testing Strategy

### Unit Tests
- Test each neural network component independently
- Test data loading and preprocessing
- Test card encoding with sample data

### Integration Tests
- Test full training pipeline with small dataset
- Test model saving and loading
- Test prediction service end-to-end

### Validation Tests
- Verify model output shapes
- Verify gradient flow during training
- Verify checkpoint compatibility

## Performance Considerations

### Training Optimization
- Use GPU when available (10-50x speedup)
- Batch processing for efficient GPU utilization
- Data loading with multiple workers
- Mixed precision training (optional)

### Memory Management
- Gradient accumulation for large batches
- Periodic cache clearing
- Efficient data structures for sequences

### Scalability
- Support for distributed training (future)
- Checkpoint resumption for long training runs
- Incremental training on new sets
