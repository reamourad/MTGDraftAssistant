# PyTorch Two-Tower Model Training Guide

This guide covers training the general-purpose PyTorch two-tower model for MTG draft pick prediction. The model uses a set-agnostic architecture that can handle cards from any MTG set using unified 407-dimensional card encodings.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Training Arguments](#training-arguments)
- [Example Commands](#example-commands)
- [Monitoring Training](#monitoring-training)
- [Evaluating the Model](#evaluating-the-model)
- [Troubleshooting](#troubleshooting)

## Overview

The PyTorch two-tower model consists of:
- **Candidate Tower**: Encodes individual card features (407 dims → 128 dims)
- **Context Tower**: Encodes draft state (pool + pack + pick number → 128 dims)
- **Scoring Head**: Combines embeddings to produce pick scores (256 dims → 1)

The model is trained on real player draft data from [17Lands](https://www.17lands.com), learning from high-skilled players with 60%+ win rates.

## Hardware Requirements

### Recommended Configuration

- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, RTX 4060, or better)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free disk space for data and checkpoints
- **CPU**: Modern multi-core processor (4+ cores recommended)

### GPU Training (Recommended)

GPU training provides 10-50x speedup compared to CPU:

| Configuration | Training Time (20 epochs, single set) |
|---------------|--------------------------------------|
| RTX 4090      | ~15-20 minutes                       |
| RTX 3070      | ~30-40 minutes                       |
| RTX 2060      | ~60-90 minutes                       |

**GPU Memory Requirements:**
- Single set training: 4-6GB VRAM
- Multi-set training (3+ sets): 6-8GB VRAM
- Batch size 32: ~4GB VRAM
- Batch size 64: ~6GB VRAM

### CPU-Only Training

CPU training is supported but significantly slower:

| Configuration | Training Time (20 epochs, single set) |
|---------------|--------------------------------------|
| 8-core CPU    | ~8-12 hours                          |
| 4-core CPU    | ~16-24 hours                         |

**To train on CPU only:**
```bash
python scripts/train_pytorch.py --sets MH3 --no-gpu
```

**CPU Training Tips:**
- Reduce batch size to 16 or 8 to lower memory usage
- Use fewer workers: `--num-workers 2`
- Consider training on a subset of data first: `--limit 50000`
- Train overnight or during off-hours

## Data Preparation

### Step 1: Download Training Data

1. Visit [17Lands Public Datasets](https://www.17lands.com/public_datasets)
2. Download Premier Draft data for your desired sets
3. Look for files named: `game_data_public.{SET}.PremierDraft.csv.gz`

**Available sets** (as of January 2025):
- MH3 (Modern Horizons 3)
- BLB (Bloomburrow)
- FIN (Foundations)
- EOE (Edges of Eternities)
- TLA (The Lost Caverns of Ixalan)

### Step 2: Organize Data Files

Place downloaded files in the appropriate set directories:

```
data/
├── MH3/
│   └── game_data_public.MH3.PremierDraft.csv.gz
├── BLB/
│   └── game_data_public.BLB.PremierDraft.csv.gz
├── FIN/
│   └── game_data_public.FIN.PremierDraft.csv.gz
└── TLA/
    └── game_data_public.TLA.PremierDraft.csv.gz
```

**Important:** Keep files in `.csv.gz` compressed format. The training script reads compressed files directly - no need to unzip!

### Step 3: Preprocess Card Data (REQUIRED)

Before training, you must run the preprocessing script to:
- Extract card names from 17Lands CSV files
- Fetch card data from MTGJson API
- Validate that all 17Lands cards exist in MTGJson
- Create 407-dimensional card encodings
- Save processed data for training

**Run preprocessing for your sets:**

```bash
# Preprocess single set
python preprocess_cards.py MH3

# Preprocess multiple sets
python preprocess_cards.py MH3 BLB FIN

# Preprocess all sets in data/ directory
python preprocess_cards.py
```

**What preprocessing creates:**

```
app/models/
├── MH3/
│   ├── cards.json              # Card data from MTGJson
│   ├── card_encodings.pkl      # Pre-encoded 407-dim vectors
│   ├── training_cards.json     # List of card names
│   ├── booster_config.json     # Booster generation rules
│   └── sheets.json             # Weighted card sheets
├── BLB/
│   └── ...
└── FIN/
    └── ...
```

**Important Notes:**
- Preprocessing validates that all cards from 17Lands exist in MTGJson
- If cards are missing, you'll see warnings but preprocessing continues
- Missing cards will be skipped during training
- You only need to run preprocessing once per set (or when data changes)

**Example output:**
```
[Step 1/6] Extracting card names from CSV...
Found 303 unique cards in CSV

[Step 2/6] Fetching MTGJson data...
Successfully fetched data for MH3

[Step 5/6] Encoding cards...
WARNING: 2 cards could not be found:
  - Some Promo Card
  - Another Missing Card

Encoded 301/303 cards
Saved to app/models/MH3/card_encodings.pkl
```

## Training the Model

### Prerequisites

Before training, ensure you have:
1. ✓ Downloaded 17Lands CSV files to `data/{SET}/`
2. ✓ Run preprocessing: `python preprocess_cards.py {SETS}`
3. ✓ Verified `app/models/{SET}/cards.json` exists for each set

### Basic Training Command

Train on a single set:

```bash
# Step 1: Preprocess (if not done already)
python preprocess_cards.py MH3

# Step 2: Train
python scripts/train_pytorch.py --sets MH3 --epochs 20
```

Train on multiple sets (general model):

```bash
# Step 1: Preprocess all sets
python preprocess_cards.py MH3 BLB FIN

# Step 2: Train on combined data
python scripts/train_pytorch.py --sets MH3 BLB FIN --epochs 30
```

### Training Process

The training script performs these steps:

1. **Load card data** from all specified sets
2. **Initialize CardEncoder** with combined card pool
3. **Load draft sequences** from 17Lands CSV files
4. **Split data** into training (80%) and validation (20%)
5. **Create PyTorch datasets** with proper batching
6. **Initialize model** (or resume from checkpoint)
7. **Train model** with early stopping
8. **Save checkpoints** and training history

### Output Files

Training creates these files in the output directory (default: `app/models/general/`):

```
app/models/general/
├── best_model.pt              # Best model checkpoint
├── checkpoint_epoch_10.pt     # Periodic checkpoints
├── checkpoint_epoch_20.pt
├── training_history.json      # Loss and accuracy curves
└── training_20250114_143022.log  # Detailed training log
```

## Training Arguments

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sets` | *required* | MTG set codes to train on (e.g., MH3 BLB) |
| `--data-dir` | `data` | Directory containing set data folders |
| `--min-win-rate` | `0.60` | Minimum player win rate to include drafts |
| `--limit` | `None` | Limit rows per set (for testing) |

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `32` | Batch size for training |
| `--lr` | `0.001` | Learning rate (Adam optimizer) |
| `--patience` | `5` | Early stopping patience (epochs) |
| `--train-split` | `0.8` | Fraction of data for training |

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | `256` | Hidden layer dimension |
| `--embedding-dim` | `128` | Embedding dimension for towers |

### Output and System

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `app/models/general` | Directory for checkpoints |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--no-gpu` | `False` | Disable GPU (use CPU only) |
| `--num-workers` | `4` | Data loading workers |
| `--verbose` | `False` | Enable verbose logging |

## Example Commands

### Single-Set Training (Fast)

Train a model on one set for quick iteration:

```bash
python scripts/train_pytorch.py \
    --sets MH3 \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.001
```

**Use case:** Testing, debugging, or set-specific model

### Multi-Set General Model (Recommended)

Train a general model on multiple sets:

```bash
python scripts/train_pytorch.py \
    --sets MH3 BLB FIN TLA \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --patience 7
```

**Use case:** Production model that works across all sets

### High-Performance Training

Use larger batch size and more workers with powerful GPU:

```bash
python scripts/train_pytorch.py \
    --sets MH3 BLB FIN \
    --epochs 25 \
    --batch-size 64 \
    --lr 0.001 \
    --num-workers 8
```

**Requirements:** 8GB+ VRAM, 8+ CPU cores

### CPU-Only Training

Train without GPU (slower but works on any machine):

```bash
python scripts/train_pytorch.py \
    --sets MH3 \
    --epochs 20 \
    --batch-size 16 \
    --no-gpu \
    --num-workers 2
```

### Quick Test Run

Test the pipeline with limited data:

```bash
python scripts/train_pytorch.py \
    --sets MH3 \
    --epochs 3 \
    --batch-size 16 \
    --limit 10000 \
    --verbose
```

**Use case:** Verify setup, test changes, debug issues

### Resume from Checkpoint

Continue training from a saved checkpoint:

```bash
python scripts/train_pytorch.py \
    --sets MH3 BLB FIN \
    --epochs 40 \
    --resume app/models/general/best_model.pt
```

**Use case:** Extend training, recover from interruption

### Custom Output Directory

Save model to a specific location:

```bash
python scripts/train_pytorch.py \
    --sets MH3 \
    --epochs 20 \
    --output-dir app/models/experimental/mh3_v2
```

## Monitoring Training

### Console Output

Training progress is logged to console in real-time:

```
2025-01-14 14:30:22 - INFO - Step 1: Loading card data
2025-01-14 14:30:23 - INFO - Loaded 303 cards from MH3
2025-01-14 14:30:23 - INFO - Total unique cards loaded: 303

2025-01-14 14:30:25 - INFO - Step 3: Loading draft sequences
2025-01-14 14:30:45 - INFO - Loaded 125,432 draft sequences

2025-01-14 14:31:10 - INFO - Step 8: Starting training
2025-01-14 14:31:10 - INFO - Epoch 1/20
2025-01-14 14:32:15 - INFO - Train Loss: 2.3456 | Val Loss: 2.1234 | Top-1: 0.3421 | Top-3: 0.6543
2025-01-14 14:33:20 - INFO - Epoch 2/20
2025-01-14 14:34:25 - INFO - Train Loss: 2.0123 | Val Loss: 1.9876 | Top-1: 0.3789 | Top-3: 0.6821
...
```

### Training Metrics

Key metrics to monitor:

- **Train Loss**: Should decrease steadily
- **Val Loss**: Should decrease; if it increases, model is overfitting
- **Top-1 Accuracy**: % of times model's top pick matches player's pick
- **Top-3 Accuracy**: % of times player's pick is in model's top 3

**Good training indicators:**
- Train loss decreases consistently
- Val loss decreases and stabilizes
- Top-1 accuracy reaches 35-45%
- Top-3 accuracy reaches 65-75%

### Log Files

Detailed logs are saved to the output directory:

```bash
# View training log
cat app/models/general/training_20250114_143022.log

# Monitor training in real-time
tail -f app/models/general/training_20250114_143022.log
```

### Training History

After training, view the history JSON:

```bash
# View training history
cat app/models/general/training_history.json
```

Example output:
```json
{
  "train_loss": [2.34, 2.01, 1.87, ...],
  "val_loss": [2.12, 1.98, 1.85, ...],
  "val_top1_acc": [0.34, 0.38, 0.41, ...],
  "val_top3_acc": [0.65, 0.68, 0.71, ...]
}
```

## Evaluating the Model

### During Training

Validation metrics are computed after each epoch:
- Top-1 accuracy: Model's top pick matches player's pick
- Top-3 accuracy: Player's pick is in model's top 3
- Validation loss: Cross-entropy loss on validation set

### After Training

Test the trained model using the API:

1. **Start the API server:**
   ```bash
   uvicorn app.api:app --reload
   ```

2. **Check model status:**
   ```bash
   curl http://localhost:8000/status
   ```

3. **Make a prediction:**
   ```bash
   curl -X POST http://localhost:8000/predict_pytorch \
     -H "Content-Type: application/json" \
     -d '{
       "set": "MH3",
       "deck": ["Lightning Bolt", "Counterspell"],
       "pack": ["Giant Growth", "Shock", "Cancel"]
     }'
   ```

### Benchmark Performance

Expected performance on validation data:

| Metric | Single Set | Multi-Set General |
|--------|-----------|-------------------|
| Top-1 Accuracy | 40-45% | 35-42% |
| Top-3 Accuracy | 70-75% | 65-72% |
| Top-5 Accuracy | 80-85% | 75-82% |

**Note:** Multi-set models have slightly lower accuracy but work across all sets.

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 16` or `--batch-size 8`
2. Reduce number of workers: `--num-workers 2`
3. Use CPU training: `--no-gpu`
4. Close other GPU applications
5. Train on fewer sets at once

### Slow Training

**Symptoms:**
- Training takes hours per epoch
- GPU utilization is low

**Solutions:**
1. Verify GPU is being used: Check console output for "GPU enabled: True"
2. Increase batch size if memory allows: `--batch-size 64`
3. Increase workers: `--num-workers 8`
4. Check GPU drivers are up to date
5. Ensure PyTorch is installed with CUDA support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Preprocessing Not Run

**Symptoms:**
```
FileNotFoundError: Card data not found for MH3: app/models/MH3/cards.json
Please run preprocessing first: python preprocess_cards.py MH3
```

**Solutions:**
1. Run preprocessing before training:
   ```bash
   python preprocess_cards.py MH3
   ```
2. Verify preprocessing completed successfully
3. Check that `app/models/MH3/cards.json` exists

### Data Loading Errors

**Symptoms:**
```
FileNotFoundError: data/MH3/game_data_public.MH3.PremierDraft.csv.gz
```

**Solutions:**
1. Verify CSV files are in correct directories
2. Check file names match expected format
3. Ensure files are not corrupted (re-download if needed)
4. Verify set codes are correct (case-sensitive)
5. Make sure you downloaded the CSV from 17Lands

### Card Encoding Errors

**Symptoms:**
```
KeyError: 'Lightning Bolt'
EncodingError: Card not found in encoder
```

**Solutions:**
1. Verify card data JSON files exist for all sets
2. Check card names match between CSV and JSON files
3. Ensure CardEncoder is initialized with all sets' card data
4. Look for typos or special characters in card names

### Model Not Improving

**Symptoms:**
- Validation loss not decreasing
- Accuracy stuck at low values
- Training loss decreasing but validation loss increasing

**Solutions:**
1. **Overfitting:** Reduce model size or add regularization
2. **Learning rate too high:** Try `--lr 0.0005` or `--lr 0.0001`
3. **Learning rate too low:** Try `--lr 0.002` or `--lr 0.005`
4. **Insufficient data:** Train on more sets or reduce `--min-win-rate`
5. **Data quality:** Check for corrupted or invalid draft sequences

### Early Stopping Too Soon

**Symptoms:**
- Training stops after 5-7 epochs
- Model hasn't converged yet

**Solutions:**
1. Increase patience: `--patience 10`
2. Adjust learning rate: `--lr 0.0005`
3. Check if validation set is too small
4. Review validation loss curve for actual convergence

### Checkpoint Loading Errors

**Symptoms:**
```
RuntimeError: Error loading checkpoint
```

**Solutions:**
1. Verify checkpoint file exists and is not corrupted
2. Ensure model architecture matches checkpoint
3. Check PyTorch version compatibility
4. Try training from scratch if checkpoint is incompatible

### GPU Not Detected

**Symptoms:**
```
GPU enabled: False
```

**Solutions:**
1. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```
3. Check PyTorch CUDA support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
4. Update GPU drivers

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch size too large | Reduce `--batch-size` |
| `No draft sequences loaded` | Missing or invalid CSV files | Check data files |
| `Card not found in encoder` | Missing card data | Verify JSON files |
| `RuntimeError: CUDA error` | GPU driver issue | Update drivers |
| `FileNotFoundError` | Wrong file path | Check directory structure |

## Best Practices

### For Production Models

1. **Run preprocessing first** to validate card data
2. **Train on multiple sets** for better generalization
3. **Use 30+ epochs** with early stopping
4. **Monitor validation metrics** closely
5. **Save training history** for analysis
6. **Test on held-out data** before deployment

### For Experimentation

1. **Start with small data** (`--limit 10000`)
2. **Use fewer epochs** (`--epochs 5`)
3. **Enable verbose logging** (`--verbose`)
4. **Save to separate directory** (`--output-dir`)

### For Resource-Constrained Environments

1. **Train on CPU** with `--no-gpu`
2. **Use smaller batches** (`--batch-size 8`)
3. **Reduce workers** (`--num-workers 2`)
4. **Train on single set** first
5. **Use data limits** for testing

## Additional Resources

- [17Lands Public Datasets](https://www.17lands.com/public_datasets)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Two-Tower Architecture Paper](https://arxiv.org/abs/1606.07792)
- [Main README](../README.md) - API documentation and setup

## Support

For issues or questions:
1. Check this troubleshooting guide
2. Review training logs for error details
3. Verify data and environment setup
4. Test with minimal configuration first
