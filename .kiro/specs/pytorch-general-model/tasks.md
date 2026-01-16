# Implementation Plan

- [x] 1. Implement neural network components





  - Create the three core PyTorch modules that form the two-tower architecture
  - Ensure proper initialization, forward passes, and output dimensions
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.1 Implement CandidateTower module


  - Create `app/ml/experimental/candidate_tower.py`
  - Implement 3-layer feedforward network: 407 → 256 → 256 → 128
  - Add ReLU activations and dropout (0.2) between layers
  - Verify output shape is (batch, 128) for input (batch, 407)
  - _Requirements: 1.2_

- [x] 1.2 Implement ContextTower module

  - Create `app/ml/experimental/context_tower.py`
  - Implement mean pooling for pool and pack cards
  - Add pick number embedding layer (45 picks → 16 dims)
  - Concatenate features (407 + 407 + 16 = 830) and process through 3-layer network
  - Verify output shape is (batch, 128) for variable-length inputs
  - _Requirements: 1.3_

- [x] 1.3 Implement ScoringHead module

  - Create `app/ml/experimental/scoring_head.py`
  - Implement 3-layer network that combines candidate and context embeddings
  - Process concatenated embeddings (256) → 128 → 64 → 1
  - Add ReLU and dropout between layers
  - Verify output shape is (batch, 1) for inputs (batch, 128) each
  - _Requirements: 1.4_

- [x] 1.4 Verify TwoTowerModel integration


  - Test that TwoTowerModel forward pass works with new components
  - Create simple test script that instantiates model and runs dummy data through it
  - Verify gradient flow through all components
  - Check that model.save_checkpoint() and load_checkpoint() work correctly
  - _Requirements: 1.5_

- [x] 2. Create training data infrastructure





  - Build the data loading and preprocessing pipeline for 17Lands CSV files
  - Convert draft logs into PyTorch-compatible format
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 2.1 Implement DraftSequence data model


  - Create `app/training/__init__.py` with DraftSequence dataclass
  - Define fields: draft_id, pick_number, pool (List[str]), pack (List[str]), picked_card
  - Add validation method to ensure picked_card is in pack
  - _Requirements: 2.2_

- [x] 2.2 Implement DraftDataLoader for CSV processing


  - Create `app/training/data_loader.py` with DraftDataLoader class
  - Implement load_set_data() to read compressed CSV files from data/{SET}/ directory
  - Filter drafts by win rate (default: >= 60%)
  - Parse draft columns (pool_*, pack_*, pick) into DraftSequence objects
  - Implement load_multi_set_data() for training on multiple sets
  - _Requirements: 2.1, 2.5_

- [x] 2.3 Implement DraftDataset for PyTorch


  - Create `app/training/dataset.py` with DraftDataset class
  - Implement __getitem__ to return dict with pool_cards, pack_cards, pick_number, target_idx
  - Use CardEncoder to convert card names to 407-dim vectors
  - Implement padding for variable-length pools and packs
  - Handle edge cases (empty pool, single card in pack)
  - _Requirements: 2.3, 2.4, 2.6_

- [x] 2.4 Create data splitting utilities


  - Add train_test_split() function to dataset.py
  - Implement stratified splitting by set code for multi-set training
  - Add validation split (default: 80/20 train/val)
  - _Requirements: 4.1_

- [x] 3. Implement training loop and optimization





  - Build the core training infrastructure with loss functions, optimizers, and checkpointing
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3.1 Create TrainingConfig dataclass


  - Create `app/training/config.py` with TrainingConfig
  - Define hyperparameters: epochs, batch_size, learning_rate, patience, use_gpu, num_workers
  - Add checkpoint_dir and logging configuration
  - Implement from_dict() and to_dict() methods for serialization
  - _Requirements: 3.1_

- [x] 3.2 Implement TwoTowerTrainer class


  - Create `app/training/trainer.py` with TwoTowerTrainer class
  - Initialize model, dataloaders, optimizer (Adam), and loss function (CrossEntropyLoss)
  - Set up device (GPU/CPU) based on config
  - Implement train() method that runs full training loop
  - _Requirements: 3.1, 3.2, 3.3, 3.6_

- [x] 3.3 Implement training epoch logic

  - Add _train_epoch() method to TwoTowerTrainer
  - Iterate through training batches
  - Compute forward pass, loss, backward pass, and optimizer step
  - Log batch losses and progress
  - Return average epoch loss
  - _Requirements: 3.3, 3.4_

- [x] 3.4 Implement checkpoint saving


  - Add _save_checkpoint() method to TwoTowerTrainer
  - Save model state, optimizer state, epoch, and metrics
  - Use TwoTowerModel.save_checkpoint() with metadata
  - Save to config.checkpoint_dir with epoch number in filename
  - Keep only best N checkpoints to save disk space
  - _Requirements: 3.5_


- [x] 4. Implement validation and evaluation




  - Add validation logic to assess model performance during training
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Create ModelEvaluator class


  - Create `app/training/evaluator.py` with ModelEvaluator class
  - Initialize with model and validation dataloader
  - Implement evaluate() method that computes metrics without gradients
  - _Requirements: 4.1, 4.2_

- [x] 4.2 Implement accuracy metrics

  - Add compute_top_k_accuracy() method to ModelEvaluator
  - Calculate top-1, top-3, and top-5 pick accuracy
  - Handle edge cases (pack size < k)
  - Return metrics as dictionary
  - _Requirements: 4.2_

- [x] 4.3 Integrate validation into training loop


  - Add _validate() method to TwoTowerTrainer
  - Call ModelEvaluator.evaluate() after each epoch
  - Log validation loss and accuracy metrics
  - Return validation metrics dictionary
  - _Requirements: 4.1, 4.3_

- [x] 4.4 Implement early stopping


  - Track best validation loss in TwoTowerTrainer
  - Increment patience counter when validation doesn't improve
  - Stop training when patience counter reaches config.patience
  - Log early stopping event
  - _Requirements: 4.4, 4.5_

- [x] 5. Create training CLI script





  - Build command-line interface for training models
  - _Requirements: 6.1, 6.2_

- [x] 5.1 Implement train_pytorch.py script


  - Create `scripts/train_pytorch.py` with argparse CLI
  - Add arguments: --sets (list), --epochs, --batch-size, --lr, --output-dir
  - Load data using DraftDataLoader for specified sets
  - Create CardEncoder with combined card data from all sets
  - Initialize DraftDataset for train and validation splits
  - Create TwoTowerModel and TwoTowerTrainer
  - Run training and save final model
  - _Requirements: 6.1_


- [x] 5.2 Add training progress logging

  - Integrate Python logging in train_pytorch.py
  - Log training start with configuration summary
  - Log epoch progress (loss, metrics, time)
  - Log checkpoint saves and early stopping
  - Save training history to JSON file
  - _Requirements: 3.4, 6.1_

- [x] 6. Integrate PyTorch model into API





  - Connect trained model to existing API infrastructure
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6.1 Implement PyTorchPredictionService


  - Create `app/core/pytorch_prediction.py` with PyTorchPredictionService class
  - Initialize with PyTorchModelLoader and CardEncoder
  - Implement predict_picks() method that encodes cards and runs inference
  - Convert model scores to probabilities using sigmoid
  - Return sorted list of CardPrediction objects
  - Handle errors gracefully (missing cards, model failures)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_



- [x] 6.2 Update API endpoint for PyTorch predictions





  - Modify `app/api/main.py` to add /predict_pytorch endpoint
  - Use PyTorchPredictionService for inference
  - Return predictions in same format as existing /predict endpoint
  - Add model_type: "pytorch" to response


  - _Requirements: 5.1_

- [x] 6.3 Update /status endpoint








  - Modify /status endpoint in app/api.py to show PyTorch model status
  - Check if general PyTorch model exists using PyTorchModelLoader.is_model_available()
  - Update pytorch.status from "not_ready" to "active" when model is trained
  - _Requirements: 5.1_



- [x] 7. Create training documentation






  - Document the training process for future reference
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 7.1 Create TRAINING.md documentation


  - Create `docs/TRAINING.md` with comprehensive training guide
  - Document data preparation (downloading 17Lands CSVs)
  - Explain training command with all arguments
  - Document hyperparameter recommendations
  - Add troubleshooting section for common issues
  - _Requirements: 6.3, 6.5_

- [x] 7.2 Document hardware requirements

  - Add section to TRAINING.md on hardware requirements
  - Specify GPU memory requirements (recommended: 8GB+ VRAM)
  - Document training time estimates for different configurations
  - Provide CPU-only training guidance
  - _Requirements: 6.4_

- [x] 7.3 Add example training commands

  - Add examples section to TRAINING.md
  - Provide command for single-set training
  - Provide command for multi-set general model training
  - Show how to resume from checkpoint
  - Document how to evaluate trained model
  - _Requirements: 6.5_

- [x] 7.4 Update main README.md


  - Add section on PyTorch two-tower model to README.md
  - Link to TRAINING.md for detailed instructions
  - Update architecture diagram to show both TensorFlow and PyTorch systems
  - Document API endpoints for both model types
  - _Requirements: 6.3_
