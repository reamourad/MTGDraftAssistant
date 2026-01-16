# Requirements Document

## Introduction

This specification defines the requirements for completing and training a general-purpose PyTorch two-tower model for MTG draft pick prediction. The model will be set-agnostic, capable of handling cards from any MTG set using a unified 407-dimensional card encoding. This represents a migration from the current set-specific TensorFlow models to a more flexible and scalable PyTorch architecture.

## Glossary

- **Two-Tower Model**: A neural network architecture with separate towers for encoding candidate cards and draft context, combined through a scoring head
- **Card Encoder**: Component that transforms MTG card data into 407-dimensional feature vectors
- **Candidate Tower**: Neural network that encodes individual card features into embeddings
- **Context Tower**: Neural network that encodes draft state (pool, pack, pick number) into embeddings
- **Scoring Head**: Neural network that combines candidate and context embeddings to produce pick scores
- **General Model**: A set-agnostic model trained on data from multiple MTG sets
- **17Lands**: Data source providing real player draft logs from Magic Arena
- **Training Pipeline**: End-to-end system for data loading, preprocessing, training, and validation

## Requirements

### Requirement 1

**User Story:** As a developer, I want to complete the missing PyTorch model components, so that the two-tower architecture is fully implemented and ready for training

#### Acceptance Criteria

1. WHEN the developer inspects the experimental PyTorch modules, THE System SHALL identify all missing component implementations (CandidateTower, ContextTower, ScoringHead)
2. THE System SHALL implement the CandidateTower module with input dimension of 407, hidden layers, and output embedding dimension of 128
3. THE System SHALL implement the ContextTower module that processes pool cards, pack cards, and pick number to produce 128-dimensional context embeddings
4. THE System SHALL implement the ScoringHead module that combines candidate and context embeddings to produce scalar pick scores
5. WHEN all components are implemented, THE System SHALL validate that the TwoTowerModel forward pass executes without errors

### Requirement 2

**User Story:** As a developer, I want to create a training data pipeline, so that I can load and preprocess 17Lands draft data for PyTorch model training

#### Acceptance Criteria

1. THE System SHALL create a data loader module that reads compressed 17Lands CSV files from the data directory
2. WHEN processing draft data, THE System SHALL extract draft sequences with pool state, pack options, and picked cards
3. THE System SHALL use the CardEncoder to convert card names to 407-dimensional feature vectors
4. THE System SHALL create PyTorch Dataset and DataLoader classes for efficient batch processing
5. THE System SHALL support loading data from multiple sets for general model training
6. WHEN data is loaded, THE System SHALL validate that batch shapes match model input requirements

### Requirement 3

**User Story:** As a developer, I want to implement the training loop, so that I can train the two-tower model on draft data

#### Acceptance Criteria

1. THE System SHALL implement a training script that accepts configuration parameters (epochs, batch size, learning rate, sets to train on)
2. THE System SHALL use an appropriate loss function for ranking/scoring tasks (e.g., cross-entropy, ranking loss)
3. THE System SHALL implement an optimizer (e.g., Adam) with configurable learning rate
4. WHEN training, THE System SHALL log training metrics (loss, accuracy) at regular intervals
5. THE System SHALL save model checkpoints at specified intervals or when validation performance improves
6. THE System SHALL support training on GPU when available with automatic fallback to CPU

### Requirement 4

**User Story:** As a developer, I want to implement validation and evaluation, so that I can assess model performance during and after training

#### Acceptance Criteria

1. THE System SHALL split data into training and validation sets
2. WHEN validation is performed, THE System SHALL compute pick accuracy (top-1, top-3, top-5)
3. THE System SHALL track validation metrics across epochs to monitor for overfitting
4. THE System SHALL implement early stopping when validation performance stops improving
5. THE System SHALL save the best model based on validation performance

### Requirement 5

**User Story:** As a developer, I want to integrate the trained PyTorch model into the API, so that users can make predictions using the general model

#### Acceptance Criteria

1. THE System SHALL create a prediction service that loads the trained PyTorch model
2. WHEN a prediction request is received, THE System SHALL encode the pool and pack cards using CardEncoder
3. THE System SHALL use the TwoTowerModel to score all cards in the pack
4. THE System SHALL return predictions sorted by score in descending order
5. THE System SHALL handle errors gracefully when cards cannot be encoded or model inference fails

### Requirement 6

**User Story:** As a developer, I want to create training documentation and scripts, so that future model training is reproducible and straightforward

#### Acceptance Criteria

1. THE System SHALL create a training script with clear command-line interface (e.g., train_pytorch.py --sets MH3 BLB --epochs 20)
2. THE System SHALL document all hyperparameters and their recommended values
3. THE System SHALL create a README or documentation explaining the training process
4. THE System SHALL document hardware requirements (GPU memory, disk space)
5. THE System SHALL provide example commands for common training scenarios
