#!/usr/bin/env python3
"""
Training script for PyTorch two-tower model.

This script provides a command-line interface for training the general-purpose
two-tower model on 17Lands draft data from one or more MTG sets.

Example usage:
    # Train on single set
    python scripts/train_pytorch.py --sets MH3 --epochs 20 --batch-size 32
    
    # Train on multiple sets (general model)
    python scripts/train_pytorch.py --sets MH3 BLB FIN --epochs 30 --lr 0.001
    
    # Resume from checkpoint
    python scripts/train_pytorch.py --sets MH3 --resume app/models/general/best_model.pt
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training import (
    DraftDataLoader,
    DraftDataset,
    TrainingConfig,
    TwoTowerTrainer,
    train_test_split
)
from app.ml.experimental.two_tower_model import TwoTowerModel


def setup_logging(output_dir: Path, verbose: bool = False):
    """
    Configure logging for training.
    
    Args:
        output_dir: Directory to save log file
        verbose: Whether to enable debug logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from some libraries
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    logging.info(f"Logging to: {log_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PyTorch two-tower model for MTG draft pick prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--sets',
        type=str,
        nargs='+',
        required=True,
        help='MTG set codes to train on (e.g., MH3 BLB FIN)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing set data folders'
    )
    parser.add_argument(
        '--min-win-rate',
        type=float,
        default=0.60,
        help='Minimum player win rate to include drafts'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of rows to load per set (for testing)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Fraction of data to use for training (rest for validation)'
    )
    
    # Model arguments
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension for towers'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='app/models/general',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # System arguments
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU training (use CPU only)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable draft sequence caching (always reload from CSV)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_card_encodings(sets: List[str]) -> dict:
    """
    Load pre-computed card encodings from all specified sets.
    
    Encodings are loaded from app/models/{SET}/card_encodings.pkl, which is created
    by the preprocess_cards.py script. This avoids re-encoding cards during training.
    
    Args:
        sets: List of set codes
    
    Returns:
        Dictionary mapping card UUIDs to their encodings and metadata
        Format: {uuid: {'encoding': np.array, 'name': str, 'uuid': str}}
    
    Raises:
        FileNotFoundError: If encodings not found for any set
    """
    logger = logging.getLogger(__name__)
    all_encodings = {}
    
    # Encodings are in app/models/{SET}/card_encodings.pkl (from preprocessing)
    models_dir = Path("app/models")
    
    for set_code in sets:
        encoding_file = models_dir / set_code / "card_encodings.pkl"
        
        if not encoding_file.exists():
            error_msg = (
                f"Card encodings not found for {set_code}: {encoding_file}\n"
                f"Please run preprocessing first: python preprocess_cards.py {set_code}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(encoding_file, 'rb') as f:
                encodings = pickle.load(f)
            
            # Merge encodings (UUID-keyed, so no duplicates)
            all_encodings.update(encodings)
            
            logger.info(f"Loaded {len(encodings)} card encodings from {set_code}")
        
        except Exception as e:
            logger.error(f"Failed to load encodings from {encoding_file}: {e}")
            raise
    
    if not all_encodings:
        raise ValueError("No card encodings loaded. Cannot proceed with training.")
    
    logger.info(f"Total unique card encodings loaded: {len(all_encodings)}")
    return all_encodings


def load_card_name_mappings(sets: List[str]) -> Dict[str, str]:
    """
    Load card name to UUID mappings from all specified sets.
    
    Args:
        sets: List of set codes
    
    Returns:
        Dictionary mapping card names to UUIDs
    
    Raises:
        FileNotFoundError: If mappings not found for any set
    """
    logger = logging.getLogger(__name__)
    all_mappings = {}
    
    models_dir = Path("app/models")
    
    for set_code in sets:
        mapping_file = models_dir / set_code / "card_name_to_uuid.json"
        
        if not mapping_file.exists():
            error_msg = (
                f"Card name mappings not found for {set_code}: {mapping_file}\n"
                f"Please run preprocessing first: python preprocess_cards.py {set_code}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(mapping_file, 'r') as f:
                mappings = json.load(f)
            
            # Merge mappings (card names should be unique across sets)
            all_mappings.update(mappings)
            
            logger.info(f"Loaded {len(mappings)} card name mappings from {set_code}")
        
        except Exception as e:
            logger.error(f"Failed to load mappings from {mapping_file}: {e}")
            raise
    
    logger.info(f"Total card name mappings loaded: {len(all_mappings)}")
    return all_mappings


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PyTorch Two-Tower Model Training")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info(f"Training sets: {', '.join(args.sets)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.lr}")
    logger.info(f"GPU enabled: {not args.no_gpu}")
    
    start_time = time.time()
    
    try:
        # Step 1: Load card data
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Loading card data")
        logger.info("=" * 80)
        
        card_encodings = load_card_encodings(args.sets)
        card_name_to_uuid = load_card_name_mappings(args.sets)
        
        if not card_encodings:
            logger.error("No card encodings loaded. Cannot proceed.")
            return 1
        
        # Step 2: Load draft data (with caching)
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Loading draft sequences")
        logger.info("=" * 80)
        
        # Create cache filename based on sets and filters
        sets_str = "_".join(sorted(args.sets))
        cache_file = output_dir / f"draft_sequences_{sets_str}.pkl"
        
        if cache_file.exists() and not args.no_cache:
            logger.info(f"Loading cached draft sequences from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    sequences = pickle.load(f)
                logger.info(f"Loaded {len(sequences)} cached draft sequences")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-loading from CSV...")
                sequences = None
        else:
            sequences = None
        
        if sequences is None:
            logger.info("Loading draft sequences from CSV files...")
            data_loader = DraftDataLoader(data_dir=args.data_dir)
            sequences = data_loader.load_multi_set_data(
                set_codes=args.sets,
                min_win_rate=args.min_win_rate,
                limit=args.limit
            )
            
            if not sequences:
                logger.error("No draft sequences loaded. Cannot proceed.")
                return 1
            
            logger.info(f"Loaded {len(sequences)} draft sequences")
            
            # Save to cache
            if not args.no_cache:
                logger.info(f"Saving draft sequences to cache: {cache_file}")
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(sequences, f)
                    logger.info("Cache saved successfully")
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
        
        # Step 3: Split data
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Splitting data into train/validation")
        logger.info("=" * 80)
        
        train_sequences, val_sequences = train_test_split(
            sequences,
            train_ratio=args.train_split,
            stratify_by_set=True,
            random_seed=42
        )
        
        logger.info(f"Train sequences: {len(train_sequences)}")
        logger.info(f"Validation sequences: {len(val_sequences)}")
        
        # Step 5: Create datasets
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Creating PyTorch datasets")
        logger.info("=" * 80)
        
        train_dataset = DraftDataset(
            draft_sequences=train_sequences,
            card_encodings=card_encodings,
            card_name_to_uuid=card_name_to_uuid,
            max_pool_size=45,
            max_pack_size=15
        )
        
        val_dataset = DraftDataset(
            draft_sequences=val_sequences,
            card_encodings=card_encodings,
            card_name_to_uuid=card_name_to_uuid,
            max_pool_size=45,
            max_pack_size=15
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Step 6: Create model
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Initializing model")
        logger.info("=" * 80)
        
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            model = TwoTowerModel.load_checkpoint(args.resume)
            logger.info("Model loaded from checkpoint")
        else:
            model = TwoTowerModel(
                card_dim=407,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim
            )
            logger.info("New model initialized")
        
        # Log model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Step 7: Create training configuration
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Configuring trainer")
        logger.info("=" * 80)
        
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            use_gpu=not args.no_gpu,
            num_workers=args.num_workers,
            checkpoint_dir=str(output_dir),
            log_interval=10,
            max_checkpoints=5
        )
        
        logger.info(f"Training configuration:\n{config}")
        
        # Step 8: Create trainer
        trainer = TwoTowerTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config
        )
        
        # Step 9: Train model
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: Starting training")
        logger.info("=" * 80)
        
        history = trainer.train()
        
        # Step 10: Save training history
        logger.info("\n" + "=" * 80)
        logger.info("Step 9: Saving training history")
        logger.info("=" * 80)
        
        history_path = output_dir / "training_history.json"
        trainer.save_training_history(history_path)
        
        # Log final results
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Total training time: {elapsed_time / 60:.2f} minutes")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Final top-1 accuracy: {history['val_top1_acc'][-1]:.4f}")
        logger.info(f"Final top-3 accuracy: {history['val_top3_acc'][-1]:.4f}")
        logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
        logger.info(f"Training history saved to: {history_path}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 1
    
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
