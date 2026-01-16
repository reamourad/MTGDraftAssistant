"""Test training infrastructure components."""

import torch
import numpy as np
from app.training.config import TrainingConfig
from app.training.trainer import TwoTowerTrainer
from app.training.dataset import DraftDataset
from app.training import DraftSequence
from app.ml.experimental.two_tower_model import TwoTowerModel
from app.ml.experimental.card_encoder import CardEncoder


def test_training_config():
    """Test TrainingConfig serialization."""
    print("Testing TrainingConfig...")
    
    # Create config
    config = TrainingConfig(
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
        patience=3
    )
    
    # Test to_dict
    config_dict = config.to_dict()
    assert config_dict['epochs'] == 10
    assert config_dict['batch_size'] == 16
    print(f"  to_dict: {config_dict}")
    
    # Test from_dict
    config2 = TrainingConfig.from_dict(config_dict)
    assert config2.epochs == 10
    assert config2.batch_size == 16
    print(f"  from_dict: OK")
    
    # Test with extra keys (should be filtered)
    config_dict_extra = {**config_dict, 'extra_key': 'value'}
    config3 = TrainingConfig.from_dict(config_dict_extra)
    assert config3.epochs == 10
    print(f"  from_dict with extra keys: OK")
    
    print("✓ TrainingConfig tests passed\n")


def test_trainer_initialization():
    """Test TwoTowerTrainer initialization."""
    print("Testing TwoTowerTrainer initialization...")
    
    # Create mock card data
    mock_cards = [
        {
            "name": "Ajani, Nacatl Pariah",
            "rarity": "mythic",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Creature"],
            "subtypes": ["Cat", "Warrior"],
            "power": 2.0,
            "toughness": 2.0,
            "can_attack": True,
            "keywords": [],
            "oracle_text": "Test card"
        },
        {
            "name": "Arid Mesa",
            "rarity": "rare",
            "colors": [],
            "mana_cost": "",
            "converted_mana_cost": 0.0,
            "types": ["Land"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test land"
        },
        {
            "name": "Flare of Fortitude",
            "rarity": "rare",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Instant"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test instant"
        }
    ]
    
    card_encoder = CardEncoder(card_list=mock_cards)
    
    # Create minimal draft sequences
    sequences = [
        DraftSequence(
            draft_id="test_1",
            pick_number=1,
            pool=[],
            pack=["Ajani, Nacatl Pariah", "Arid Mesa"],
            picked_card="Ajani, Nacatl Pariah"
        ),
        DraftSequence(
            draft_id="test_2",
            pick_number=2,
            pool=["Ajani, Nacatl Pariah"],
            pack=["Arid Mesa", "Flare of Fortitude"],
            picked_card="Arid Mesa"
        )
    ]
    
    # Create datasets
    train_dataset = DraftDataset(sequences, card_encoder)
    val_dataset = DraftDataset(sequences, card_encoder)
    
    # Create model
    model = TwoTowerModel()
    
    # Create config
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        num_workers=0,  # Avoid multiprocessing issues in test
        use_gpu=False
    )
    
    # Create trainer
    trainer = TwoTowerTrainer(model, train_dataset, val_dataset, config)
    
    assert trainer.device == torch.device('cpu')
    assert trainer.current_epoch == 0
    assert trainer.best_val_loss == float('inf')
    print(f"  Device: {trainer.device}")
    print(f"  Train batches: {len(trainer.train_loader)}")
    print(f"  Val batches: {len(trainer.val_loader)}")
    
    print("✓ TwoTowerTrainer initialization tests passed\n")


def test_training_epoch():
    """Test single training epoch."""
    print("Testing training epoch...")
    
    # Create mock card data
    mock_cards = [
        {
            "name": "Ajani, Nacatl Pariah",
            "rarity": "mythic",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Creature"],
            "subtypes": ["Cat", "Warrior"],
            "power": 2.0,
            "toughness": 2.0,
            "can_attack": True,
            "keywords": [],
            "oracle_text": "Test card"
        },
        {
            "name": "Arid Mesa",
            "rarity": "rare",
            "colors": [],
            "mana_cost": "",
            "converted_mana_cost": 0.0,
            "types": ["Land"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test land"
        },
        {
            "name": "Flare of Fortitude",
            "rarity": "rare",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Instant"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test instant"
        }
    ]
    
    card_encoder = CardEncoder(card_list=mock_cards)
    
    # Create minimal draft sequences
    sequences = [
        DraftSequence(
            draft_id="test_1",
            pick_number=1,
            pool=[],
            pack=["Ajani, Nacatl Pariah", "Arid Mesa"],
            picked_card="Ajani, Nacatl Pariah"
        ),
        DraftSequence(
            draft_id="test_2",
            pick_number=2,
            pool=["Ajani, Nacatl Pariah"],
            pack=["Arid Mesa", "Flare of Fortitude"],
            picked_card="Arid Mesa"
        )
    ]
    
    # Create datasets
    train_dataset = DraftDataset(sequences, card_encoder)
    val_dataset = DraftDataset(sequences, card_encoder)
    
    # Create model
    model = TwoTowerModel()
    
    # Create config
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        num_workers=0,
        use_gpu=False,
        log_interval=1
    )
    
    # Create trainer
    trainer = TwoTowerTrainer(model, train_dataset, val_dataset, config)
    
    # Run one epoch
    train_loss = trainer._train_epoch()
    
    assert isinstance(train_loss, float)
    assert train_loss > 0
    print(f"  Train loss: {train_loss:.4f}")
    
    # Run validation
    val_metrics = trainer._validate()
    
    assert isinstance(val_metrics, dict)
    assert 'loss' in val_metrics
    assert 'top1_acc' in val_metrics
    assert 'top3_acc' in val_metrics
    assert 'top5_acc' in val_metrics
    assert val_metrics['loss'] > 0
    print(f"  Val loss: {val_metrics['loss']:.4f}")
    print(f"  Val top-1 acc: {val_metrics['top1_acc']:.4f}")
    print(f"  Val top-3 acc: {val_metrics['top3_acc']:.4f}")
    
    print("✓ Training epoch tests passed\n")


def test_checkpoint_saving():
    """Test checkpoint saving."""
    print("Testing checkpoint saving...")
    
    # Create mock card data
    mock_cards = [
        {
            "name": "Ajani, Nacatl Pariah",
            "rarity": "mythic",
            "colors": ["W"],
            "mana_cost": "{1}{W}",
            "converted_mana_cost": 2.0,
            "types": ["Creature"],
            "subtypes": ["Cat", "Warrior"],
            "power": 2.0,
            "toughness": 2.0,
            "can_attack": True,
            "keywords": [],
            "oracle_text": "Test card"
        },
        {
            "name": "Arid Mesa",
            "rarity": "rare",
            "colors": [],
            "mana_cost": "",
            "converted_mana_cost": 0.0,
            "types": ["Land"],
            "subtypes": [],
            "power": 0.0,
            "toughness": 0.0,
            "can_attack": False,
            "keywords": [],
            "oracle_text": "Test land"
        }
    ]
    
    card_encoder = CardEncoder(card_list=mock_cards)
    
    sequences = [
        DraftSequence(
            draft_id="test_1",
            pick_number=1,
            pool=[],
            pack=["Ajani, Nacatl Pariah", "Arid Mesa"],
            picked_card="Ajani, Nacatl Pariah"
        )
    ]
    
    # Create datasets
    train_dataset = DraftDataset(sequences, card_encoder)
    val_dataset = DraftDataset(sequences, card_encoder)
    
    # Create model
    model = TwoTowerModel()
    
    # Create config with test checkpoint dir
    config = TrainingConfig(
        epochs=1,
        batch_size=1,
        num_workers=0,
        use_gpu=False,
        checkpoint_dir="test_checkpoints"
    )
    
    # Create trainer
    trainer = TwoTowerTrainer(model, train_dataset, val_dataset, config)
    
    # Save checkpoint
    metrics = {'train_loss': 1.5, 'val_loss': 1.6}
    trainer._save_checkpoint(0, metrics)
    
    # Check files exist
    checkpoint_path = trainer.checkpoint_dir / "checkpoint_epoch_1.pt"
    best_path = trainer.checkpoint_dir / "best_model.pt"
    
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    assert best_path.exists(), f"Best model not found: {best_path}"
    print(f"  Checkpoint saved: {checkpoint_path}")
    print(f"  Best model saved: {best_path}")
    
    # Load checkpoint to verify
    loaded_model = TwoTowerModel.load_checkpoint(str(checkpoint_path))
    assert loaded_model is not None
    print(f"  Checkpoint loaded successfully")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_checkpoints")
    print(f"  Cleaned up test checkpoints")
    
    print("✓ Checkpoint saving tests passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Training Infrastructure")
    print("=" * 60 + "\n")
    
    test_training_config()
    test_trainer_initialization()
    test_training_epoch()
    test_checkpoint_saving()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
