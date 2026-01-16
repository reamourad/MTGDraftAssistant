"""
Configuration management for MTG Draft Assistant.

This module centralizes all configuration constants and settings,
removing hardcoded values from various modules.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Centralized configuration for MTG Draft Assistant.
    
    All configuration values should be defined here to avoid
    hardcoded values scattered throughout the codebase.
    """
    
    # Directory paths
    models_dir: str = "app/models"
    data_dir: str = "data"
    
    # API configuration
    api_title: str = "Lotus Draft Assistant API"
    api_description: str = "API for Magic: The Gathering draft pick predictions and booster generation"
    api_version: str = "1.0.0"
    cors_origins: list = field(default_factory=lambda: ["*"])
    
    # Model configuration
    sequence_length: int = 64
    embed_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 512
    dropout_rate: float = 0.2
    
    # Training configuration
    default_epochs: int = 50
    default_batch_size: int = 32
    default_validation_split: float = 0.2
    min_player_win_rate: float = 0.60
    early_stopping_patience: int = 10
    lr_reduction_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_learning_rate: float = 1e-7
    initial_learning_rate: float = 0.0001
    
    # PyTorch model configuration
    pytorch_embed_dim: int = 128
    pytorch_hidden_dim: int = 256
    card_encoding_dim: int = 407
    max_pool_size: int = 45
    max_pack_size: int = 15
    max_pick_number: int = 45
    
    # Cache configuration
    enable_model_cache: bool = True
    enable_data_cache: bool = True
    cache_ttl_seconds: Optional[int] = None  # None = no expiration
    max_cache_size_mb: Optional[int] = None  # None = unlimited
    
    # HTTP configuration
    icon_cache_max_age: int = 86400  # 1 day in seconds
    
    # File paths
    config_filename: str = "config.json"
    training_cards_filename: str = "training_cards.json"
    booster_config_filename: str = "booster_config.json"
    sheets_filename: str = "sheets.json"
    card_encodings_filename: str = "card_encodings.pkl"
    icon_filename: str = "icon.png"
    
    # Model file patterns
    keras_model_pattern: str = "{set_code}_model.keras"
    pytorch_model_pattern: str = "{set_code}_two_tower.pt"
    best_model_filename: str = "best_model.keras"
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Device configuration
    default_device: str = "cuda"  # "cuda" or "cpu"
    
    def get_model_dir(self, set_code: str) -> str:
        """
        Get the model directory path for a specific set.
        
        Args:
            set_code: Set code (e.g., 'MH3', 'BLB')
            
        Returns:
            Full path to the set's model directory
        """
        return os.path.join(self.models_dir, set_code.upper())
    
    def get_config_path(self, set_code: str) -> str:
        """Get path to config.json for a set."""
        return os.path.join(self.get_model_dir(set_code), self.config_filename)
    
    def get_training_cards_path(self, set_code: str) -> str:
        """Get path to training_cards.json for a set."""
        return os.path.join(self.get_model_dir(set_code), self.training_cards_filename)
    
    def get_booster_config_path(self, set_code: str) -> str:
        """Get path to booster_config.json for a set."""
        return os.path.join(self.get_model_dir(set_code), self.booster_config_filename)
    
    def get_sheets_path(self, set_code: str) -> str:
        """Get path to sheets.json for a set."""
        return os.path.join(self.get_model_dir(set_code), self.sheets_filename)
    
    def get_card_encodings_path(self, set_code: str) -> str:
        """Get path to card_encodings.pkl for a set."""
        return os.path.join(self.get_model_dir(set_code), self.card_encodings_filename)
    
    def get_icon_path(self, set_code: str) -> str:
        """Get path to icon.png for a set."""
        return os.path.join(self.get_model_dir(set_code), self.icon_filename)
    
    def get_keras_model_path(self, set_code: str) -> str:
        """Get path to Keras model file for a set."""
        filename = self.keras_model_pattern.format(set_code=set_code.lower())
        return os.path.join(self.get_model_dir(set_code), filename)
    
    def get_pytorch_model_path(self, set_code: str) -> str:
        """Get path to PyTorch model file for a set."""
        filename = self.pytorch_model_pattern.format(set_code=set_code.lower())
        return os.path.join(self.get_model_dir(set_code), filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'models_dir': self.models_dir,
            'data_dir': self.data_dir,
            'api_title': self.api_title,
            'api_description': self.api_description,
            'api_version': self.api_version,
            'cors_origins': self.cors_origins,
            'sequence_length': self.sequence_length,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'default_epochs': self.default_epochs,
            'default_batch_size': self.default_batch_size,
            'default_validation_split': self.default_validation_split,
            'min_player_win_rate': self.min_player_win_rate,
            'pytorch_embed_dim': self.pytorch_embed_dim,
            'pytorch_hidden_dim': self.pytorch_hidden_dim,
            'card_encoding_dim': self.card_encoding_dim,
            'enable_model_cache': self.enable_model_cache,
            'enable_data_cache': self.enable_data_cache,
            'log_level': self.log_level,
            'default_device': self.default_device
        }


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        The global Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def set_config(config: Config):
    """
    Set the global configuration instance.
    
    Args:
        config: New configuration instance
    """
    global _config_instance
    _config_instance = config


def reset_config():
    """Reset configuration to default values."""
    global _config_instance
    _config_instance = Config()
