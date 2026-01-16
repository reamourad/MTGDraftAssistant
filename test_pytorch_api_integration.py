"""
Test PyTorch API Integration

This test verifies that the PyTorch prediction service is properly integrated
into the API endpoints.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.core.pytorch_prediction import PyTorchPredictionService, CardPrediction, PyTorchPredictionError
from app.ml.experimental.model_loader import PyTorchModelLoader, ModelLoadError
from app.ml.experimental.card_encoder import CardEncoder


def test_pytorch_prediction_service_initialization():
    """Test that PyTorchPredictionService can be initialized."""
    mock_loader = Mock(spec=PyTorchModelLoader)
    mock_encoder = Mock(spec=CardEncoder)
    
    service = PyTorchPredictionService(mock_loader, mock_encoder)
    
    assert service.model_loader == mock_loader
    assert service.encoder == mock_encoder


def test_predict_picks_with_empty_pack():
    """Test that predict_picks raises error with empty pack."""
    mock_loader = Mock(spec=PyTorchModelLoader)
    mock_encoder = Mock(spec=CardEncoder)
    
    service = PyTorchPredictionService(mock_loader, mock_encoder)
    
    with pytest.raises(PyTorchPredictionError, match="Pack cannot be empty"):
        service.predict_picks("MH3", [], [])


def test_predict_picks_model_not_available():
    """Test that predict_picks raises error when model is not available."""
    mock_loader = Mock(spec=PyTorchModelLoader)
    mock_loader.load_model.side_effect = ModelLoadError("Model not found")
    mock_encoder = Mock(spec=CardEncoder)
    
    service = PyTorchPredictionService(mock_loader, mock_encoder)
    
    with pytest.raises(PyTorchPredictionError, match="General PyTorch model not available"):
        service.predict_picks("MH3", ["Card1"], ["Card2", "Card3"])


def test_is_model_available():
    """Test is_model_available method."""
    mock_loader = Mock(spec=PyTorchModelLoader)
    mock_loader.is_model_available.return_value = True
    mock_encoder = Mock(spec=CardEncoder)
    
    service = PyTorchPredictionService(mock_loader, mock_encoder)
    
    assert service.is_model_available("general") is True
    mock_loader.is_model_available.assert_called_once_with("general")


def test_card_prediction_dataclass():
    """Test CardPrediction dataclass."""
    pred = CardPrediction(card_name="Lightning Bolt", probability=0.95)
    
    assert pred.card_name == "Lightning Bolt"
    assert pred.probability == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
