"""
Test for /status endpoint PyTorch model status reporting.
"""
import pytest
from fastapi.testclient import TestClient
from app.api import app
from pathlib import Path
import os


client = TestClient(app)


def test_status_endpoint_structure():
    """Test that /status endpoint returns correct structure."""
    response = client.get("/status")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "systems" in data
    assert "tensorflow" in data["systems"]
    assert "pytorch" in data["systems"]
    assert "migration" in data
    
    # Check TensorFlow system
    tf_system = data["systems"]["tensorflow"]
    assert tf_system["status"] == "active"
    assert tf_system["endpoint"] == "/predict"
    
    # Check PyTorch system
    pytorch_system = data["systems"]["pytorch"]
    assert "status" in pytorch_system
    assert pytorch_system["status"] in ["not_ready", "active"]
    assert pytorch_system["endpoint"] == "/predict_pytorch"


def test_status_pytorch_not_ready_when_no_model():
    """Test that PyTorch status is 'not_ready' when general model doesn't exist."""
    # Ensure general model doesn't exist
    general_model_path = Path("app/models/general/pytorch_model.pt")
    
    if general_model_path.exists():
        pytest.skip("General model exists, cannot test not_ready state")
    
    response = client.get("/status")
    assert response.status_code == 200
    
    data = response.json()
    pytorch_status = data["systems"]["pytorch"]["status"]
    
    assert pytorch_status == "not_ready"
    
    # Check migration phase
    assert "Phase 1" in data["migration"]["current_phase"]


def test_status_pytorch_active_when_model_exists():
    """Test that PyTorch status is 'active' when general model exists."""
    general_model_path = Path("app/models/general/pytorch_model.pt")
    
    if not general_model_path.exists():
        pytest.skip("General model doesn't exist, cannot test active state")
    
    response = client.get("/status")
    assert response.status_code == 200
    
    data = response.json()
    pytorch_status = data["systems"]["pytorch"]["status"]
    
    assert pytorch_status == "active"
    
    # Check migration phase
    assert "Phase 2" in data["migration"]["current_phase"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
