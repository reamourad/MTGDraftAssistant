"""
Quick verification script for /status endpoint.
"""
from app.api import get_system_status
from pathlib import Path


def verify_status_endpoint():
    """Verify the status endpoint works correctly."""
    print("Testing /status endpoint implementation...")
    print("-" * 60)
    
    # Call the endpoint function directly
    result = get_system_status()
    
    print("\n✓ Status endpoint executed successfully")
    print("\nResponse structure:")
    print(f"  - systems: {list(result['systems'].keys())}")
    print(f"  - migration: {list(result['migration'].keys())}")
    
    # Check TensorFlow system
    tf_system = result["systems"]["tensorflow"]
    print(f"\nTensorFlow System:")
    print(f"  - status: {tf_system['status']}")
    print(f"  - endpoint: {tf_system['endpoint']}")
    
    # Check PyTorch system
    pytorch_system = result["systems"]["pytorch"]
    print(f"\nPyTorch System:")
    print(f"  - status: {pytorch_system['status']}")
    print(f"  - endpoint: {pytorch_system['endpoint']}")
    
    # Check if general model exists
    general_model_path = Path("app/models/general/pytorch_model.pt")
    model_exists = general_model_path.exists()
    
    print(f"\nGeneral Model Status:")
    print(f"  - Path: {general_model_path}")
    print(f"  - Exists: {model_exists}")
    print(f"  - Reported Status: {pytorch_system['status']}")
    
    # Verify correctness
    expected_status = "active" if model_exists else "not_ready"
    if pytorch_system['status'] == expected_status:
        print(f"\n✓ Status correctly reports '{expected_status}'")
    else:
        print(f"\n✗ Status mismatch! Expected '{expected_status}', got '{pytorch_system['status']}'")
        return False
    
    # Check migration phase
    migration = result["migration"]
    print(f"\nMigration Info:")
    print(f"  - Current Phase: {migration['current_phase']}")
    print(f"  - Next Phase: {migration['next_phase']}")
    print(f"  - Timeline: {migration['timeline']}")
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! Status endpoint is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = verify_status_endpoint()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
