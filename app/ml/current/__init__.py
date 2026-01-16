"""
LEGACY TENSORFLOW SYSTEM - PRODUCTION (CURRENT)

This directory contains the legacy TensorFlow/Keras-based ML system that is currently
running in production. This system will be gradually replaced by the PyTorch two-tower
architecture located in ml/experimental/.

Status: ACTIVE - Currently serving production traffic
Framework: TensorFlow/Keras
Migration Target: ml/experimental/ (PyTorch two-tower architecture)

DO NOT ADD NEW FEATURES TO THIS SYSTEM. All new development should target the
experimental PyTorch architecture. This system is maintained for backward compatibility
and will be deprecated once the PyTorch system is fully trained and validated.

Components:
- model_builder.py: Transformer-based model for draft pick prediction
- draft_data.py: Data loading and preprocessing for TensorFlow models

Transition Plan:
1. Keep this system running while PyTorch model is being trained
2. Run both systems in parallel during validation period
3. Gradually migrate traffic to PyTorch system
4. Deprecate this system once PyTorch is proven stable
"""