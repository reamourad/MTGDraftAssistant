"""
EXPERIMENTAL PYTORCH TWO-TOWER ARCHITECTURE - FUTURE SYSTEM

This directory contains the experimental PyTorch-based two-tower architecture that will
replace the legacy TensorFlow system in ml/current/.

Status: EXPERIMENTAL - Under development, not yet serving production traffic
Framework: PyTorch
Replaces: ml/current/ (TensorFlow/Keras legacy system)

🚀 FUTURE PRODUCTION SYSTEM 🚀
All new ML development should happen in this directory. This is the target architecture
for the MTG Draft Assistant going forward.

Architecture Overview:
- Two-Tower Model: Separate encoding for candidates (cards) and context (draft state)
- CandidateTower: Encodes individual card features into embeddings
- ContextTower: Encodes draft context (current pool, pack state) into embeddings
- ScoringHead: Combines candidate and context embeddings to predict pick probability

Components:
- two_tower_model.py: Main model integrating all components
- candidate_tower.py: Card feature encoding tower
- context_tower.py: Draft context encoding tower
- scoring_head.py: Final scoring layer for pick prediction
- card_encoder.py: Card feature extraction (407-dim vectors)
- model_loader.py: PyTorch model checkpoint loading

Transition Plan:
1. Complete model training with PyTorch architecture
2. Validate model performance against TensorFlow baseline
3. Deploy alongside TensorFlow system for A/B testing
4. Gradually migrate traffic from TensorFlow to PyTorch
5. Deprecate TensorFlow system once PyTorch is proven stable

Current Status:
- Model architecture: ✅ Complete
- Training pipeline: 🚧 In progress
- Production deployment: ⏳ Pending training completion
"""