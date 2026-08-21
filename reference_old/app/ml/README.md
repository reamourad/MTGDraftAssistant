# Machine Learning Layer

This directory contains the ML components for the MTG Draft Assistant. Currently, we maintain **two parallel systems** during the transition from TensorFlow to PyTorch.

## Directory Structure

```
ml/
├── current/          # Legacy TensorFlow system (PRODUCTION)
│   ├── model_builder.py
│   └── draft_data.py
└── experimental/     # Future PyTorch system (EXPERIMENTAL)
    ├── two_tower_model.py
    ├── candidate_tower.py
    ├── context_tower.py
    ├── scoring_head.py
    ├── card_encoder.py
    └── model_loader.py
```

## Current System (TensorFlow/Keras) - LEGACY

**Status:** ✅ ACTIVE - Currently serving production traffic

**Location:** `ml/current/`

**Framework:** TensorFlow/Keras

**Architecture:** Transformer-based sequence model

The current system uses a transformer architecture to predict draft picks based on the current pool and available pack. This system is **stable and production-ready** but is being phased out in favor of the PyTorch two-tower architecture.

### Components:
- `model_builder.py`: Transformer model with multi-head attention
- `draft_data.py`: Data loading and preprocessing

### ⚠️ Deprecation Notice
**Do NOT add new features to this system.** It is maintained for backward compatibility only. All new development should target the experimental PyTorch system.

## Experimental System (PyTorch) - FUTURE

**Status:** 🚧 EXPERIMENTAL - Under development

**Location:** `ml/experimental/`

**Framework:** PyTorch

**Architecture:** Two-tower retrieval model

The experimental system uses a modern two-tower architecture that separately encodes card features (candidate tower) and draft context (context tower), then combines them for scoring. This architecture is more scalable and maintainable than the legacy transformer approach.

### Components:
- `two_tower_model.py`: Main model integrating all components
- `candidate_tower.py`: Encodes card features into embeddings
- `context_tower.py`: Encodes draft context into embeddings
- `scoring_head.py`: Combines embeddings to predict pick probability
- `card_encoder.py`: Extracts 407-dimensional card feature vectors
- `model_loader.py`: Handles PyTorch model checkpoint loading

### 🚀 Future Production System
This is the target architecture going forward. All new ML development should happen here.

## Transition Plan

### Phase 1: Parallel Development (Current)
- ✅ Legacy TensorFlow system serves production traffic
- 🚧 PyTorch system architecture complete, training in progress
- Both systems coexist in the codebase

### Phase 2: Training & Validation
- ⏳ Complete PyTorch model training
- ⏳ Validate PyTorch model performance vs TensorFlow baseline
- ⏳ Ensure PyTorch model meets quality thresholds

### Phase 3: Parallel Deployment
- ⏳ Deploy PyTorch system alongside TensorFlow
- ⏳ A/B test both systems with production traffic
- ⏳ Monitor performance, latency, and accuracy

### Phase 4: Migration
- ⏳ Gradually shift traffic from TensorFlow to PyTorch
- ⏳ Monitor for issues and rollback capability maintained
- ⏳ Full cutover once PyTorch proven stable

### Phase 5: Deprecation
- ⏳ Remove TensorFlow system from codebase
- ⏳ Clean up legacy dependencies
- ⏳ Update documentation

## API Endpoints

During the transition, both systems are accessible:

- `/predict` - Legacy TensorFlow system (current production)
- `/predict_v2` - Experimental PyTorch system (when ready)

## Development Guidelines

### For Legacy System (ml/current/)
- ❌ Do NOT add new features
- ✅ Bug fixes only for critical issues
- ✅ Maintain for production stability
- ✅ Keep running until PyTorch is validated

### For Experimental System (ml/experimental/)
- ✅ All new ML development goes here
- ✅ Focus on training pipeline completion
- ✅ Optimize for production deployment
- ✅ Document architecture decisions

## Dependencies

Both systems require their respective frameworks:

```
# TensorFlow (Legacy)
tensorflow>=2.10.0
keras>=2.10.0

# PyTorch (Future)
torch>=2.0.0
```

Both are maintained in `requirements.txt` during the transition period.

## Questions?

For questions about:
- **Legacy system:** Review existing TensorFlow code and documentation
- **Future system:** Check PyTorch architecture docs and design documents
- **Migration timeline:** Consult the project roadmap and spec documents
