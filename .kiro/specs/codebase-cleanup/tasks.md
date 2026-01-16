# Implementation Plan

- [x] 1. Organize framework components and create clear separation





  - Create ml/current/ directory for TensorFlow-based production system
  - Create ml/experimental/ directory for PyTorch two-tower architecture
  - Move PyTorch components (CandidateTower.py, ContextTower.py, ScoringHead.py) to experimental/
  - Keep both TensorFlow and PyTorch dependencies in requirements.txt with clear documentation
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Create new layered directory structure





  - Create new directory structure: api/, core/, ml/, data/, services/, utils/
  - Focus on ml/experimental/ for PyTorch two-tower architecture
  - Add __init__.py files to all new directories
  - Create placeholder files for main components in each layer
  - _Requirements: 2.1, 3.2, 6.3_

- [x] 3. Extract and refactor API layer




  - [x] 3.1 Create app/api/main.py with FastAPI application and routing logic


    - Move FastAPI app initialization and CORS setup from api.py
    - Extract route handlers into clean, focused functions
    - _Requirements: 2.2, 4.1_
  
  - [x] 3.2 Create app/api/models.py with Pydantic request/response models


    - Define PredictRequest, PredictResponse, BoosterResponse, SetsResponse models
    - Add proper validation and type hints
    - _Requirements: 2.2, 4.4_
  
  - [x] 3.3 Create app/api/dependencies.py for dependency injection


    - Set up dependency injection for services
    - Remove global caches from API layer
    - _Requirements: 4.2, 2.5_

- [x] 4. Create core business logic layer (skeleton for future PyTorch model)




  - [x] 4.1 Create app/core/prediction.py skeleton for PyTorch prediction orchestration



    - Create PredictionService skeleton with interface for 2-tower architecture
    - Define methods for card encoding, context embedding, and candidate scoring
    - Add comprehensive docstrings for future implementation
    - Note: Implementation will be added when PyTorch model is trained
    - _Requirements: 2.3, 4.1, 4.3_
  
  - [x] 4.2 Create app/core/booster.py for booster generation


    - Move booster generation logic from app/booster/generator.py
    - Create BoosterService class with clean interface
    - _Requirements: 2.3, 4.1_

- [x] 5. Refactor machine learning layer for PyTorch two-tower architecture






  - [x] 5.1 Create app/ml/experimental/two_tower_model.py for complete model integration

    - Integrate CandidateTower, ContextTower, and ScoringHead into unified PyTorch model
    - Implement forward pass that combines all three components
    - Add model save/load functionality
    - Note: Training pipeline will be implemented separately
    - _Requirements: 2.4, 4.1_
  
  - [x] 5.2 Create app/ml/experimental/model_loader.py for PyTorch model loading


    - Implement PyTorch model checkpoint loading with proper error handling
    - Handle GPU device allocation and model.eval() mode
    - Add caching for loaded models
    - _Requirements: 2.4, 4.1_
  
  - [x] 5.3 Enhance CardEncoder with proper interfaces


    - Add batch encoding support for efficiency
    - Add proper error handling and validation
    - Ensure output format matches expected 407-dim vectors
    - _Requirements: 2.4, 3.2_

- [ ] 6. Refactor data layer for PyTorch architecture





  - [x] 6.1 Create app/data/repositories.py for data access patterns


    - Create DataRepository class for centralized data access
    - Implement methods for loading set data, card encodings, booster configs
    - Focus on data formats compatible with PyTorch tensors
    - _Requirements: 2.5, 4.1, 4.3_
  
  - [x] 6.2 Create app/data/card_data.py for card encoding and processing


    - Create unified interface for card data processing
    - Integrate with CardEncoder for consistent data flow
    - _Requirements: 2.5, 3.1, 3.3_

- [x] 7. Create services layer for external integrations




  - [x] 7.1 Create app/services/mtgjson.py for MTGJson API integration


    - Move MTGJson fetching logic from app/booster/mtgjson_fetcher.py
    - Create clean service interface for external API calls
    - _Requirements: 2.5, 4.1_

- [x] 8. Create utilities layer





  - [x] 8.1 Create app/utils/config.py for configuration management


    - Centralize configuration constants and settings
    - Remove hardcoded values from various modules
    - _Requirements: 3.3, 4.4_
  
  - [x] 8.2 Create app/utils/cache.py for caching utilities


    - Extract caching logic into reusable utilities
    - Implement proper cache management and cleanup
    - _Requirements: 4.2_
  
  - [x] 8.3 Create app/utils/exceptions.py for centralized error handling


    - Define domain-specific exception classes
    - Implement consistent error handling patterns
    - _Requirements: 4.4_

- [x] 9. Optimize JSON file usage and data persistence




  - [x] 9.1 Audit and consolidate JSON files in preprocessing


    - Modify preprocess_cards.py to generate only essential JSON files
    - Remove or consolidate cards.json with other data files
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 9.2 Update data loading to work with optimized file structure


    - Modify data repositories to work with consolidated JSON files
    - Ensure backward compatibility during transition
    - _Requirements: 7.4, 7.5_

- [x] 10. Update main application entry points for PyTorch architecture





  - [x] 10.1 Create compatibility layer in api.py for gradual migration


    - Keep existing TensorFlow /predict endpoint working
    - Add new /predict_v2 endpoint for PyTorch 2-tower model (when ready)
    - Document migration path from TensorFlow to PyTorch
    - _Requirements: 6.1, 6.4_
  
  - [x] 10.2 Update preprocess_cards.py to generate PyTorch-compatible data


    - Ensure CardEncoder can process the generated card data
    - Generate card data in format expected by 2-tower model (407-dim features)
    - Keep backward compatibility with existing TensorFlow data
    - _Requirements: 6.1, 6.5_

- [x] 11. Clean up old files and prepare for PyTorch migration





  - [x] 11.1 Mark TensorFlow files as legacy (keep for now)


    - Add comments marking ml/current/ as legacy TensorFlow system
    - Document that ml/experimental/ is the future PyTorch system
    - Keep both systems running during transition period
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 11.2 Update all import statements to use new module structure


    - Fix imports in all files to use new layered structure (api/, core/, ml/, etc.)
    - Ensure consistent import patterns across the codebase
    - Update references to point to correct module locations
    - _Requirements: 3.3, 6.5_

- [ ]* 12. Add comprehensive testing for PyTorch architecture
  - [ ]* 12.1 Create unit tests for PyTorch 2-tower components
    - Write unit tests for CandidateTower, ContextTower, ScoringHead
    - Test CardEncoder output dimensions (407-dim vectors)
    - Test tensor shapes through the entire pipeline
    - Mock GPU operations for consistent testing
    - _Requirements: 4.4, 6.4_
  
  - [ ]* 12.2 Create integration tests for complete 2-tower workflow
    - Test PredictionService skeleton interface
    - Test data flow from API → PredictionService → (future model)
    - Verify error handling and edge cases
    - _Requirements: 6.1, 6.4_

- [-] 13. Final validation and cleanup for PyTorch migration



  - [ ] 13.1 Verify both TensorFlow and PyTorch systems work after refactoring


    - Test existing TensorFlow /predict endpoint still works
    - Verify PyTorch skeleton is ready for model integration
    - Test that both systems can coexist during transition
    - Document which endpoints use which model
    - _Requirements: 6.1, 6.4_
  
  - [ ] 13.2 Update documentation for dual-model architecture
    - Update README.md to reflect new code structure
    - Document TensorFlow (legacy) vs PyTorch (future) systems
    - Add documentation for 2-tower architecture design
    - Document migration plan and timeline
    - Add setup instructions for both TensorFlow and PyTorch dependencies
    - _Requirements: 4.4, 6.5_