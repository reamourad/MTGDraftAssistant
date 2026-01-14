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

- [ ] 4. Create core business logic layer for PyTorch model
  - [ ] 4.1 Create app/core/prediction.py for PyTorch prediction orchestration
    - Extract prediction logic from api.py into PredictionService class
    - Implement interface between API and PyTorch ML layers
    - Handle tensor conversion and GPU/CPU management
    - _Requirements: 2.3, 4.1, 4.3_
  
  - [ ] 4.2 Create app/core/booster.py for booster generation
    - Move booster generation logic from app/booster/generator.py
    - Create BoosterService class with clean interface
    - _Requirements: 2.3, 4.1_

- [ ] 5. Refactor machine learning layer for PyTorch two-tower architecture
  - [ ] 5.1 Create app/ml/experimental/two_tower_model.py for complete model integration
    - Integrate CandidateTower, ContextTower, and ScoringHead into unified model
    - Implement training and inference pipeline for two-tower architecture
    - _Requirements: 2.4, 4.1_
  
  - [ ] 5.2 Create app/ml/experimental/model_loader.py for PyTorch model loading
    - Implement PyTorch model loading and caching
    - Handle model state management and GPU/CPU allocation
    - _Requirements: 2.4, 4.1_
  
  - [ ] 5.3 Enhance PyTorch components with proper interfaces
    - Add proper error handling and validation to all PyTorch components
    - Standardize input/output formats across towers
    - _Requirements: 2.4, 3.2_

- [ ] 6. Refactor data layer for PyTorch architecture
  - [ ] 6.1 Create app/data/repositories.py for data access patterns
    - Create DataRepository class for centralized data access
    - Implement methods for loading set data, card encodings, booster configs
    - Focus on data formats compatible with PyTorch tensors
    - _Requirements: 2.5, 4.1, 4.3_
  
  - [ ] 6.2 Create app/data/card_data.py for card encoding and processing
    - Create unified interface for card data processing
    - Integrate with CardEncoder for consistent data flow
    - _Requirements: 2.5, 3.1, 3.3_

- [ ] 7. Create services layer for external integrations
  - [ ] 7.1 Create app/services/mtgjson.py for MTGJson API integration
    - Move MTGJson fetching logic from app/booster/mtgjson_fetcher.py
    - Create clean service interface for external API calls
    - _Requirements: 2.5, 4.1_

- [ ] 8. Create utilities layer
  - [ ] 8.1 Create app/utils/config.py for configuration management
    - Centralize configuration constants and settings
    - Remove hardcoded values from various modules
    - _Requirements: 3.3, 4.4_
  
  - [ ] 8.2 Create app/utils/cache.py for caching utilities
    - Extract caching logic into reusable utilities
    - Implement proper cache management and cleanup
    - _Requirements: 4.2_
  
  - [ ] 8.3 Create app/utils/exceptions.py for centralized error handling
    - Define domain-specific exception classes
    - Implement consistent error handling patterns
    - _Requirements: 4.4_

- [ ] 9. Optimize JSON file usage and data persistence
  - [ ] 9.1 Audit and consolidate JSON files in preprocessing
    - Modify preprocess_cards.py to generate only essential JSON files
    - Remove or consolidate cards.json with other data files
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [ ] 9.2 Update data loading to work with optimized file structure
    - Modify data repositories to work with consolidated JSON files
    - Ensure backward compatibility during transition
    - _Requirements: 7.4, 7.5_

- [ ] 10. Update main application entry points for PyTorch architecture
  - [ ] 10.1 Update existing api.py to import from new PyTorch structure
    - Create compatibility layer that imports from new app/api/main.py
    - Ensure PyTorch model integration works with existing deployment
    - _Requirements: 6.1, 6.4_
  
  - [ ] 10.2 Update preprocess_cards.py to work with PyTorch data requirements
    - Update imports to use new module locations
    - Ensure preprocessing pipeline generates PyTorch-compatible data
    - _Requirements: 6.1, 6.5_

- [ ] 11. Clean up old files and update imports for PyTorch focus
  - [ ] 11.1 Remove old TensorFlow files after migration
    - Archive or remove TensorFlow-based components that are no longer needed
    - Keep PyTorch experimental components as the primary ML system
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [ ] 11.2 Update all import statements for PyTorch architecture
    - Fix imports in all files to use new PyTorch-focused module structure
    - Ensure consistent import patterns for PyTorch components
    - Update all references to use experimental PyTorch components as primary
    - _Requirements: 3.3, 6.5_

- [ ]* 12. Add comprehensive testing for PyTorch architecture
  - [ ]* 12.1 Create unit tests for PyTorch components
    - Write unit tests for CandidateTower, ContextTower, ScoringHead
    - Test card encoding and tensor operations
    - Mock GPU operations for consistent testing
    - _Requirements: 4.4, 6.4_
  
  - [ ]* 12.2 Create integration tests for PyTorch two-tower workflow
    - Test complete prediction workflow from API to PyTorch model
    - Test tensor flow through the entire two-tower architecture
    - _Requirements: 6.1, 6.4_

- [ ] 13. Final validation and cleanup for PyTorch architecture
  - [ ] 13.1 Verify PyTorch two-tower architecture works after refactoring
    - Test API endpoints with PyTorch model predictions
    - Verify model loading and PyTorch prediction accuracy
    - Test GPU/CPU compatibility and performance
    - _Requirements: 6.1, 6.4_
  
  - [ ] 13.2 Update documentation for PyTorch architecture
    - Update README.md to reflect new PyTorch-focused code structure
    - Add documentation for two-tower architecture and interfaces
    - Document PyTorch model training and inference procedures
    - _Requirements: 4.4, 6.5_