# Requirements Document

## Introduction

The MTG Draft Assistant codebase has grown organically and now suffers from architectural inconsistencies, mixed ML frameworks (TensorFlow and PyTorch), unclear separation of concerns, and messy code organization. This cleanup effort aims to create a clean, maintainable, and consistent codebase with clear architectural boundaries and unified patterns.

## Glossary

- **MTG_Draft_Assistant**: The main application system for predicting optimal Magic: The Gathering draft picks
- **API_Layer**: FastAPI-based REST interface for serving predictions
- **Model_Layer**: Machine learning components for card pick prediction
- **Data_Layer**: Components handling card data, draft data, and encoding
- **Framework**: Either TensorFlow/Keras or PyTorch for ML operations
- **Component**: A logical unit of functionality (API, model, data processing, etc.)

## Requirements

### Requirement 1

**User Story:** As a developer, I want a unified ML framework throughout the codebase, so that I can maintain consistency and avoid framework conflicts.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL choose either TensorFlow or PyTorch as the primary ML framework for the production system
2. THE MTG_Draft_Assistant SHALL clearly separate current working models from experimental/future architecture components
3. THE MTG_Draft_Assistant SHALL organize framework-specific components into distinct modules
4. THE MTG_Draft_Assistant SHALL ensure the chosen framework is used consistently for the active prediction pipeline

### Requirement 2

**User Story:** As a developer, I want clear separation of concerns between API, model, and data layers, so that I can easily understand and modify each component independently.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL organize code into distinct layers: API, Model, and Data
2. THE API_Layer SHALL only handle HTTP requests, responses, and routing logic
3. THE Model_Layer SHALL only handle ML model operations, training, and predictions
4. THE Data_Layer SHALL only handle data loading, preprocessing, and encoding
5. WHEN one layer needs functionality from another, THE MTG_Draft_Assistant SHALL use well-defined interfaces

### Requirement 3

**User Story:** As a developer, I want consistent naming conventions and code organization, so that I can quickly navigate and understand the codebase.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL use consistent Python naming conventions (snake_case for functions/variables, PascalCase for classes)
2. THE MTG_Draft_Assistant SHALL organize files into logical directories based on their responsibility
3. THE MTG_Draft_Assistant SHALL remove unused or duplicate code files
4. THE MTG_Draft_Assistant SHALL use consistent import patterns throughout the codebase

### Requirement 4

**User Story:** As a developer, I want clean, well-documented interfaces between components, so that I can understand how different parts of the system interact.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL define clear interfaces between API, Model, and Data layers
2. THE MTG_Draft_Assistant SHALL use dependency injection or factory patterns to manage component dependencies
3. THE MTG_Draft_Assistant SHALL remove tight coupling between unrelated components
4. THE MTG_Draft_Assistant SHALL document public interfaces with type hints and docstrings

### Requirement 5

**User Story:** As a developer, I want to remove dead code and unused components, so that the codebase is lean and maintainable.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL identify and remove unused Python files
2. THE MTG_Draft_Assistant SHALL identify and remove unused functions and classes
3. THE MTG_Draft_Assistant SHALL identify and remove unused imports
4. THE MTG_Draft_Assistant SHALL consolidate duplicate functionality into single implementations

### Requirement 6

**User Story:** As a coder, I want to have a clean structure and code flow, so that I can easily read the code.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL organize code with a clear, logical flow from entry points to core functionality
2. THE MTG_Draft_Assistant SHALL minimize circular dependencies between modules
3. THE MTG_Draft_Assistant SHALL structure directories to reflect the application's logical architecture
4. THE MTG_Draft_Assistant SHALL ensure each module has a single, well-defined responsibility
5. THE MTG_Draft_Assistant SHALL use consistent patterns for similar operations across the codebase

### Requirement 7

**User Story:** As a developer, I want to save only relevant components to the API/training process, so that I don't have unnecessary JSON files cluttering the codebase.

#### Acceptance Criteria

1. THE MTG_Draft_Assistant SHALL identify all JSON files currently being saved throughout the codebase
2. THE MTG_Draft_Assistant SHALL determine which JSON files are essential for API operations or training processes
3. THE MTG_Draft_Assistant SHALL remove or consolidate non-essential JSON file generation
4. THE MTG_Draft_Assistant SHALL ensure only necessary data persistence occurs during normal operations
5. THE MTG_Draft_Assistant SHALL document what data is saved and why it's needed