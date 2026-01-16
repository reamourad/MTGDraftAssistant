# Requirements Document

## Introduction

Enhance the card preprocessing system to properly fetch and process bonus sheet cards from MTGJson for sets like TLE (Avatar: The Last Airbender Eternal). TLE contains the "Source Material" bonus sheet (61 cards), Jumpstart cards, and Commander-specific content that are not in the main set but appear in booster packs. The system currently only checks SPG and PLST bonus sheets, but needs to dynamically discover and fetch all bonus sheet cards referenced in a set's booster configuration.

## Glossary

- **MTGJson**: A JSON API that provides comprehensive Magic: The Gathering card data
- **Bonus Sheet**: Cards from other sets that appear in booster packs (e.g., Special Guests, The List, Source Material)
- **Booster Configuration**: MTGJson data structure that defines how booster packs are constructed, including which sheets and cards appear
- **UUID**: Unique identifier for each card in MTGJson
- **17Lands**: A data collection service that provides draft pick data from real players
- **Card Preprocessing System**: The system that fetches card data from MTGJson and encodes it for training
- **Source Material**: A bonus sheet in TLE containing 61 reprinted cards
- **Jumpstart Cards**: Special cards designed for the Jumpstart format
- **Commander Content**: Cards from Commander decks associated with a set

## Requirements

### Requirement 1: Dynamic Bonus Sheet Discovery

**User Story:** As a developer preprocessing a new set, I want the system to automatically discover all bonus sheet sets referenced in the booster configuration, so that I don't need to manually configure which bonus sets to check for each new set release.

#### Acceptance Criteria

1. WHEN the Card Preprocessing System processes a set's booster configuration, THE System SHALL extract all unique set codes referenced in the booster sheets
2. WHEN the System identifies set codes that differ from the main set code, THE System SHALL treat these as potential bonus sheet sets
3. WHEN the System discovers bonus sheet sets, THE System SHALL log the discovered set codes for transparency
4. THE System SHALL maintain backward compatibility with the existing SPG and PLST bonus sheet handling

### Requirement 2: Bonus Sheet Card Fetching

**User Story:** As a developer preprocessing TLE, I want the system to fetch cards from all bonus sheet sets (including Source Material), so that all cards appearing in booster packs are available for training.

#### Acceptance Criteria

1. WHEN the System identifies a bonus sheet set code in the booster configuration, THE System SHALL fetch the complete card data for that set from MTGJson
2. WHEN the System fetches bonus sheet card data, THE System SHALL filter cards to only those whose UUIDs appear in the main set's booster configuration
3. WHEN the System filters bonus sheet cards, THE System SHALL further filter to only cards present in the 17Lands training data
4. IF a bonus sheet set cannot be fetched from MTGJson, THEN THE System SHALL log a warning and continue processing without failing
5. THE System SHALL handle HTTP errors gracefully when fetching bonus sheet data

### Requirement 3: Booster Configuration Analysis

**User Story:** As a developer, I want the system to analyze the booster configuration to identify which sets contribute cards to boosters, so that I can understand the composition of complex sets like TLE.

#### Acceptance Criteria

1. WHEN the System processes a booster configuration, THE System SHALL extract all card UUIDs from all sheets
2. WHEN the System has extracted UUIDs, THE System SHALL group UUIDs by their source set code
3. WHEN the System groups UUIDs by set, THE System SHALL log statistics showing how many cards come from each set
4. THE System SHALL provide clear logging that distinguishes between main set cards, companion set cards, and bonus sheet cards

### Requirement 4: Integration with Existing Workflow

**User Story:** As a developer, I want the bonus sheet enhancement to integrate seamlessly with the existing preprocessing workflow, so that I can process any set (TLE, MH3, etc.) without changing my workflow.

#### Acceptance Criteria

1. THE System SHALL maintain the existing preprocessing workflow in preprocess_cards.py
2. THE System SHALL use the enhanced fetch_all_card_data function without requiring changes to the calling code
3. WHEN processing a set without bonus sheets, THE System SHALL complete successfully without errors
4. WHEN processing a set with bonus sheets, THE System SHALL automatically fetch and include bonus sheet cards
5. THE System SHALL save all fetched cards (main set, companion sets, and bonus sheets) to the same output files

### Requirement 5: UUID-to-Set Mapping

**User Story:** As a developer, I want the system to determine which set a card UUID belongs to, so that bonus sheet cards can be identified and fetched from the correct source.

#### Acceptance Criteria

1. WHEN the System encounters a card UUID in the booster configuration, THE System SHALL determine which set that UUID belongs to
2. THE System SHALL use MTGJson's set list API to build a UUID-to-set-code mapping
3. IF a UUID cannot be mapped to a set, THEN THE System SHALL log a warning with the UUID
4. THE System SHALL cache the set list to avoid repeated API calls during processing
5. THE System SHALL handle cases where a UUID appears in multiple sets by using the first match found
