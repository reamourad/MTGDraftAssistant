# Design Document

## Overview

This design enhances the MTGJson fetcher to dynamically discover and fetch bonus sheet cards by analyzing the booster configuration's UUID references. Instead of hardcoding specific bonus sheet sets (SPG, PLST), the system will:

1. Extract all UUIDs from the booster configuration
2. Map each UUID to its source set using MTGJson's card database
3. Identify which sets contribute cards to boosters (beyond the main set)
4. Fetch cards from all identified bonus sheet sets
5. Filter to only cards present in 17Lands training data

This approach is future-proof and works for any set with bonus sheets (TLE, MH3, future sets).

## Architecture

### Current Flow
```
preprocess_cards.py
  └─> mtgjson_fetcher.fetch_all_card_data()
       ├─> fetch_set_data(main_set)
       ├─> fetch_companion_sets() [Commander decks]
       └─> fetch_bonus_sheet_cards() [SPG, PLST only]
```

### Enhanced Flow
```
preprocess_cards.py
  └─> mtgjson_fetcher.fetch_all_card_data()
       ├─> fetch_set_data(main_set)
       ├─> fetch_companion_sets() [Commander decks]
       └─> fetch_bonus_sheet_cards_dynamic() [ALL bonus sheets]
            ├─> extract_uuids_from_booster_config()
            ├─> map_uuids_to_sets() [NEW]
            ├─> identify_bonus_sheet_sets() [NEW]
            └─> fetch_cards_from_bonus_sets() [NEW]
```

## Components and Interfaces

### 1. UUID-to-Set Mapper

**Purpose:** Build a mapping from card UUIDs to their source set codes.

**Implementation:**
```python
def build_uuid_to_set_mapping(uuids: Set[str]) -> Dict[str, str]:
    """
    Map card UUIDs to their source set codes.
    
    Args:
        uuids: Set of card UUIDs to map
    
    Returns:
        Dictionary mapping UUID -> set_code
    """
```

**Strategy:**
- Use MTGJson's `AllPrintings.json` or individual set files
- For efficiency, only fetch sets that might contain the UUIDs
- Cache results to avoid repeated API calls
- Use the existing `get_set_list()` cache

**Alternative Approach (Chosen):**
- Fetch individual set files on-demand
- Start with known bonus sheet sets (SPG, PLST, J25, etc.)
- Use a fallback search if UUID not found in common sets

### 2. Bonus Sheet Set Identifier

**Purpose:** Identify which sets contribute bonus sheet cards to the main set's boosters.

**Implementation:**
```python
def identify_bonus_sheet_sets(
    main_set_code: str,
    booster_uuids: Set[str],
    main_set_uuids: Set[str]
) -> List[str]:
    """
    Identify bonus sheet sets by finding UUIDs not in main set.
    
    Args:
        main_set_code: Main set code (e.g., 'TLE')
        booster_uuids: All UUIDs in booster config
        main_set_uuids: UUIDs from main set cards
    
    Returns:
        List of bonus sheet set codes
    """
```

**Logic:**
1. Find UUIDs in booster config but not in main set
2. Map these UUIDs to their source sets
3. Exclude companion sets (already handled separately)
4. Return unique set codes

### 3. Dynamic Bonus Sheet Fetcher

**Purpose:** Replace the hardcoded `fetch_bonus_sheet_cards()` with a dynamic version.

**Implementation:**
```python
def fetch_bonus_sheet_cards_dynamic(
    set_code: str,
    booster_config: Dict[str, Any],
    all_main_cards: List[Dict[str, Any]],
    card_names_filter: set
) -> List[Dict[str, Any]]:
    """
    Dynamically fetch bonus sheet cards by analyzing booster config.
    
    Args:
        set_code: Main set code
        booster_config: Booster configuration with UUIDs
        all_main_cards: Cards from main set (to identify bonus UUIDs)
        card_names_filter: Card names from 17Lands CSV
    
    Returns:
        List of bonus sheet cards
    """
```

**Steps:**
1. Extract all UUIDs from booster config
2. Extract UUIDs from main set cards
3. Identify bonus UUIDs (in booster but not in main set)
4. Map bonus UUIDs to their source sets
5. Fetch cards from each identified bonus set
6. Filter to only cards in 17Lands data
7. Return combined list

### 4. Known Bonus Sheet Sets Registry

**Purpose:** Maintain a list of common bonus sheet sets for efficient lookup.

**Implementation:**
```python
KNOWN_BONUS_SHEET_SETS = [
    'SPG',   # Special Guests
    'PLST',  # The List
    'J25',   # Jumpstart 2025
    'J22',   # Jumpstart 2022
    'J21',   # Jumpstart 2021
    # Add more as discovered
]
```

**Usage:**
- Check these sets first when mapping UUIDs
- Reduces API calls for common cases
- Can be extended as new bonus sheet types are discovered

## Data Models

### UUID Mapping Cache
```python
{
    "uuid1": "SPG",
    "uuid2": "PLST",
    "uuid3": "TLE",
    ...
}
```

### Bonus Sheet Discovery Result
```python
{
    "main_set": "TLE",
    "bonus_sets": ["SPG", "J25"],
    "stats": {
        "total_booster_uuids": 350,
        "main_set_uuids": 280,
        "bonus_uuids": 70,
        "bonus_by_set": {
            "SPG": 10,
            "J25": 60
        }
    }
}
```

## Error Handling

### UUID Mapping Failures
- **Issue:** UUID not found in any known set
- **Handling:** Log warning with UUID, continue processing
- **Impact:** Card won't be available for training (acceptable if not in 17Lands data)

### Bonus Set Fetch Failures
- **Issue:** HTTP error fetching bonus set from MTGJson
- **Handling:** Log warning, skip that bonus set, continue with others
- **Impact:** Some bonus cards may be missing (same as current behavior)

### Empty Booster Configuration
- **Issue:** Set has no booster configuration
- **Handling:** Return empty list, log info message
- **Impact:** No bonus cards fetched (expected for some sets)

## Testing Strategy

### Unit Tests
1. **test_build_uuid_to_set_mapping()**
   - Test with known UUIDs from different sets
   - Test with invalid UUIDs
   - Test caching behavior

2. **test_identify_bonus_sheet_sets()**
   - Test with TLE data (should find J25/SPG)
   - Test with MH3 data (should find SPG)
   - Test with set that has no bonus sheets

3. **test_fetch_bonus_sheet_cards_dynamic()**
   - Test full workflow with TLE
   - Test filtering by 17Lands card names
   - Test error handling for missing sets

### Integration Tests
1. **test_preprocess_tle()**
   - Run full preprocessing on TLE
   - Verify Source Material cards are included
   - Verify card counts match expectations

2. **test_backward_compatibility()**
   - Run preprocessing on MH3, BLB
   - Verify results unchanged from current implementation

### Manual Testing
1. Run `python preprocess_cards.py TLE`
2. Verify output includes Source Material cards
3. Check logs for bonus sheet discovery messages
4. Verify `card_encodings.pkl` contains expected card count

## Performance Considerations

### API Call Optimization
- **Cache set list:** Already implemented in `get_set_list()`
- **Batch UUID lookups:** Check multiple UUIDs per set fetch
- **Known sets first:** Check common bonus sets before searching all sets

### Memory Usage
- **Stream large sets:** Don't load entire AllPrintings.json
- **Filter early:** Apply 17Lands filter as soon as possible
- **Release references:** Clear intermediate data structures

### Expected Performance
- **TLE preprocessing:** ~2-3 minutes (similar to current)
- **Additional API calls:** 2-3 bonus set fetches (SPG, J25, etc.)
- **Memory overhead:** Minimal (<100MB for UUID mapping)

## Migration Strategy

### Phase 1: Add New Functions (Non-Breaking)
- Add `build_uuid_to_set_mapping()`
- Add `identify_bonus_sheet_sets()`
- Add `fetch_bonus_sheet_cards_dynamic()`
- Keep existing `fetch_bonus_sheet_cards()` unchanged

### Phase 2: Update fetch_all_card_data()
- Replace call to `fetch_bonus_sheet_cards()` with `fetch_bonus_sheet_cards_dynamic()`
- Add logging for bonus sheet discovery
- Test with existing sets (MH3, BLB, etc.)

### Phase 3: Deprecate Old Function
- Mark `fetch_bonus_sheet_cards()` as deprecated
- Remove after confirming all sets work correctly

## Alternative Approaches Considered

### Approach 1: Fetch AllPrintings.json
- **Pros:** Single API call, complete UUID mapping
- **Cons:** Very large file (~500MB), slow download, memory intensive
- **Decision:** Rejected due to performance concerns

### Approach 2: Hardcode TLE Bonus Sets
- **Pros:** Simple, fast, no UUID mapping needed
- **Cons:** Not future-proof, requires updates for each new set
- **Decision:** Rejected, doesn't solve the general problem

### Approach 3: Use MTGJson's Card Search API
- **Pros:** Direct UUID lookup
- **Cons:** MTGJson doesn't have a card-by-UUID endpoint
- **Decision:** Not available in MTGJson API

### Approach 4: Analyze Sheet Names (Chosen Hybrid)
- **Pros:** Booster config often has descriptive sheet names
- **Cons:** Sheet names not standardized
- **Decision:** Use as hint, but rely on UUID mapping for accuracy

## Implementation Notes

### Sheet Name Hints
Some booster configs have descriptive sheet names that hint at bonus sets:
- "source_material" → Likely J25 or similar
- "special_guest" → SPG
- "the_list" → PLST

We can use these as hints to check specific sets first, improving performance.

### UUID Format
MTGJson UUIDs are consistent across all sets:
- Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- Example: `02fe4019-ae2e-53fc-b011-46dec318bde8`
- Unique per printing (same card in different sets has different UUIDs)

### Companion Sets vs Bonus Sheets
- **Companion Sets:** Have `parentCode` metadata, fetched by `fetch_companion_sets()`
- **Bonus Sheets:** No parent relationship, cards from unrelated sets
- **Distinction:** Important to avoid double-fetching

## Success Criteria

1. TLE preprocessing completes successfully
2. Source Material cards (61 cards) are included in output
3. All cards from 17Lands CSV are found and encoded
4. Existing sets (MH3, BLB, etc.) continue to work
5. Logs clearly show which bonus sets were discovered
6. No performance regression (processing time similar to current)
