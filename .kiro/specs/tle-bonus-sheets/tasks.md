# Implementation Plan

- [ ] 1. Add UUID-to-set mapping infrastructure
  - Add `KNOWN_BONUS_SHEET_SETS` constant with common bonus sheet set codes (SPG, PLST, J25, J22, J21)
  - Implement `build_uuid_to_set_mapping()` function that maps UUIDs to their source set codes
  - Use efficient lookup strategy: check known bonus sets first, then search if needed
  - Add caching to avoid repeated API calls for the same UUIDs
  - _Requirements: 1.1, 1.2, 5.1, 5.2, 5.4_

- [ ] 2. Implement bonus sheet set identification
  - Implement `identify_bonus_sheet_sets()` function that compares booster UUIDs vs main set UUIDs
  - Extract UUIDs from main set cards to identify which UUIDs are bonus cards
  - Use UUID-to-set mapping to determine source sets for bonus UUIDs
  - Filter out companion sets (already handled by `fetch_companion_sets()`)
  - Return list of unique bonus sheet set codes
  - _Requirements: 1.1, 1.2, 3.1, 3.2, 5.1_

- [ ] 3. Create dynamic bonus sheet fetcher
  - Implement `fetch_bonus_sheet_cards_dynamic()` function that replaces hardcoded bonus sheet logic
  - Call `identify_bonus_sheet_sets()` to discover which bonus sets to fetch
  - Fetch card data from each identified bonus set using `fetch_set_data()`
  - Filter fetched cards to only those whose UUIDs appear in booster config
  - Further filter to only cards present in 17Lands training data (card_names_filter)
  - Add comprehensive logging showing discovered bonus sets and card counts
  - _Requirements: 2.1, 2.2, 2.3, 3.3, 4.4_

- [ ] 4. Add error handling and logging
  - Add try-except blocks around bonus set fetching with graceful degradation
  - Log warnings when bonus sets cannot be fetched (HTTP errors, missing sets)
  - Log info messages showing bonus sheet discovery statistics (sets found, cards added)
  - Add debug logging for UUID mapping process
  - Ensure processing continues even if some bonus sets fail to fetch
  - _Requirements: 2.4, 2.5, 3.3, 3.4, 5.3_

- [ ] 5. Integrate with fetch_all_card_data()
  - Update `fetch_all_card_data()` to call `fetch_bonus_sheet_cards_dynamic()` instead of `fetch_bonus_sheet_cards()`
  - Pass main set cards to the new function (needed to identify bonus UUIDs)
  - Ensure all fetched cards (main, companion, bonus) are combined correctly
  - Maintain existing function signature for backward compatibility
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. Update preprocess_cards.py integration
  - Verify `preprocess_set()` works with enhanced `fetch_all_card_data()`
  - Ensure no changes needed to calling code (backward compatible)
  - Verify output files (card_encodings.pkl, sheets.json, etc.) are generated correctly
  - _Requirements: 4.1, 4.2, 4.5_

- [ ]* 7. Add unit tests for new functions
  - Test `build_uuid_to_set_mapping()` with known UUIDs from different sets
  - Test `identify_bonus_sheet_sets()` with mock data (main set + bonus UUIDs)
  - Test `fetch_bonus_sheet_cards_dynamic()` with TLE-like scenario
  - Test error handling (missing sets, HTTP errors, invalid UUIDs)
  - Test caching behavior for UUID mapping
  - _Requirements: All requirements (validation)_

- [ ]* 8. Add integration test for TLE preprocessing
  - Create integration test that runs full preprocessing on TLE
  - Verify Source Material cards are included in output
  - Verify card counts match expectations (main set + bonus sheets)
  - Verify backward compatibility with existing sets (MH3, BLB)
  - _Requirements: All requirements (end-to-end validation)_
