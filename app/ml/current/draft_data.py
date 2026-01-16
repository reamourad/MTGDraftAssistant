"""
LEGACY TENSORFLOW DRAFT DATA - PRODUCTION SYSTEM

This module handles data loading and preprocessing for the TensorFlow-based model.
This is part of the LEGACY system currently running in production.

⚠️ DEPRECATION NOTICE ⚠️
This module is being replaced by the PyTorch-compatible data layer in:
- app/data/repositories.py
- app/ml/experimental/card_encoder.py

Status: ACTIVE (Legacy)
Framework: TensorFlow/Keras
Replacement: app/data/ and app/ml/experimental/ (PyTorch)

Do NOT add new features to this module. All new data processing should target
the PyTorch-compatible data layer.
"""

import pandas as pd
import random

SEP_POOL_TOKEN_NAME = "[SEP_POOL]"
SEP_PACK_TOKEN_NAME = "[SEP_PACK]"
PAD_TOKEN_NAME = "[PAD]"

class DraftData:
    # Define constants for special token names

    def __init__(self, data_path=None, card_list=None):
        """
        Initialize DraftData with either a CSV path or a card list.

        Args:
            data_path: Path to CSV file (for training)
            card_list: List of card names (for prediction)
        """
        if data_path is None and card_list is None:
            raise ValueError("Must provide either data_path or card_list")

        if card_list is not None:
            # Lightweight mode: only card names, no CSV data
            self._draft_data = None
            self._base_cards = card_list
            self._full_pack_indices = []
        else:
            # Full mode: load CSV for training
            self._draft_data = pd.read_csv(data_path, low_memory=False, nrows=1000000)
            self._base_cards = [column[len("pack_card_"):] for column in self._draft_data.columns if column.startswith("pack_card")]
            #takes only the pack that has the full 14 cards
            self._full_pack_indices = self._draft_data[self._draft_data['pick_number'] == 0].index.tolist()

        self._cards = self._base_cards + [SEP_POOL_TOKEN_NAME, SEP_PACK_TOKEN_NAME, PAD_TOKEN_NAME]

        self._card_to_int = dict((c, i) for i, c in enumerate(self._cards))
        self._int_to_card = dict((i, c) for i, c in enumerate(self._cards))

        self._n_vocab = len(self._cards) + 1
        self._sep_pool_token = self._card_to_int[SEP_POOL_TOKEN_NAME]
        self._sep_pack_token = self._card_to_int[SEP_PACK_TOKEN_NAME]
        self._pad_token = self._card_to_int[PAD_TOKEN_NAME]

    @property
    def draft_data(self):
        return self._draft_data

    @property
    def cards(self):
        return self._cards

    @property
    def n_vocab(self):
        return self._n_vocab

    @property
    def int_to_card(self):
        return self._int_to_card

    @property
    def cards_to_int(self):
        return self._card_to_int
    
    @property
    def pad_token(self):
        return self._pad_token

    @property
    def empty_card(self):
        return len(self._cards)