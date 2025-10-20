import pandas as pd
import random

SEP_POOL_TOKEN_NAME = "[SEP_POOL]"
SEP_PACK_TOKEN_NAME = "[SEP_PACK]"
PAD_TOKEN_NAME = "[PAD]"

class DraftData:
    # Define constants for special token names

    def __init__(self, data_path):
        self._draft_data = pd.read_csv(data_path, low_memory=False)

        self._base_cards = [column[len("pack_card_"):] for column in self._draft_data.columns if column.startswith("pack_card")]
        self._cards = self._base_cards + [SEP_POOL_TOKEN_NAME, SEP_PACK_TOKEN_NAME, PAD_TOKEN_NAME]


        self._card_to_int = dict((c, i) for i, c in enumerate(self._cards))
        self._int_to_card = dict((i, c) for i, c in enumerate(self._cards))

        self._n_vocab = len(self._cards) + 1
        self._sep_pool_token = self._card_to_int[SEP_POOL_TOKEN_NAME]
        self._sep_pack_token = self._card_to_int[SEP_PACK_TOKEN_NAME]
        self._pad_token = self._card_to_int[PAD_TOKEN_NAME]
        
        #takes only the pack that has the full 14 cards
        self._full_pack_indices = self._draft_data[self._draft_data['pick_number'] == 0].index.tolist()

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

    #picks a random booster within the dataset
    def boosterCreater(self):
        if not self._full_pack_indices:
            raise ValueError("No full packs found in dataset")
        
        # Randomly select an index from pre-computed list
        index = random.choice(self._full_pack_indices)
        row = self._draft_data.iloc[index]

        pack = []
        for k, v in row.items():
            if k.startswith("pack_card_") and v == 1:
                pack.append(self._card_to_int[k[len("pack_card_"):]])
        
        return pack

    @property
    def empty_card(self):
        return len(self._cards)