import pandas as pd
import random

class DraftData:
    def __init__(self, data_path):
        self._draft_data = pd.read_csv(data_path, low_memory=False)

        self._cards = [column[len("pack_card_"):] for column in self._draft_data.columns if column.startswith("pack_card")]

        self._n_vocab = len(self._cards) + 1
        self._card_to_int = dict((c, i) for i, c in enumerate(self._cards))
        self._int_to_card = dict((i, c) for i, c in enumerate(self._cards))

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

    #picks a random booster within the dataset
    def boosterCreater(self):
            index = random.randint(0, int((len(self._draft_data)/14) - 1)) * 14
            print(index)
            row = self._draft_data.iloc[index]

            pack = []
            for k, v in row.items():
                if k.startswith("pack_card_") and v == 1:
                    pack.append(self._card_to_int[k[len("pack_card_"):]])
            return pack

    @property
    def empty_card(self):
        return len(self._cards)