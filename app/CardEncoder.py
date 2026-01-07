import json
import re
from typing import Any

import numpy as np
from numpy import ndarray, dtype
from sentence_transformers import SentenceTransformer


class CardEncoder:
    def __init__(self, data_path=None, card_list=None):
        #Load model
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model.to('cuda')


        if not data_path is None:
            self.data_path = data_path

            with open(data_path, 'r') as f:
                self.cards = json.load(f)

        else:
            if card_list is None:
                raise ValueError("Card Encoder: data_path or card_list is required")
            self.cards = card_list

    def encode(self, card):
        # 1. Gather all sub-encodings
        rarity_vec = self.encode_rarity(card.get('rarity', 'common'))

        mana_vec = self.encode_mana_cost(
            card.get('converted_mana_cost', 0),
            card.get('mana_cost', '')
        )

        type_vec = self.encode_types(card.get('types', []))

        pt_vec = self.encode_power_toughhness(
            # can_attack logic: usually if it has power/toughness
            can_attack=(card.get('can_attack', False)),
            power=card.get('power'),
            toughness=card.get('toughness')
        )

        text_vec = self.encode_oracle_text(
            card.get('oracle_text', ''),
            card.get('subtypes', [])
        )

        #Flatten the vector into one vector
        full_vector = np.concatenate([
            rarity_vec,
            mana_vec,
            type_vec,
            pt_vec,
            text_vec
        ])

        return full_vector


    #ONE HOT ENCODING
    def encode_rarity(self, rarity):
        RARITIES=["common", "uncommon", "rare", "mythic"]
        encoding = np.array([0] * len(RARITIES), dtype=np.float32)
        if rarity not in RARITIES:
            raise ValueError("Unknown rarity")
        else:
            encoding[RARITIES.index(rarity)] = 1

        return encoding


    def encode_mana_cost(self, converted_mana_cost, mana_cost):
        COLORS = ["W", "B", "U", "R", "G", "P"]
        encoding = np.array(([0] * (len(COLORS) + 1)), dtype=np.float32)


        if not mana_cost == "":
            #use regex
            separated_colors = re.findall(r"{(.*?)}", mana_cost)
            for pips in separated_colors:
                #start with hybrid
                if "/" in pips:
                    parts = pips.split("/")
                    print(parts)
                    for part in parts:
                        if not part.isdigit():
                            encoding[COLORS.index(part)] += 0.5

                elif not pips.isdigit():
                    encoding[COLORS.index(pips)] += 1

            encoding = encoding / 8
            # Add converted mana cost
            encoding[len(COLORS)] = (converted_mana_cost / 16)
        return encoding

    def encode_types(self, types):
        TYPES = ["artifact", "battle", "creature", "enchantment", "instant", "land", "planeswalker", "sorcery", "kindred"]
        encoding = np.array([0] * len(TYPES), dtype=np.float32)

        for type in types:
            if type.lower() in TYPES:
                encoding[TYPES.index(type.lower())] = 1

        return encoding

    def encode_power_toughhness(self, can_attack, power=None, toughness=None):
        #goes can_attack, power, toughness
        encoding = np.zeros(3, dtype=np.float32)

        if can_attack:
            encoding[0] =  1

            #check for weird symbols
            if power is None:
                encoding[1] = 0
            else:
                encoding[1] = power/15

            if toughness is None:
                encoding[2] = 0
            else:
                encoding[2] = toughness/15
        return encoding

    def encode_oracle_text(self, oracle_text, subtypes):
        if not oracle_text:
            return np.zeros(384, dtype=np.float32)

        encoding = self.text_model.encode(oracle_text + " " + str(subtypes))
        return encoding

