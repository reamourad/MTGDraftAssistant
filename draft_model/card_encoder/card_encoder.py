import re
import logging
import time
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

RARITIES = ["common", "uncommon", "rare", "mythic"]
COLORS = ["W", "B", "U", "R", "G", "P"]
TYPES = ["artifact", "battle", "creature", "enchantment", "instant", "land", "planeswalker", "sorcery", "kindred"]


class CardEncoderError(Exception):
    pass


class CardEncoder:
    """
    Encodes a card dict (matching MTGJson.get_uuid_to_card_features's shape) into a fixed vector:
      - Rarity: 4 dims (one-hot)
      - Mana cost: 7 dims (6 colors + CMC)
      - Types: 9 dims (multi-hot)
      - Power/Toughness: 3 dims (can_attack, power, toughness)
      - Oracle text: 384 dims (raw sentence embedding, not yet compressed)
    Total: 407 dims.
    """

    STRUCTURED_DIM = 4 + (len(COLORS) + 1) + len(TYPES) + 3
    ORACLE_TEXT_DIM = 384
    TOTAL_DIM = STRUCTURED_DIM + ORACLE_TEXT_DIM

    def __init__(self, use_gpu: bool = True):
        try:
            self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            self.text_model.to(device)
            logger.info(f"CardEncoder initialized with device: {device}")
        except Exception as e:
            raise CardEncoderError(f"Failed to load sentence transformer model: {e}") from e

        self._cache: dict[str, np.ndarray] = {}

    def encode(self, card: dict) -> np.ndarray:
        #check if we previously encoded the card, reuse if we did
        name = card.get('name')
        if name is not None and name in self._cache:
            return self._cache[name]

        try:
            structured_vec = self._encode_structured(card)
            text_vec = self._encode_oracle_text(
                card.get('oracle_text', ''),
                card.get('subtypes', []),
            )

            full_vector = np.concatenate([structured_vec, text_vec])
            self._check_dim(full_vector, name)

            if name is not None:
                self._cache[name] = full_vector

            return full_vector
        except Exception as e:
            card_name = card.get('name', 'Unknown')
            raise CardEncoderError(f"Failed to encode card '{card_name}': {e}") from e

    def encode_batch(self, cards: List[dict]) -> np.ndarray:
        if not cards:
            return np.zeros((0, self.TOTAL_DIM), dtype=np.float32)

        results: List[Optional[np.ndarray]] = [None] * len(cards)
        miss_indices = []
        for i, card in enumerate(cards):
            name = card.get('name')
            if name is not None and name in self._cache:
                results[i] = self._cache[name]
            else:
                miss_indices.append(i)

        if miss_indices:
            start = time.perf_counter()
            structured_by_index = {i: self._encode_structured(cards[i]) for i in miss_indices}
            text_input_by_index = {
                i: self._oracle_text_input(cards[i].get('oracle_text', ''), cards[i].get('subtypes', []))
                for i in miss_indices
            }

            text_vec_by_index = {i: np.zeros(self.ORACLE_TEXT_DIM, dtype=np.float32) for i in miss_indices}
            non_empty_indices = [i for i in miss_indices if text_input_by_index[i]]
            if non_empty_indices:
                texts = [text_input_by_index[i] for i in non_empty_indices]
                encoded = self.text_model.encode(texts, convert_to_numpy=True)
                for row, i in enumerate(non_empty_indices):
                    text_vec_by_index[i] = encoded[row].astype(np.float32)

            for i in miss_indices:
                card = cards[i]
                name = card.get('name')
                full_vector = np.concatenate([structured_by_index[i], text_vec_by_index[i]])
                self._check_dim(full_vector, name)
                if name is not None:
                    self._cache[name] = full_vector
                results[i] = full_vector

            elapsed = time.perf_counter() - start
            print(f"[CardEncoder] batch-encoded {len(miss_indices)} cache-miss card(s) "
                  f"({len(non_empty_indices)} with text) in {elapsed:.2f}s")

        return np.stack(results, axis=0)

    def _encode_structured(self, card: dict) -> np.ndarray:
        rarity_vec = self._encode_rarity(card.get('rarity', 'common'))
        mana_vec = self._encode_mana_cost(
            card.get('converted_mana_cost', 0),
            card.get('mana_cost', '')
        )
        type_vec = self._encode_types(card.get('types', []))
        pt_vec = self._encode_power_toughness(
            can_attack=card.get('can_attack', False),
            power=card.get('power'),
            toughness=card.get('toughness'),
        )
        return np.concatenate([rarity_vec, mana_vec, type_vec, pt_vec])

    def _check_dim(self, full_vector: np.ndarray, name: Optional[str]):
        if full_vector.shape[0] != self.TOTAL_DIM:
            raise CardEncoderError(
                f"Encoded vector has wrong dimension for card '{name}': "
                f"{full_vector.shape[0]}, expected {self.TOTAL_DIM}"
            )

    def _encode_rarity(self, rarity: str) -> np.ndarray:
        encoding = np.zeros(len(RARITIES), dtype=np.float32)
        rarity_lower = rarity.lower()
        if rarity_lower not in RARITIES:
            logger.warning(f"Unknown rarity '{rarity}', defaulting to 'common'")
            rarity_lower = "common"
        encoding[RARITIES.index(rarity_lower)] = 1
        return encoding

    def _encode_mana_cost(self, converted_mana_cost: float, mana_cost: str) -> np.ndarray:
        encoding = np.zeros(len(COLORS) + 1, dtype=np.float32)

        if mana_cost:
            try:
                separated_colors = re.findall(r"{(.*?)}", mana_cost)
                for pips in separated_colors:
                    if "/" in pips:
                        for part in pips.split("/"):
                            if not part.isdigit() and part in COLORS:
                                encoding[COLORS.index(part)] += 0.5
                    elif pips in COLORS:
                        encoding[COLORS.index(pips)] += 1
                encoding[:len(COLORS)] = encoding[:len(COLORS)] / 8
            except Exception as e:
                logger.warning(f"Failed to parse mana cost '{mana_cost}': {e}")

        encoding[len(COLORS)] = min((converted_mana_cost or 0) / 16, 1.0)
        return encoding

    def _encode_types(self, types: List[str]) -> np.ndarray:
        encoding = np.zeros(len(TYPES), dtype=np.float32)
        for card_type in types:
            type_lower = card_type.lower()
            if type_lower in TYPES:
                encoding[TYPES.index(type_lower)] = 1
        return encoding

    def _encode_power_toughness(
        self,
        can_attack: bool,
        power: Optional[float] = None,
        toughness: Optional[float] = None,
    ) -> np.ndarray:
        encoding = np.zeros(3, dtype=np.float32)
        if can_attack:
            encoding[0] = 1
            if power is not None:
                encoding[1] = min(power / 15, 1.0)
            if toughness is not None:
                encoding[2] = min(toughness / 15, 1.0)
        return encoding

    def _oracle_text_input(self, oracle_text: str, subtypes: List[str]) -> str:
        if not oracle_text:
            return ''
        return oracle_text + " " + " ".join(subtypes)

    def _encode_oracle_text(self, oracle_text: str, subtypes: List[str]) -> np.ndarray:
        text_to_encode = self._oracle_text_input(oracle_text, subtypes)
        if not text_to_encode:
            return np.zeros(self.ORACLE_TEXT_DIM, dtype=np.float32)
        try:
            encoding = self.text_model.encode(text_to_encode, convert_to_numpy=True)
            return encoding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode oracle text, returning zeros: {e}")
            return np.zeros(self.ORACLE_TEXT_DIM, dtype=np.float32)
