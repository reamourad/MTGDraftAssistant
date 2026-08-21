import json
import re
from typing import Any, List, Union, Optional
import logging

import numpy as np
from numpy import ndarray, dtype
from sentence_transformers import SentenceTransformer
import torch


logger = logging.getLogger(__name__)


class CardEncoderError(Exception):
    """Raised when card encoding fails."""
    pass


class CardEncoder:
    """
    Encodes MTG cards into 407-dimensional feature vectors.
    
    Feature breakdown:
    - Rarity: 4 dims (one-hot)
    - Mana cost: 7 dims (6 colors + CMC)
    - Types: 9 dims (one-hot)
    - Power/Toughness: 3 dims (can_attack, power, toughness)
    - Oracle text: 384 dims (sentence embedding)
    Total: 407 dims
    """
    
    EXPECTED_DIM = 407
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        card_list: Optional[List[dict]] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the card encoder.
        
        Args:
            data_path: Path to JSON file containing card data
            card_list: List of card dictionaries (alternative to data_path)
            use_gpu: Whether to use GPU for text encoding (default: True)
        
        Raises:
            ValueError: If neither data_path nor card_list is provided
            CardEncoderError: If model loading fails
        """
        try:
            # Load sentence transformer model
            self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Set device
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            self.text_model.to(device)
            logger.info(f"CardEncoder initialized with device: {device}")
            
        except Exception as e:
            raise CardEncoderError(f"Failed to load sentence transformer model: {e}") from e

        # Load card data
        if data_path is not None:
            self.data_path = data_path
            try:
                with open(data_path, 'r') as f:
                    self.cards = json.load(f)
                logger.info(f"Loaded {len(self.cards)} cards from {data_path}")
            except Exception as e:
                raise CardEncoderError(f"Failed to load card data from {data_path}: {e}") from e
        elif card_list is not None:
            self.cards = card_list
            logger.info(f"Initialized with {len(card_list)} cards")
        else:
            raise ValueError("CardEncoder: data_path or card_list is required")

    def encode(self, card: dict) -> np.ndarray:
        """
        Encode a single card into a 407-dimensional feature vector.
        
        Args:
            card: Dictionary containing card data
        
        Returns:
            numpy array of shape (407,) with card features
        
        Raises:
            CardEncoderError: If encoding fails
        """
        try:
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

            # Flatten the vector into one vector
            full_vector = np.concatenate([
                rarity_vec,
                mana_vec,
                type_vec,
                pt_vec,
                text_vec
            ])
            
            # Validate output dimension
            if full_vector.shape[0] != self.EXPECTED_DIM:
                raise CardEncoderError(
                    f"Encoded vector has wrong dimension: {full_vector.shape[0]}, expected {self.EXPECTED_DIM}"
                )

            return full_vector
            
        except Exception as e:
            card_name = card.get('name', 'Unknown')
            raise CardEncoderError(f"Failed to encode card '{card_name}': {e}") from e
    
    def encode_batch(self, cards: List[dict]) -> np.ndarray:
        """
        Encode multiple cards efficiently.
        
        Args:
            cards: List of card dictionaries
        
        Returns:
            numpy array of shape (num_cards, 407) with card features
        
        Raises:
            CardEncoderError: If batch encoding fails
        """
        if not cards:
            return np.zeros((0, self.EXPECTED_DIM), dtype=np.float32)
        
        try:
            # Encode all cards
            encoded_cards = []
            for card in cards:
                encoded_cards.append(self.encode(card))
            
            # Stack into batch
            batch = np.stack(encoded_cards, axis=0)
            
            logger.debug(f"Encoded batch of {len(cards)} cards with shape {batch.shape}")
            return batch
            
        except Exception as e:
            raise CardEncoderError(f"Failed to encode batch of {len(cards)} cards: {e}") from e
    
    def encode_by_name(self, card_name: str) -> np.ndarray:
        """
        Encode a card by its name from the loaded card data.
        
        Args:
            card_name: Name of the card to encode
        
        Returns:
            numpy array of shape (407,) with card features
        
        Raises:
            CardEncoderError: If card not found or encoding fails
        """
        # Find card in loaded data
        card = None
        for c in self.cards:
            if c.get('name') == card_name:
                card = c
                break
        
        if card is None:
            raise CardEncoderError(f"Card '{card_name}' not found in loaded data")
        
        return self.encode(card)
    
    def encode_batch_by_names(self, card_names: List[str]) -> np.ndarray:
        """
        Encode multiple cards by their names.
        
        Args:
            card_names: List of card names
        
        Returns:
            numpy array of shape (num_cards, 407) with card features
        
        Raises:
            CardEncoderError: If any card not found or encoding fails
        """
        cards = []
        for name in card_names:
            card = None
            for c in self.cards:
                if c.get('name') == name:
                    card = c
                    break
            
            if card is None:
                raise CardEncoderError(f"Card '{name}' not found in loaded data")
            
            cards.append(card)
        
        return self.encode_batch(cards)


    # ONE HOT ENCODING
    def encode_rarity(self, rarity: str) -> np.ndarray:
        """
        Encode card rarity as one-hot vector.
        
        Args:
            rarity: Rarity string (common, uncommon, rare, mythic)
        
        Returns:
            numpy array of shape (4,) with one-hot encoding
        
        Raises:
            CardEncoderError: If rarity is invalid
        """
        RARITIES = ["common", "uncommon", "rare", "mythic"]
        encoding = np.zeros(len(RARITIES), dtype=np.float32)
        
        rarity_lower = rarity.lower()
        if rarity_lower not in RARITIES:
            logger.warning(f"Unknown rarity '{rarity}', defaulting to 'common'")
            rarity_lower = "common"
        
        encoding[RARITIES.index(rarity_lower)] = 1
        return encoding


    def encode_mana_cost(self, converted_mana_cost: float, mana_cost: str) -> np.ndarray:
        """
        Encode mana cost information.
        
        Args:
            converted_mana_cost: Total mana cost (CMC)
            mana_cost: Mana cost string (e.g., "{2}{U}{U}")
        
        Returns:
            numpy array of shape (7,) with color pips and CMC
        """
        COLORS = ["W", "B", "U", "R", "G", "P"]
        encoding = np.zeros(len(COLORS) + 1, dtype=np.float32)

        if mana_cost:
            try:
                # Use regex to extract mana symbols
                separated_colors = re.findall(r"{(.*?)}", mana_cost)
                for pips in separated_colors:
                    # Handle hybrid mana
                    if "/" in pips:
                        parts = pips.split("/")
                        for part in parts:
                            if not part.isdigit() and part in COLORS:
                                encoding[COLORS.index(part)] += 0.5
                    elif pips in COLORS:
                        encoding[COLORS.index(pips)] += 1

                # Normalize color pips
                encoding[:len(COLORS)] = encoding[:len(COLORS)] / 8
            except Exception as e:
                logger.warning(f"Failed to parse mana cost '{mana_cost}': {e}")
        
        # Add converted mana cost (normalized)
        encoding[len(COLORS)] = min(converted_mana_cost / 16, 1.0)
        
        return encoding


    def encode_types(self, types: List[str]) -> np.ndarray:
        """
        Encode card types as multi-hot vector.
        
        Args:
            types: List of card types
        
        Returns:
            numpy array of shape (9,) with multi-hot encoding
        """
        TYPES = ["artifact", "battle", "creature", "enchantment", "instant", "land", "planeswalker", "sorcery", "kindred"]
        encoding = np.zeros(len(TYPES), dtype=np.float32)

        for card_type in types:
            type_lower = card_type.lower()
            if type_lower in TYPES:
                encoding[TYPES.index(type_lower)] = 1

        return encoding


    def encode_power_toughhness(
        self,
        can_attack: bool,
        power: Optional[Union[int, str]] = None,
        toughness: Optional[Union[int, str]] = None
    ) -> np.ndarray:
        """
        Encode power/toughness information.
        
        Args:
            can_attack: Whether the card can attack
            power: Power value (or None)
            toughness: Toughness value (or None)
        
        Returns:
            numpy array of shape (3,) with [can_attack, power, toughness]
        """
        encoding = np.zeros(3, dtype=np.float32)

        if can_attack:
            encoding[0] = 1

            # Handle power
            if power is not None:
                try:
                    power_val = float(power) if isinstance(power, (int, float)) else 0
                    encoding[1] = min(power_val / 15, 1.0)
                except (ValueError, TypeError):
                    encoding[1] = 0

            # Handle toughness
            if toughness is not None:
                try:
                    toughness_val = float(toughness) if isinstance(toughness, (int, float)) else 0
                    encoding[2] = min(toughness_val / 15, 1.0)
                except (ValueError, TypeError):
                    encoding[2] = 0
        
        return encoding


    def encode_oracle_text(self, oracle_text: str, subtypes: List[str]) -> np.ndarray:
        """
        Encode oracle text and subtypes using sentence transformer.
        
        Args:
            oracle_text: Card's oracle text
            subtypes: List of card subtypes
        
        Returns:
            numpy array of shape (384,) with text embedding
        """
        if not oracle_text:
            return np.zeros(384, dtype=np.float32)

        try:
            # Combine oracle text with subtypes
            text_to_encode = oracle_text + " " + " ".join(subtypes)
            encoding = self.text_model.encode(text_to_encode, convert_to_numpy=True)
            
            # Ensure correct dtype
            return encoding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode oracle text, returning zeros: {e}")
            return np.zeros(384, dtype=np.float32)